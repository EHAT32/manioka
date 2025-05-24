from fastapi import FastAPI, File, UploadFile, Query
from PIL import Image
import base64
import io
import pandas as pd 
import numpy as np
import asyncio
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from infer_triton import InferenceModule  # Импортируем ваш модуль инференса
import os
import uuid
import shutil
import zipfile
import asyncio


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_module = InferenceModule()  # Создаем экземпляр модуля инференса

@app.post("/predict/", description="Выполняет классификацию изображения с использованием указанной модели.")
async def predict(
    zip_file: UploadFile = File(..., description="Архив (zip), содержащий папки с изображениями."), 
    csv_file: UploadFile = File(..., description="CSV-файл, содержащий мета-данные о растениях"),
    model_name: str = Query(..., description="Имя модели тритона для использования")
):
    """
    Выполнить классификацию изображения.

    Args:
        file (UploadFile): Загружаемое изображение.
        model_name (str): Имя модели для использования в инференсе.

    Returns:
        dict: Таблица с полученными предсказаниями.
    """
    try:
        # Конвертация загруженного файла в base64
        # contents = await file.read()
        # img_base64 = base64.b64encode(contents).decode("utf-8")
        temp_dir = f"temp_data"
        os.makedirs(temp_dir, exist_ok=True)

        # Сохраняем архив во временное место
        archive_path = f"{temp_dir}/{zip_file.filename}"

        with open(archive_path, "wb") as buffer:
            shutil.copyfileobj(zip_file.file, buffer)

        # Распаковываем архив
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail=f"Файл не является корректным ZIP-архивом.")
        print("Free")
        test_data_path = "temp_data/test"
        test_df = pd.read_csv("Test.csv")

        output_root = "merged_images"
        output_test = os.path.join(output_root)

        os.makedirs(output_test, exist_ok=True)

        new_test_df = generate_regression_dataset(test_df, output_test, test_data_path, type = "test")

        new_test_df['ImageBase64'] = new_test_df['ImageSegments'].apply(encode_image_to_base64)
        new_test_df = await infer_all_images(new_test_df, model_name)
        print(f"new_test_df")
        new_test_df = new_test_df.drop(columns=["ImageBase64", "WasSegmented", "Side", "ImageSegments", "Width"])
        result = new_test_df.to_dict(orient="records")
        return {
            "predictions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def infer_all_images(df, model_name):
    async def infer_row(row):
        return await inference_module.infer_image(row['ImageBase64'], model_name=model_name)

    tasks = [infer_row(row) for _, row in df.iterrows()]
    predictions = await asyncio.gather(*tasks)
    df['Preds'] = [float(arr[0]) for arr in predictions]
    print(f"predictions")
    print(df)
    return df

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # удалить файл или символическую ссылку
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # удалить папку рекурсивно
        except Exception as e:
            print(f'Ошибка при удалении {file_path}: {e}')

def encode_image_to_base64(file_path):
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        return None 

def generate_regression_dataset(df: pd.DataFrame, output: str, src: str, type: str, do_segment: bool = False):
    paths = []
    was_segmented = []
    widths = []
    for _, row in df.iterrows():
        img, segm = merge_segmented_images(
            root=src,
            folder=row["FolderName"],
            side=row["Side"],
            start=row["Start"],
            end=row["End"],
            do_segment=do_segment
        )
        img_path = os.path.join(output, f'{row["ID"]}.png')
        img.save(img_path)
        paths.append(img_path)
        was_segmented.append(segm)
        widths.append(img.width)
    
    df["WasSegmented"] = was_segmented
    df["ImageSegments"] = paths
    df["Width"] = widths
    df.to_csv(f"{type}CNN.csv")
    return df

def merge_segmented_images(root: str, folder: str, side: str, start: int, end: int, do_segment = True) -> Image:
    """Компоновка сегментов по серии срезов

    Args:
        root (str): корневой путь
        folder (str): папка корня
        side (str): (L|R) сторона
        start (int): начальный срез
        end (int): конечный срез

    Returns:
        Image: Итоговое изображение PIL.Image
    """
    images_in_range = get_images(root, folder, side, start, end)
    segmented_images, was_segmented = get_segmented_images(images_in_range, do_segment=do_segment)
    total_width = sum(img.width for img in segmented_images)
    max_height = max(img.height for img in segmented_images)
    res = Image.new("RGBA", (total_width, max_height * len(segmented_images)), (0, 0, 0, 0))
    sqr_width = int(np.ceil(np.sqrt(total_width * max_height)))
    x_offset = 0
    y_offset = 0
    actual_width = 0
    for segment in segmented_images:
        if x_offset + segment.width > sqr_width:
            actual_width = max(actual_width, x_offset)
            x_offset = 0
            y_offset += max_height
        res.paste(segment, (x_offset, y_offset))
        x_offset += segment.width
    actual_width = max(actual_width, x_offset)
    actual_height = y_offset + max_height
    res = res.crop((0, 0, actual_width, actual_height))
    return res, was_segmented

def get_segmented_images(image_paths, display_image=False, do_segment = True):
    if not do_segment:
        return [Image.open(img) for img in image_paths], False
    return [], False

def get_images(img_root : str, folder : str, side : str, start : int, end : int) -> list[str]:
    """Получение списка последовательности изображений

    Args:
        img_root (str): корневая папка с изображениями
        folder (str): папка корня
        side (str): (L|R) сторона
        start (int): начальный срез
        end (int): конечный срез

    Returns:
        list[str]: список путей до изображений
    """
    images = []
    for i in range(start, end + 1):
        path = os.path.join(
            img_root,
            folder,
            f"{folder}_{side}_{i:03d}.png"
        )
        if os.path.exists(path):
            images.append(path)
    return images