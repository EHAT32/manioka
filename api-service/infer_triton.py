import os
import numpy as np
from PIL import Image
from torchvision import transforms
import grpc
import tritonclient.grpc.aio as grpcclient
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype
import base64
import random
from io import BytesIO


class InferenceModule:
    def __init__(self) -> None:
        """Initialize."""
        self.url = os.environ.get("TRITON_SERVER_URL", "127.0.0.1:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=self.url)

    async def infer_image(
        self,
        img: str,
        model_name: str = "cnn_model",
    ) -> dict:
        """
        Perform inference on the input image.

        Args:
            img (str): Base64 encoded image string.
            model_name (str): Name of the deployed model.

        Returns:
            dict: class ID, and logit.
        """
        model_meta, model_config = self.parse_model_metadata(model_name)
        shape = model_meta.inputs[0].shape
        channels, height, width = shape[1:]
        dtype = model_meta.inputs[0].datatype
        # Preprocess the image
        img = self.preprocess_image_torchvision(img)  # или preprocess_image_pil
        # Create input tensor for Triton
        inputs = [grpcclient.InferInput(model_meta.inputs[0].name, [1, channels, height, width], dtype)]
        inputs[0].set_data_from_numpy(img.astype(triton_to_np_dtype(dtype)))

        # Define output tensor
        outputs = [grpcclient.InferRequestedOutput(model_meta.outputs[0].name)]

        # Perform inference
        results = await self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        # Process the output
        output = results.as_numpy(model_meta.outputs[0].name)[0]

        return output

    def parse_model_metadata(self, model_name: str) -> object:
        """
        Parse metadata and configuration of the model.

        Args:
            model_name (str): Name of the deployed model.

        Returns:
            tuple: Metadata and configuration of the model.
        """
        channel = grpc.insecure_channel(self.url)
        grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
        metadata_request = service_pb2.ModelMetadataRequest(name=model_name)
        metadata_response = grpc_stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(name=model_name)
        config_response = grpc_stub.ModelConfig(config_request)

        return metadata_response, config_response

    
    @staticmethod
    def center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
        """
        Центрированная обрезка изображения до указанного размера.

        Args:
            img (Image.Image): Исходное изображение.
            width (int): Требуемая ширина.
            height (int): Требуемая высота.

        Returns:
            Image.Image: Обрезанное изображение.
        """
        w, h = img.size  # Текущие размеры изображения
        left = (w - width) / 2
        top = (h - height) / 2
        right = (w + width) / 2
        bottom = (h + height) / 2

        return img.crop((left, top, right, bottom))

    @staticmethod
    def decode_img(img_base64: str) -> Image.Image:
        """
        Декодирует base64-строку в объект изображения PIL.

        Args:
            img_base64 (str): Base64-кодированная строка изображения.

        Returns:
            Image.Image: Объект изображения PIL.
        """
        # Удаляем префикс data URL, если он есть
        if img_base64.startswith("data:image/"):
            header, encoded = img_base64.split(",", 1)
        else:
            encoded = img_base64

        # Декодируем base64 в байты
        img_bytes = base64.b64decode(encoded)

        # Создаем объект BytesIO для чтения байтов
        img_buffer = BytesIO(img_bytes)

        # Открываем изображение с помощью PIL
        return Image.open(img_buffer).convert("RGB")

    
    def preprocess_image_torchvision(
        self,
        img: str,
    ) -> np.ndarray:
        """
        Preprocess the input image for ResNet using torchvision.transforms.

        Args:
            img (str): Base64 encoded image string.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        """
        # Decode base64 image to PIL Image
        pil_img = self.decode_img(img)

        # Define preprocessing pipeline using torchvision.transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        # Apply preprocessing pipeline
        pil_img = self.preprocess(pil_img)
        # Center crop 224x224
        pil_img = self.center_crop(pil_img, 128, 128) 

        if random.random() < 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT) 

        # Convert to NumPy array, scale to [0,1]
        np_img = np.array(pil_img).astype(np.float32) / 255.0

        normalized_img = (np_img - 0.5) / 0.5

        # Transpose to match NCHW format
        ordered_img = np.transpose(normalized_img, (2, 0, 1))
        # Add batch dimension
        batched_img = np.expand_dims(ordered_img, axis=0)

        return batched_img