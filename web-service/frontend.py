import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import requests

def extract_zip(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

st.title("Определение размеров корней маниоки")

api_url = os.getenv("API_URL", "http://localhost:5000")
uploaded_zip = st.file_uploader("Загрузите архив с изображениями корней", type=["zip"])
uploaded_csv = st.file_uploader("Загрузите CSV конфиг по растениям", type=["csv"])


if uploaded_zip and uploaded_csv:
    if st.button("Рассчитать"):
        with st.spinner("Обрабатываем данные..."):
            # Временные файлы
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(uploaded_zip.read())
                zip_path = tmp_zip.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                tmp_csv.write(uploaded_csv.read())
                csv_path = tmp_csv.name

            try:
                with open(zip_path, 'rb') as zf, open(csv_path, 'rb') as cf:
                    files = {
                        "zip_file": zf,
                        "csv_file": cf
                    }
                    response = requests.post(f"{api_url}/predict", files=files)

                if response.status_code == 200:
                    result_df = pd.DataFrame(response.json())
                    st.success("Анализ завершён!")
                    st.dataframe(result_df)
                else:
                    st.error(f"Ошибка при обработке запроса: {response.status_code}")
                    st.text(response.text)
            finally:
                os.remove(zip_path)
                os.remove(csv_path)