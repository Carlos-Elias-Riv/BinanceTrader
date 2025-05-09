import streamlit as st
import pandas as pd
import boto3
import os
from io import StringIO

# Configura tu bucket y prefijo
BUCKET_NAME = "mi-bucket"
PREFIX = "predicciones/"

# Accede a S3
s3 = boto3.client('s3')

# Lista los archivos CSV disponibles
objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
csv_files = [obj["Key"] for obj in objects.get("Contents", []) if obj["Key"].endswith(".csv")]

# Lee los archivos y combínalos
dataframes = []
for key in csv_files[-5:]:  # Solo los últimos 5 por simplicidad
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    body = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(body))
    dataframes.append(df)

# Concatenar todos
if dataframes:
    final_df = pd.concat(dataframes, ignore_index=True)
    st.title("📈 Dashboard de Predicción Cripto")
    st.dataframe(final_df)

    st.line_chart(final_df[['prediction']])
else:
    st.warning("No se encontraron datos en S3.")
