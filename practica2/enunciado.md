Empleando las técnicas de forzado de gramática, implementad por parejas una utilidad para extraer información relevante de alguna de las fuentes de datos clínicos presentes en el dataset ClinText-SP.

ClinText-SP:
https://huggingface.co/datasets/IIC/ClinText-SP/viewer

Estos ficheros están en formato parquet, un formato diseñado para BigData. 

Ejemplo de carga de ficheros .parquet (filtrando por columna de fuente de datos):

import pandas as pd

df = pd.read_parquet("data.parquet") 
filtered = df[df["source"] == "wikidisease"]
print(filtered)

Necesitaréis tener instalados:

    pandas
    fastparquet

!pip install pandas
!pip install fastparquet