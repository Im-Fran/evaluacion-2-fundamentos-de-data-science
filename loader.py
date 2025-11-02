import requests
import pandas as pd
from config import mp2_5_dataset_file

def descargar_datasets():
    print('Descargando mediciones MP2.5 21-10-2022 > 21-10-2025')
    # Verificar si el archivo ya existe y tiene contenido
    if mp2_5_dataset_file.exists() and mp2_5_dataset_file.stat().st_size > 0:
        print(f"El archivo ya existe y no está vacío: {mp2_5_dataset_file}")
        print("Omitiendo descarga.")
        return mp2_5_dataset_file

    # Descripción columnas:
    # C1 - Fecha
    # C2 - Hora
    # C3 - Registros Validados
    # C4 - Registros Preliminares
    # C5 - Registros No Validados
    # C6 - ??
    url = (
        "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi"
        "?outtype=xcl"
        "&macro=.%2FRM%2FD14%2FCal%2FPM25%2F%2FPM25.diario.diario.ic"
        "&from=221021&to=251021"
        "&path=%2Fusr%2Fairviro%2Fdata%2FCONAMA%2F"
        "&lang=esp&rsrc&macropath"
    )

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Guardar el archivo a medida que se descarga
    with open(mp2_5_dataset_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Descarga completa: {mp2_5_dataset_file}")
    return None


def cargar_df_mp2_5():
    """
    Genera un DataFrame del dataset de MP2.5,
    Si el archivo no existe o está vacío, lo descarga.
    """
    if not (mp2_5_dataset_file.exists() and mp2_5_dataset_file.stat().st_size > 0):
        descargar_datasets()

    columnas = ["Fecha","Hora","Registros_Validados","Registros_Preliminares","Registros_No_Validados","Desconocido"]
    df = pd.read_csv(
        mp2_5_dataset_file,
        sep=";",          # separador correcto
        header=None,      # ignorar encabezados del archivo
        skiprows=1,       # saltar la primera fila (encabezado incompleto)
        names=columnas,   # asignar nuestros encabezados
        dtype=str         # mantener formato de texto (por ejemplo fechas 221021)
    )

    df = df.drop(columns=["Desconocido"], errors="ignore") # Elimina la columna desconocida
    df = df.drop(columns=["Hora"], errors="ignore") # Elimina la columna Hora ya que siempre es 0000
    df = df.dropna(how="all") # Limpiar espacios o valores nulos
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%y%m%d", errors="coerce")
    for col in ["Registros_Validados","Registros_Preliminares","Registros_No_Validados"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df