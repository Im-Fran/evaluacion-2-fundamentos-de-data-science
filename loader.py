import requests
import pandas as pd
import numpy as np
from config import archivo_dataset_mp25, archivo_humedad_relativa, archivo_temperatura, archivo_viento, archivo_procesado

def descargar_mp25():
    print('Descargando mediciones MP2.5 21-10-2003 > 21-10-2025')
    # Verificar si el archivo ya existe y tiene contenido
    if archivo_dataset_mp25.exists() and archivo_dataset_mp25.stat().st_size > 0:
        print(f"El archivo ya existe y no está vacío: {archivo_dataset_mp25}")
        print("Omitiendo descarga.")
        return archivo_dataset_mp25

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
        "&from=031021&to=251021"
        "&path=%2Fusr%2Fairviro%2Fdata%2FCONAMA%2F"
        "&lang=esp&rsrc&macropath"
    )

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Guardar el archivo a medida que se descarga
    with open(archivo_dataset_mp25, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Descarga completa: {archivo_dataset_mp25}")
    return None

def cargar_mp25():
    """
    Genera un DataFrame del dataset de MP2.5,
    Si el archivo no existe o está vacío, lo descarga.
    """
    if not (archivo_dataset_mp25.exists() and archivo_dataset_mp25.stat().st_size > 0):
        descargar_mp25()

    columnas = ["Fecha","Hora","Registros_Validados","Registros_Preliminares","Registros_No_Validados","Desconocido"]
    df = pd.read_csv(
        archivo_dataset_mp25,
        sep=";",          # separador correcto
        header=None,      # ignorar encabezados del archivo
        skiprows=1,       # saltar la primera fila (encabezado incompleto)
        names=columnas,   # asignar nuestros encabezados
        dtype=str         # mantener formato de texto (por ejemplo fechas 031021)
    )

    df = df.drop(columns=["Desconocido"], errors="ignore") # Elimina la columna desconocida
    df = df.drop(columns=["Hora"], errors="ignore") # Elimina la columna Hora ya que siempre es 0000
    df = df.dropna(how="all") # Limpiar espacios o valores nulos
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%y%m%d", errors="coerce")
    df["Registros_Validados"] = pd.to_numeric(df["Registros_Validados"], errors="coerce")
    df["Registros_Preliminares"] = pd.to_numeric(df["Registros_Preliminares"], errors="coerce")
    df["MP2.5"] = df["Registros_Validados"].combine_first(df["Registros_Preliminares"])
    df = df.drop(columns=["Registros_Validados", "Registros_Preliminares", "Registros_No_Validados"], errors="ignore")
    df["MP2.5"] = pd.to_numeric(df["MP2.5"], errors="coerce").round(2)

    return df

def descargar_humedad_relativa():
    print("Descargando humedad relativa 21-10-2003 > 21-10-2025")
    if archivo_humedad_relativa.exists() and archivo_humedad_relativa.stat().st_size > 0:
        print(f"El archivo {archivo_humedad_relativa} ya existe")
        return archivo_humedad_relativa

    url = "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&macro=./RM/D14/Met/RHUM//horario_003.ic&from=031021&to=251021&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
    response = requests.get(url)
    with open(archivo_humedad_relativa, "w") as archivo:
        archivo.write(response.text)

    print(f"Descarga completa: {archivo_humedad_relativa}")
    return archivo_humedad_relativa

def cargar_humedad_relativa():
    if not(archivo_humedad_relativa.exists() and archivo_humedad_relativa.stat().st_size > 0):
        descargar_humedad_relativa()

    columnas = ["Fecha", "Hora", "Humedad_Relativa", "Desconocido"]
    df = pd.read_csv(archivo_humedad_relativa, sep=";", header=None, skiprows=1, names=columnas, dtype=str)
    df = df[df["Hora"] == "0100"]
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%y%m%d", errors="coerce")
    df["Humedad_Relativa"] = pd.to_numeric(df["Humedad_Relativa"].str.replace(",", "."), errors="coerce").round(2)
    df = df.drop(columns=["Hora", "Desconocido"], errors="ignore")

    return df

def descargar_temperatura():
    print("Descargando temperatura 21-10-2003 > 21-10-2025")
    if archivo_temperatura.exists() and archivo_temperatura.stat().st_size > 0:
        print(f"El archivo {archivo_temperatura} ya existe")
        return archivo_temperatura

    url = "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&macro=./RM/D14/Met/TEMP//horario_003.ic&from=031021&to=251021&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
    response = requests.get(url)
    with open(archivo_temperatura, "w") as archivo:
        archivo.write(response.text)

    print(f"Descarga completa: {archivo_temperatura}")
    return archivo_temperatura

def cargar_temperatura():
    if not(archivo_temperatura.exists() and archivo_temperatura.stat().st_size > 0):
        descargar_temperatura()

    columnas = ["Fecha", "Hora", "Temperatura", "Desconocido"]
    df = pd.read_csv(archivo_temperatura, sep=";", header=None, skiprows=1, names=columnas, dtype=str)
    df = df[df["Hora"] == "0100"]
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%y%m%d", errors="coerce")
    df["Temperatura"] = pd.to_numeric(df["Temperatura"].str.replace(",", "."), errors="coerce").round(2)
    df = df.drop(columns=["Hora", "Desconocido"], errors="ignore")
    return df

def descargar_viento():
    print("Descargando datos de viento 21-10-2003 > 21-10-2025")
    if archivo_viento.exists() and archivo_viento.stat().st_size > 0:
        print(f"El archivo {archivo_viento} ya existe")
        return archivo_viento

    url = "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&macro=./RM/D14/Met/WSPD//horario_010.ic&from=031021&to=251021&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
    response = requests.get(url)
    with open(archivo_viento, "w") as archivo:
        archivo.write(response.text)

    print(f"Descarga completa: {archivo_viento}")
    return archivo_viento

def cargar_viento():
    if not(archivo_viento.exists() and archivo_viento.stat().st_size > 0):
        descargar_viento()

    columnas = ["Fecha", "Hora", "Viento", "Desconocido"]
    df = pd.read_csv(archivo_viento, sep=";", header=None, skiprows=1, names=columnas, dtype=str)
    df = df[df["Hora"] == "0100"]
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%y%m%d", errors="coerce")
    df["Viento"] = pd.to_numeric(df["Viento"].str.replace(",", "."), errors="coerce").round(2)
    df = df.drop(columns=["Hora", "Desconocido"], errors="ignore")
    return df

def prepara_df():
    if archivo_procesado.exists() and archivo_procesado.stat().st_size > 0:
        print(f"Cargando dataset procesado desde {archivo_procesado}")
        df = pd.read_csv(archivo_procesado, parse_dates=["Fecha"])
        return df

    df_mp25 = cargar_mp25()
    df_humedad_relativa = cargar_humedad_relativa()
    df_temperatura = cargar_temperatura()
    df_viento = cargar_viento()

    df = pd.merge(df_mp25, df_humedad_relativa, on="Fecha", how="inner")
    df = pd.merge(df, df_temperatura, on="Fecha", how="inner")
    df = pd.merge(df, df_viento, on="Fecha", how="inner")
    df = df.dropna()

    # Guarda el df final para futuras referencias
    df.to_csv(archivo_procesado, index=False)
    return df