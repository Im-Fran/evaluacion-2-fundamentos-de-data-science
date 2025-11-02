from loader import descargar_mp25, descargar_humedad_relativa
from loader import prepara_df
import numpy as np

def main():
    descargar_mp25() # Descarga los datasets necesarios para el trabajo
    descargar_humedad_relativa()

    df = prepara_df()
    df["Calidad_Aire"] = np.where(df["MP2.5"] < 25, "Buena", "Mala")
    print(df.head())

if __name__ == '__main__':
    main()