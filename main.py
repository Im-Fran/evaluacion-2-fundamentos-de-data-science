from loader import descargar_datasets
from loader import cargar_df_mp2_5

def main():
    descargar_datasets() # Descarga los datasets necesarios para el trabajo

    df = cargar_df_mp2_5()  # Carga el DF de las mediciones de MP2.5
    print(df)


if __name__ == '__main__':
    main()