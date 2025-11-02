from pathlib import Path

datasets_dir = Path('datasets')
datasets_dir.mkdir(parents=True, exist_ok=True)

out_dir = Path('out')
out_dir.mkdir(parents=True, exist_ok=True)

archivo_dataset_mp25 = datasets_dir / "mediciones_mp2.5.csv"
archivo_humedad_relativa = datasets_dir / "humedad_relativa.csv"
archivo_temperatura = datasets_dir / "temperatura.csv"
archivo_viento = datasets_dir / "viento.csv"

archivo_procesado = out_dir / "dataset_procesado.csv"