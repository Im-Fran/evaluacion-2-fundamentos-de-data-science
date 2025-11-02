from pathlib import Path

datasets_dir = Path('datasets')
datasets_dir.mkdir(parents=True, exist_ok=True)

out_dir = Path('out')
out_dir.mkdir(parents=True, exist_ok=True)

mp2_5_dataset_file = datasets_dir / "mediciones_mp2.5_21102022-21102025.csv"