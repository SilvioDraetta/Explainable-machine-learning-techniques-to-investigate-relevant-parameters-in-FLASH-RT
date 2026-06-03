import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_excel
from src.preprocessing.transform_prezado import transform_prezado
from src.preprocessing.transform_toschini import transform_toschini
from src.preprocessing.merge_datasets import merge_datasets
from src.preprocessing.scale_data import scale_dataset

df1 = load_excel("data/raw/electrons-protons/PREZADO-MICE-FLASH-INVIVO-NORMAL.xlsx")
df2 = load_excel("data/raw/electrons-protons/TOSCHINI-GUT-FLASH-NORMAL.xlsx")

prezado = transform_prezado(df1)
toschini = transform_toschini(df2)

final_df = merge_datasets(prezado, toschini)

final_df_scaled = scale_dataset(final_df)

final_df_scaled.to_excel("data/processed/processed_elec-prot/elec-prot.xlsx", index=False)
