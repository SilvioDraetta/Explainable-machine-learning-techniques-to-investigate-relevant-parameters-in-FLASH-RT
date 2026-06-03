import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_excel, save_excel
from src.preprocessing.split_data import split_dataset

df = load_excel("data/processed/processed_elec-prot/elec-prot.xlsx")

df_train, df_val = split_dataset(df)

save_excel(df_train, "data/processed/processed_elec-prot/elec-prot_training.xlsx")
save_excel(df_val, "data/processed/processed_elec-prot/elec-prot_validation.xlsx")
