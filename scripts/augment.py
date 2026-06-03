import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.utils.io import load_excel, save_excel
from src.preprocessing.augment_data import augment_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="Number of augmented variants per sample")
    args = parser.parse_args()

    # File di input fisso
    input_path = "data/processed/processed_elec-prot/elec-prot_training.xlsx"

    # File di output generato automaticamente
    output_path = f"data/processed/processed_elec-prot/augmented_{args.N}.xlsx"

    # Carica dataset
    df = load_excel(input_path)

    # Augment
    df_aug = augment_dataset(df, num_variants=args.N)

    # Salva
    save_excel(df_aug, output_path)

    print(f"Augmentation completed. Saved in: {output_path}")

if __name__ == "__main__":
    main()
