# import pandas as pd

# files = [
#     (r"src\toxicity\data\train_model_input.csv", r"src\toxicity\data\train_clean.csv"),
#     (r"src\toxicity\data\validation_model_input.csv", r"src\toxicity\data\val_clean.csv"),
#     (r"src\toxicity\data\test_model_input.csv", r"src\toxicity\data\test_clean.csv")
# ]

# for in_path, out_path in files:
#     df = pd.read_csv(in_path)
#     df = df[['smiles', 'label']]
#     df.to_csv(out_path, index=False)
#     print(f"Saved cleaned file to: {out_path}")
