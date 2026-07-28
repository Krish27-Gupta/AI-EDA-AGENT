
import os
import pandas as pd


def read_uploaded_file(file_path):
    # Extract the file extension (in lowercase)
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Read the file based on its extension
    if file_extension == ".csv":
        # You can add parameters like sep=',' if needed
        df = pd.read_csv(file_path)

    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)

    elif file_extension == ".json":
        df = pd.read_json(file_path)

    elif file_extension == ".parquet":
        df = pd.read_parquet(file_path)

    elif file_extension in [".txt", ".tsv"]:
        # Assuming tab-separated for .txt/.tsv, adjust if necessary
        df = pd.read_csv(file_path, sep="\t")

    elif file_extension == ".pickle" or file_extension == ".pkl":
        df = pd.read_pickle(file_path)

    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return df


# --- Example Usage ---
# df = read_uploaded_file("path/to/your/uploaded_file.xlsx")
# print(df.head())
