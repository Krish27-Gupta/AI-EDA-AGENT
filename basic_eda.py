
import pandas as pd
import numpy as np


def perform_eda(df: pd.DataFrame):
    """Performs comprehensive basic Exploratory Data Analysis (EDA) on a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    """
    print("=" * 60)
    print("🚀 BASIC EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Dataset Shape
    print("\n[1] DATASET SHAPE")
    print(f"Rows:    {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 2. Column Names & Data Types
    print("\n[2] COLUMNS & DATA TYPES")
    dtypes_df = pd.DataFrame(
        {"Data Type": df.dtypes, "Non-Null Count": df.notnull().sum()}
    )
    print(dtypes_df)

    # 3. Missing Values Summary
    print("\n[3] MISSING VALUES SUMMARY")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame(
        {"Missing Values": missing_count, "Percentage (%)": missing_percent}
    )
    # Filter to show only columns with missing values, or message if none
    if missing_count.sum() > 0:
        print(missing_df[missing_df["Missing Values"] > 0])
    else:
        print("🎉 Great news! There are no missing values in this dataset.")

    # 4. Duplicate Rows
    print("\n[4] DUPLICATE ROWS")
    duplicates = df.duplicated().sum()
    print(
        f"Number of duplicate rows: {duplicates} ({round((duplicates / len(df)) * 100, 2)}% of total rows)"
    )

    # 5. Statistical Summary (Numerical Columns)
    print("\n[5] STATISTICAL SUMMARY (Numerical Columns)")
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        print(df.describe().T)
    else:
        print("No numerical columns found in the dataset.")

    # 6. Statistical Summary (Categorical Columns)
    print("\n[6] STATISTICAL SUMMARY (Categorical Columns)")
    cat_cols = df.select_dtypes(include=["object", "category"])
    if not cat_cols.empty:
        print(df.describe(include=["object", "category"]).T)
    else:
        print("No categorical columns found in the dataset.")

    print("\n" + "=" * 60)
    print("✨ EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)
