
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda_by_ai(df):
    # Agent-generated advanced EDA code goes here
    # Set aesthetic style
  sns.set_theme(style='whitegrid')
  plt.rcParams['figure.figsize'] = (12, 6)

  # ==========================================
  # 2. DATA INITIAL INSPECTION
  # ==========================================
  print('--- DATA INFO ---')
  print(df.info())

  print('\n--- MISSING VALUES ---')
  print(df.isnull().sum())

  print('\n--- NUMERICAL DESCRIBE ---')
  print(df.describe())

  print('\n--- OBJECT/CATEGORICAL DESCRIBE ---')
  print(df.describe(include=['O']))

  # Correlation Matrix
  plt.figure(figsize=(8, 6))
  numeric_df = df.select_dtypes(include=[np.number])
  if not numeric_df.empty:
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Columns')
    plt.show()

  # ==========================================
  # 3. UNIVARIATE ANALYSIS
  # ==========================================
  # Numerical: Sales/First Numeric Distribution & Outliers
  numerical_cols = df.select_dtypes(include=[np.number]).columns
  if len(numerical_cols) > 0:
    target_num = numerical_cols[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[target_num], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'{target_num} Distribution & KDE')

    sns.boxplot(x=df[target_num], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'{target_num} Boxplot (Outlier Detection)')
    plt.tight_layout()
    plt.show()

    print(f'{target_num} Skewness: {df[target_num].skew():.2f}')
    print(f'{target_num} Kurtosis: {df[target_num].kurt():.2f}')

  # Object Column: Region/First Categorical Frequency
  categorical_cols = df.select_dtypes(include=['O']).columns
  if len(categorical_cols) > 0:
    target_cat = categorical_cols[0]
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df,
        x=target_cat,
        order=df[target_cat].value_counts().index,
        palette='viridis',
    )
    plt.title(f'Frequency Count of {target_cat}')
    plt.xticks(rotation=45)
    plt.show()

  # ==========================================
  # 4. BIVARIATE ANALYSIS
  # ==========================================
  if len(categorical_cols) > 0 and len(numerical_cols) > 0:
    plt.figure(figsize=(10, 5))
    cat_metric = (
        df.groupby(categorical_cols[0])[numerical_cols[0]]
        .sum()
        .reset_index()
        .sort_values(by=numerical_cols[0], ascending=False)
    )
    sns.barplot(
        data=cat_metric, x=categorical_cols[0], y=numerical_cols[0], palette='magma'
    )
    plt.title(f'Total {numerical_cols[0]} by {categorical_cols[0]}')
    plt.xticks(rotation=45)
    plt.show()

  # ==========================================
  # 5. MULTIVARIATE ANALYSIS (Bar Plot with Hue)
  # ==========================================
  if len(categorical_cols) >= 2 and len(numerical_cols) > 0:
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df,
        x=categorical_cols[0],
        y=numerical_cols[0],
        hue=categorical_cols[1],
        estimator=np.sum,
        ci=None,
        palette='Set2',
    )
    plt.title(
        f'Total {numerical_cols[0]} by {categorical_cols[0]} and {categorical_cols[1]} (Hue)'
    )
    plt.ylabel(f'Total {numerical_cols[0]}')
    plt.xlabel(categorical_cols[0])
    plt.xticks(rotation=45)
    plt.legend(title=categorical_cols[1])

    # Adding data labels on top of bars
    for p in ax.patches:
      height = p.get_height()
      if height > 0:
        ax.annotate(
            f'{height:,.0f}',
            (p.get_x() + p.get_width() / 2.0, height),
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            xytext=(0, 3),
            textcoords='offset points',
        )

    plt.tight_layout()
    plt.show()

  # ==========================================
  # 6. TIME SERIES ANALYSIS
  # ==========================================
  date_cols = [
      col
      for col in df.columns
      if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])
  ]
  if len(date_cols) > 0 and len(numerical_cols) > 0:
    date_col = date_cols[0]
    # Ensure datetime format
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Prepare time series dataframe
    ts_df = (
        df.set_index(date_col)
        .resample('M')[numerical_cols[0]]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 5))
    plt.plot(
        ts_df[date_col],
        ts_df[numerical_cols[0]],
        marker='o',
        linestyle='-',
        color='b',
    )
    plt.title(f'Monthly {numerical_cols[0]} Trend Over Time')
    plt.xlabel(date_col)
    plt.ylabel(f'Total {numerical_cols[0]}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

