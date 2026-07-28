# ==============================================================================
# ADVANCED DATA ANALYSIS SCRIPT (`advance_eda.py`)
# ==============================================================================
# Setup & Installation:
# Run this in your terminal if required modules are missing:
# pip install pandas numpy matplotlib seaborn scikit-learn
# ==============================================================================

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def eda_by_ai():
    """Performs an advanced, comprehensive Exploratory Data Analysis (EDA)

    on a pre-loaded pandas DataFrame named `df`.

    This function handles:
      1. Setup & basic dataset description (shape, types, missing values, duplicates).
      2. Correlation analysis (matrix calculation & heatmap visualization).
      3. Univariate analysis (histograms, KDE plots, and value count bar plots).
      4. Bivariate analysis (scatter plots with regression lines & box plots).
      5. Time series analysis (automatic datetime conversion and resampling).
      6. Multivariate analysis (focused on sales, region, and segment breakdowns).
    """
    # Global styling setup for plots
    warnings.filterwarnings('ignore')
    sns.set_theme(style='whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)

    # Reference the globally loaded DataFrame
    global df

    if 'df' not in globals() or df is None:
        print(
            "Error: No DataFrame named 'df' is loaded in the global scope. Please load your DataFrame first."
        )
        return

    print('============================================================')
    print('1. BASIC DATASET DESCRIPTION')
    print('============================================================')
    print(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')

    print('\n--- Data Types & Non-Null Counts ---')
    print(df.dtypes)

    print('\n--- Missing Value Counts ---')
    print(df.isnull().sum())

    print(f'\n--- Duplicate Rows Count: {df.duplicated().sum()} ---')

    print('\n--- Numerical Feature Summary ---')
    try:
        display(df.describe())
    except NameError:
        print(df.describe())

    print('\n--- Categorical Feature Summary ---')
    try:
        display(df.describe(include=['O', 'category']))
    except NameError:
        print(df.describe(include=['O', 'category']))

    print('\n============================================================')
    print('2. CORRELATION ANALYSIS')
    print('============================================================')
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty and num_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr = num_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features', fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print(
            'Skipped: Insufficient numerical columns found for correlation matrix.'
        )

    print('\n============================================================')
    print('3. UNIVARIATE ANALYSIS')
    print('============================================================')

    # Numerical columns: Histograms with KDE & Boxplots
    num_cols = num_df.columns.tolist()
    for col in num_cols[:4]:  # Limiting to first 4 numerical columns for brevity
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Histogram & KDE: {col}')
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot: {col}')
        plt.tight_layout()
        plt.show()

    # Categorical/Object columns: Frequency count plots (top 10 categories)
    cat_cols = df.select_dtypes(include=['O', 'category']).columns.tolist()
    for col in cat_cols:
        plt.figure(figsize=(10, 4))
        top_cats = df[col].value_counts().nlargest(10)
        sns.barplot(
            x=top_cats.index,
            y=top_cats.values,
            palette='viridis',
            hue=top_cats.index,
            legend=False,
        )
        plt.title(f'Top Categories Distribution: {col}')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

    print('\n============================================================')
    print('4. BIVARIATE ANALYSIS')
    print('============================================================')
    # Scatter plot with regression line for first two numerical columns if available
    if len(num_cols) >= 2:
        col_x, col_y = num_cols[0], num_cols[1]
        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=df,
            x=col_x,
            y=col_y,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'},
        )
        plt.title(f'Bivariate Scatter Plot with Regression Line: {col_y} vs {col_x}')
        plt.tight_layout()
        plt.show()

    # Boxplot / Violin plot between categorical and numerical variable
    if len(cat_cols) > 0 and len(num_cols) > 0:
        cat_sample = cat_cols[0]
        num_sample = num_cols[0]
        # Limit categories to avoid overcrowding
        top_cats_list = df[cat_sample].value_counts().nlargest(5).index
        filtered_df = df[df[cat_sample].isin(top_cats_list)]

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=filtered_df, x=cat_sample, y=num_sample, palette='Set3'
        )
        plt.title(f'Distribution of {num_sample} across top categories of {cat_sample}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    print('\n============================================================')
    print('5. TIME SERIES ANALYSIS')
    print('============================================================')
    # Automatically detect date-related columns or check for common naming conventions
    date_col = None
    potential_date_keywords = ['date', 'time', 'year', 'period', 'day']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in potential_date_keywords):
            try:
                pd.to_datetime(df[col], errors='raise')
                date_col = col
                break
            except (ValueError, TypeError):
                continue

    if date_col:
        print(f"Detected date column: '{date_col}'. Proceeding with Time Series analysis...")
        # Create a working copy for datetime conversion
        ts_df = df.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
        ts_df = ts_df.sort_values(by=date_col).set_index(date_col)

        # Identify a sales or numerical metric to aggregate over time
        sales_col = None
        for col in num_cols:
            if 'sale' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower():
                sales_col = col
                break
        if not sales_col and len(num_cols) > 0:
            sales_col = num_cols[0]  # Fallback to first numerical column

        if sales_col:
            monthly_trend = ts_df[sales_col].resample('M').sum()
            plt.figure(figsize=(14, 5))
            plt.plot(
                monthly_trend.index,
                monthly_trend.values,
                marker='o',
                linestyle='-',
                color='b',
            )
            plt.title(f'Monthly Aggregated Trend of {sales_col}', fontsize=14)
            plt.xlabel('Date')
            plt.ylabel(f'Total {sales_col}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print('Skipped Time Series trend plot: No suitable numerical metric found.')
    else:
        print('Skipped Time Series analysis: No valid datetime column automatically detected.')

    print('\n============================================================')
    print('6. MULTIVARIATE ANALYSIS (Sales, Region, Segment focus)')
    print('============================================================')
    # Look for matching columns case-insensitively
    cols_lower = {col.lower(): col for col in df.columns}
    
    # Try to locate standard columns referenced in prompts ('sales', 'region', 'segment')
    target_sales = cols_lower.get('sales')
    target_region = cols_lower.get('region')
    target_segment = cols_lower.get('segment')

    # Fallbacks if exact names aren't found
    if not target_sales and len(num_cols) > 0:
        target_sales = num_cols[0]
    if not target_region and len(cat_cols) > 0:
        target_region = cat_cols[0]
    if not target_segment and len(cat_cols) > 1:
        target_segment = cat_cols[1]

    if target_sales and target_region and target_segment and target_region != target_segment:
        print(f"Using columns -> Sales: '{target_sales}', Region: '{target_region}', Segment: '{target_segment}'")
        plt.figure(figsize=(12, 6))
        
        # Using Seaborn barplot with hue configuration
        sns.barplot(
            data=df,
            x=target_region,
            y=target_sales,
            hue=target_segment,
            estimator=np.sum,
            errorbar=None,
            palette='Set2',
        )
        plt.title(f'Total {target_sales} by {target_region} and {target_segment}', fontsize=14)
        plt.xlabel(target_region.capitalize())
        plt.ylabel(f'Total {target_sales}')
        plt.legend(title=target_segment.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print('Skipped Multivariate analysis: Could not identify distinct columns for sales, region, and segment.')

    print('\n============================================================')
    print('EDA COMPLETE SUCCESSFULLY.')
    print('============================================================')

# To run the EDA, simply call the function in your environment where `df` is loaded:
# eda_by_ai()
