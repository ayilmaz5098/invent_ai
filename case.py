###

import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
# --- Step 0: Environment & Data Loading ---
print("--- Step 0: Environment & Data Loading ---")
script_dir = os.getcwd()
file_a = os.path.join(script_dir, "Assignment4.1a.csv")
file_b = os.path.join(script_dir, "Assignment4.1b.csv")
file_c = os.path.join(script_dir, "Assignment4.1c.csv")
file_promo = os.path.join(script_dir, "PromotionDates.csv")

try:
    df_a = pd.read_csv(file_a, parse_dates=['Date'])
    df_b = pd.read_csv(file_b, parse_dates=['Date'])
    df_c = pd.read_csv(file_c)
    df_promo = pd.read_csv(file_promo, parse_dates=['StartDate', 'EndDate'])
    print("All data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Missing file: {e.filename}.")
    exit()

# FIX: Clean up whitespace in Period column to ensure all promos are found
df_promo['Period'] = df_promo['Period'].str.strip()

# --- Helper Function for Data Preparation ---
def create_full_sales_df(df, min_date, max_date):
    """Fills in missing dates for each store-product pair with 0 sales."""
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    unique_store_product = df[['StoreCode', 'ProductCode']].drop_duplicates()
    idx = pd.MultiIndex.from_product([
        unique_store_product['StoreCode'].unique(),
        unique_store_product['ProductCode'].unique(),
        full_date_range
    ], names=['StoreCode', 'ProductCode', 'Date'])
    df_full = pd.DataFrame(index=idx).reset_index()
    df_full = pd.merge(df_full, df, on=['StoreCode', 'ProductCode', 'Date'], how='left')
    df_full['SalesQuantity'] = df_full['SalesQuantity'].fillna(0).astype(int)
    return df_full

# --- Step 1: Part A Preprocessing ---
print("\n--- Step 1: Preprocessing for Part A ---")
df_sales_full_a = create_full_sales_df(df_a, df_a['Date'].min(), df_a['Date'].max())
df_promo_part_a = df_promo[df_promo['Period'].isin(['Promo1', 'Promo2', 'Promo3', 'Promo4'])].copy()
promo_periods_a = [(row['StartDate'], row['EndDate']) for _, row in df_promo_part_a.iterrows()]
df_sales_full_a['IsPromotion'] = df_sales_full_a['Date'].apply(lambda d: any(s <= d <= e for s, e in promo_periods_a))
df_sales_full_a['GrossSales'] = df_sales_full_a['SalesQuantity'].where(df_sales_full_a['SalesQuantity'] > 0, 0)
df_sales_full_a['Returns'] = -df_sales_full_a['SalesQuantity'].where(df_sales_full_a['SalesQuantity'] < 0, 0)
print("Part A data prepared.")

# --- Step 2: Descriptive Analysis & Seasonality Check ---
print("\n--- Step 2: Descriptive Analysis & Seasonality ---")
plt.style.use('seaborn-v0_8-darkgrid')

# Overall Sales Trend
daily_sales_a = df_sales_full_a.groupby('Date')['SalesQuantity'].sum()
plt.figure(figsize=(15, 7))
daily_sales_a.plot(label='Net Daily Sales', color='royalblue')
plt.title('Total Net Daily Sales with Promotion Periods', fontsize=16)
for _, row in df_promo_part_a.iterrows():
    plt.axvspan(row['StartDate'], row['EndDate'], color='orange', alpha=0.3, label=f"_{row['Period']}")
plt.legend(['Net Daily Sales', 'Promotion Periods'])
plt.tight_layout()
plt.savefig("descriptive_sales_timeseries.png")
plt.show()

# Time-Based Patterns
df_sales_full_a['DayOfWeek'] = df_sales_full_a['Date'].dt.day_name()
df_sales_full_a['Month'] = df_sales_full_a['Date'].dt.month_name()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July']

df_sales_full_a.groupby('DayOfWeek')['SalesQuantity'].mean().reindex(days_order).plot(kind='bar', figsize=(10,6), color=sns.color_palette('crest', 7))
plt.title('Average Daily Sales by Day of the Week', fontsize=16)
plt.ylabel('Average Sales per Item-Store')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sales_by_day_of_week.png")
plt.show()

df_sales_full_a.groupby('Month')['SalesQuantity'].mean().reindex(months_order).plot(kind='bar', figsize=(10,6), color=sns.color_palette('flare', 7))
plt.title('Average Daily Sales by Month', fontsize=16)
plt.ylabel('Average Sales per Item-Store')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sales_by_month.png")
plt.show()


# --- Step 3: Clustering and Uplift Calculation ---
print("\n--- Step 3: Clustering and Uplift Calculation ---")
df_non_promo_a = df_sales_full_a[~df_sales_full_a['IsPromotion']].copy()
product_sales = df_non_promo_a.groupby('ProductCode')['SalesQuantity'].mean().reset_index()
product_sales['AvgWeeklySales'] = product_sales['SalesQuantity'] * 7
p_q1, p_q2 = product_sales['AvgWeeklySales'].quantile(0.33), product_sales['AvgWeeklySales'].quantile(0.67)
product_sales['ProductSalesCategory'] = product_sales['AvgWeeklySales'].apply(lambda s: 'Slow' if s <= p_q1 else ('Medium' if s <= p_q2 else 'Fast'))

store_sales = df_non_promo_a.groupby('StoreCode')['SalesQuantity'].mean().reset_index()
store_sales['AvgWeeklySales'] = store_sales['SalesQuantity'] * 7
s_q1, s_q2 = store_sales['AvgWeeklySales'].quantile(0.33), store_sales['AvgWeeklySales'].quantile(0.67)
store_sales['StoreSalesCategory'] = store_sales['AvgWeeklySales'].apply(lambda s: 'Slow' if s <= s_q1 else ('Medium' if s <= s_q2 else 'Fast'))

df_sales_full_a = pd.merge(df_sales_full_a, product_sales[['ProductCode', 'ProductSalesCategory']], on='ProductCode')
df_sales_full_a = pd.merge(df_sales_full_a, store_sales[['StoreCode', 'StoreSalesCategory']], on='StoreCode')

avg_sales_by_promo = df_sales_full_a.groupby(['StoreCode', 'ProductCode', 'ProductSalesCategory', 'StoreSalesCategory', 'IsPromotion'])['SalesQuantity'].mean().unstack(fill_value=0)
avg_sales_by_promo.rename(columns={False: 'AvgSales_NonPromo', True: 'AvgSales_Promo'}, inplace=True)
avg_sales_by_promo['Absolute_Uplift'] = avg_sales_by_promo['AvgSales_Promo'] - avg_sales_by_promo['AvgSales_NonPromo']
avg_sales_by_promo = avg_sales_by_promo.reset_index()
print("Clustering and uplift calculation complete.")


# --- Step 4: Part A - Key Questions & Visualizations ---
# A.c & A.d: Find top performing items and stores
print("\n--- Answering A.c & A.d: Top Performers ---")
# Find top items by absolute increase
biggest_item_increase = avg_sales_by_promo.sort_values(by='Absolute_Uplift', ascending=False).head(10)
biggest_item_increase['Label'] = 'Prod ' + biggest_item_increase['ProductCode'].astype(str) + ' / Store ' + biggest_item_increase['StoreCode'].astype(str)

print("\nTop 10 Item-Store pairs by Absolute Sales Increase:")
print(biggest_item_increase[['Label', 'Absolute_Uplift']])

# Visualize the top 10 items
plt.figure(figsize=(12, 7))
sns.barplot(x='Absolute_Uplift', y='Label', data=biggest_item_increase, palette='viridis')
plt.title('Top 10 Items by Biggest Sales Increase (Absolute Uplift)', fontsize=16)
plt.xlabel('Additional Units Sold Per Day During Promotion')
plt.ylabel('Product / Store Combination')
plt.tight_layout()
plt.savefig("top_10_item_uplift.png")
plt.show()

# Visualize the top 10 stores by reaction


# Find top stores by relative reaction
avg_sales_by_promo['Relative_Uplift'] = ((avg_sales_by_promo['Absolute_Uplift'] / avg_sales_by_promo['AvgSales_NonPromo']) * 100).replace([np.inf, -np.inf], 0).fillna(0)
store_reaction = avg_sales_by_promo.groupby('StoreCode')['Relative_Uplift'].mean().reset_index()
highest_store_reaction = store_reaction.sort_values(by='Relative_Uplift', ascending=False).head(10)

print("\nTop 10 Stores by Highest Promotion Reaction (%):")
print(highest_store_reaction)
plt.figure(figsize=(12, 7))
sns.barplot(x='Relative_Uplift', y='StoreCode', data=highest_store_reaction, palette='plasma', orient='h')
plt.title('Top 10 Stores by Highest Promotion Reaction (%)', fontsize=16)
plt.xlabel('Average Relative Uplift (%)')
plt.ylabel('Store Code')
plt.tight_layout()
plt.savefig("top_10_store_reaction.png")
plt.show()
print("\n--- Step 4: Answering Key Questions with Visuals ---")

print("\n--- Answering A.e: What is the biggest effect explaining sales? ---")
df_reg = df_sales_full_a.sample(n=min(100000, len(df_sales_full_a)), random_state=42)
y = df_reg['SalesQuantity']
X = df_reg[['IsPromotion', 'ProductSalesCategory', 'StoreSalesCategory']]
X = pd.get_dummies(X, columns=['ProductSalesCategory', 'StoreSalesCategory'], drop_first=True, dtype=int)
X['IsPromotion'] = X['IsPromotion'].astype(int)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# T-Tests and Box Plots
fast_items_uplift = avg_sales_by_promo[avg_sales_by_promo['ProductSalesCategory'] == 'Fast']['Absolute_Uplift']
slow_items_uplift = avg_sales_by_promo[avg_sales_by_promo['ProductSalesCategory'] == 'Slow']['Absolute_Uplift']
_, p_val_items = stats.ttest_ind(fast_items_uplift, slow_items_uplift, equal_var=False)
print(f"\nT-test for Fast vs. Slow Items P-value: {p_val_items:.4f} -> Difference is {'SIGNIFICANT' if p_val_items < 0.05 else 'NOT SIGNIFICANT'}")
plt.figure(figsize=(10, 6)); sns.boxplot(x='ProductSalesCategory', y='Absolute_Uplift', data=avg_sales_by_promo[avg_sales_by_promo['ProductSalesCategory'].isin(['Fast', 'Slow'])], palette='viridis', order=['Slow', 'Fast']); plt.title('Comparison of Sales Uplift: Fast vs. Slow Items'); plt.savefig("ttest_items_boxplot.png"); plt.show()

fast_stores_uplift = avg_sales_by_promo[avg_sales_by_promo['StoreSalesCategory'] == 'Fast']['Absolute_Uplift']
slow_stores_uplift = avg_sales_by_promo[avg_sales_by_promo['StoreSalesCategory'] == 'Slow']['Absolute_Uplift']
_, p_val_stores = stats.ttest_ind(fast_stores_uplift, slow_stores_uplift, equal_var=False)
print(f"T-test for Fast vs. Slow Stores P-value: {p_val_stores:.4f} -> Difference is {'SIGNIFICANT' if p_val_stores < 0.05 else 'NOT SIGNIFICANT'}")
plt.figure(figsize=(10, 6)); sns.boxplot(x='StoreSalesCategory', y='Absolute_Uplift', data=avg_sales_by_promo[avg_sales_by_promo['StoreSalesCategory'].isin(['Fast', 'Slow'])], palette='plasma', order=['Slow', 'Fast']); plt.title('Comparison of Sales Uplift: Fast vs. Slow Stores'); plt.savefig("ttest_stores_boxplot.png"); plt.show()

# Improved Uplift Visuals
uplift_summary = avg_sales_by_promo.groupby('ProductSalesCategory')['Absolute_Uplift'].describe()
print("\nDescriptive Statistics for Absolute Uplift by Product Category:"); print(uplift_summary[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'}))
plt.figure(figsize=(10, 6)); sns.barplot(x=uplift_summary.index, y=uplift_summary['mean'], palette='viridis', order=['Slow', 'Medium', 'Fast']); plt.title('Mean (Average) Sales Uplift by Product Category'); plt.axhline(0, color='k', lw=0.8, ls='--'); plt.savefig("mean_uplift_by_category.png"); plt.show()

def get_uplift_type(uplift): return 'Positive Uplift' if uplift > 0 else ('Negative Uplift' if uplift < 0 else 'No Change')
avg_sales_by_promo['Uplift_Type'] = avg_sales_by_promo['Absolute_Uplift'].apply(get_uplift_type)
uplift_distribution_percent = avg_sales_by_promo.groupby(['ProductSalesCategory', 'Uplift_Type']).size().unstack(fill_value=0).apply(lambda x: x / x.sum() * 100, axis=1)
ax = uplift_distribution_percent.plot(kind='bar', stacked=True, color=['#d65f5f', '#f7f7f7', '#67a9cf'], figsize=(12, 7))
for c in ax.containers: ax.bar_label(c, labels=[f'{w:.1f}%' if (w := v.get_height()) > 5 else '' for v in c], label_type='center', weight='bold')
ax.legend(title='Uplift Type', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.title('Distribution of Promotion Uplift Type by Product Category'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig("uplift_distribution_by_category_labeled.png"); plt.show()


# --- Step 5: Forecasting and Evaluation ---
def evaluate_promotion(df_sales_data, df_promo_info, promo_name, product_cats, store_cats, uplift_model):
    print(f"\n--- Evaluating Forecast for {promo_name} ---")
    try:
        promo_dates = df_promo_info[df_promo_info['Period'] == promo_name].iloc[0]
        promo_start, promo_end = promo_dates['StartDate'], promo_dates['EndDate']
    except IndexError:
        print(f"Warning: {promo_name} not found. Skipping.")
        return

    df_eval = create_full_sales_df(df_sales_data, df_sales_data['Date'].min(), df_sales_data['Date'].max())
    df_eval = pd.merge(df_eval, product_cats, on='ProductCode', how='left')
    df_eval = pd.merge(df_eval, store_cats, on='StoreCode', how='left')
    df_eval['ProductSalesCategory'] = df_eval['ProductSalesCategory'].fillna('Medium')
    df_eval['StoreSalesCategory'] = df_eval['StoreSalesCategory'].fillna('Medium')

    df_baseline_period = df_eval[df_eval['Date'] < promo_start]
    df_promo_period = df_eval[(df_eval['Date'] >= promo_start) & (df_eval['Date'] <= promo_end)]
    
    baseline_sales = df_baseline_period.groupby(['StoreCode', 'ProductCode', 'ProductSalesCategory', 'StoreSalesCategory'])['SalesQuantity'].mean().reset_index(name='BaselineDailySales')
    actual_promo_sales = df_promo_period.groupby(['StoreCode', 'ProductCode'])['SalesQuantity'].mean().reset_index(name='ActualPromoDailySales')

    forecast_df = pd.merge(baseline_sales, uplift_model, on=['ProductSalesCategory', 'StoreSalesCategory'], how='left')
    forecast_df['ForecastedPromoDailySales'] = forecast_df['BaselineDailySales'] * (1 + forecast_df['Forecast_Uplift_Percent'] / 100)
    
    eval_df = pd.merge(forecast_df, actual_promo_sales, on=['StoreCode', 'ProductCode'])
    eval_df.dropna(subset=['ActualPromoDailySales', 'ForecastedPromoDailySales'], inplace=True)
    if eval_df.empty:
        print(f"No matching sales data found for {promo_name} evaluation.")
        return

    mae = mean_absolute_error(eval_df['ActualPromoDailySales'], eval_df['ForecastedPromoDailySales'])
    rmse = np.sqrt(mean_squared_error(eval_df['ActualPromoDailySales'], eval_df['ForecastedPromoDailySales']))
    print(f"Goodness of Fit for {promo_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    plt.figure(figsize=(10, 7)); sns.scatterplot(x='ActualPromoDailySales', y='ForecastedPromoDailySales', data=eval_df, alpha=0.5); plt.plot([eval_df['ActualPromoDailySales'].min(), eval_df['ActualPromoDailySales'].max()], [eval_df['ActualPromoDailySales'].min(), eval_df['ActualPromoDailySales'].max()], 'r--', lw=2, label='Perfect Fit'); plt.title(f'Forecasted vs. Actual Sales for {promo_name}'); plt.legend(); plt.tight_layout(); plt.savefig(f"forecast_vs_actual_{promo_name}.png"); plt.show()

print("\n--- Step 5: Evaluating Model on Future Promotions ---")
avg_sales_by_promo['Relative_Uplift'] = ((avg_sales_by_promo['Absolute_Uplift'] / avg_sales_by_promo['AvgSales_NonPromo']) * 100).replace([np.inf, -np.inf], 0).fillna(0)
uplift_model = avg_sales_by_promo.groupby(['ProductSalesCategory', 'StoreSalesCategory'])['Relative_Uplift'].mean().reset_index(name='Forecast_Uplift_Percent')
product_categories = product_sales[['ProductCode', 'ProductSalesCategory']]
store_categories = store_sales[['StoreCode', 'StoreSalesCategory']]

evaluate_promotion(df_b, df_promo, 'Promo5', product_categories, store_categories, uplift_model)
evaluate_promotion(df_b, df_promo, 'Promo6', product_categories, store_categories, uplift_model)

# --- Step 6: Bonus Questions ---
print("\n--- Step 6: Answering Bonus Questions ---")
promo_return_rate = df_sales_full_a[df_sales_full_a['IsPromotion']]['Returns'].sum() / df_sales_full_a[df_sales_full_a['IsPromotion']]['GrossSales'].sum()
non_promo_return_rate = df_sales_full_a[~df_sales_full_a['IsPromotion']]['Returns'].sum() / df_sales_full_a[~df_sales_full_a['IsPromotion']]['GrossSales'].sum()
print(f"\nReturn Rate during Promotions: {promo_return_rate:.4%}")
print(f"Return Rate during Non-Promotions: {non_promo_return_rate:.4%}")

try:
    df_analysis_c = pd.merge(avg_sales_by_promo, df_c, on='ProductCode', how='left').dropna(subset=['ProductGroup1'])
    uplift_by_prod_group1 = df_analysis_c.groupby('ProductGroup1')['Relative_Uplift'].mean().sort_values(ascending=False)
    print("\nAverage Relative Uplift % by ProductGroup1:"); print(uplift_by_prod_group1)
    uplift_by_prod_group1.plot(kind='bar', figsize=(12, 7), color=sns.color_palette("magma", len(uplift_by_prod_group1))); plt.title('Average Promotion Uplift by Product Hierarchy'); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig("uplift_by_product_group.png"); plt.show()
except Exception as e:
    print(f"Could not perform Bonus 2 analysis. Error: {e}")

print("\n--- End of Analysis ---")