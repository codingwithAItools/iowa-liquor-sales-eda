import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": [],
    "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "colab": {"name": "iowa_liquor_sales_eda.ipynb", "provenance": []},
    },
}


def md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    })


def code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    })


md("""
# Iowa Liquor Sales: Exploratory Data Analysis

**Portfolio project | Retail analytics | Python**

This project investigates spirits purchased by Iowa liquor retailers during 2024. It follows the **PACE** framework: Plan, Analyse, Construct and Execute.

> **Important:** Each record represents a purchase by a licensed retailer, not a transaction made by an individual consumer.
""")

md("""
## PACE: Plan

### Business problem

Retailers need to understand which products, locations and periods generate the most sales so they can improve stock planning and identify performance differences.

### Primary question

**How can retailers use product, seasonal and geographic sales patterns to improve stock planning and understand store performance?**

### Supporting questions

1. How do revenue, units and estimated profit change by month?
2. Which categories, vendors and products account for the most revenue?
3. Which counties and stores purchase the most spirits?
4. How strongly are price, units and sales value related?
5. Are extreme transactions plausible bulk orders or likely data-quality problems?

### Success criteria

- Produce a clean, reproducible analytical dataset.
- Explain missing, duplicate and implausible records.
- Present accessible visualisations with direct titles and readable labels.
- Translate findings into practical inventory recommendations.
""")

code("""
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

sns.set_theme(style="whitegrid", context="notebook")
BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#10B981"
RED = "#EF4444"
""")

md("""
## PACE: Analyse

### Load the data

Download the CSV files from the official dataset page and upload all of them to the Colab session. The loader searches both `/content` (Colab uploads) and the repository's `data` folder, then reads every CSV in chunks.

This local-file approach avoids the HTTP 403 error returned by the Iowa download server when pandas requests the CSV directly. To keep the notebook practical in free Google Colab, it takes a reproducible sample from every chunk. Set `SAMPLE_PER_CHUNK = None` to retain every row.
""")

code("""
CHUNK_SIZE = 100_000
SAMPLE_PER_CHUNK = 12_500
RANDOM_STATE = 42

search_folders = [Path("/content"), Path("data")]
csv_files = sorted({
    file.resolve()
    for folder in search_folders
    if folder.exists()
    for file in folder.glob("*.csv")
})

if not csv_files:
    raise FileNotFoundError(
        "No CSV files were found. Upload all five CSV files to Colab, "
        "or place them inside the project's data folder."
    )

print(f"CSV files found: {len(csv_files)}")
for file in csv_files:
    print(f"- {file.name}")

chunks = []
chunk_number = 0

for file in csv_files:
    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
        if SAMPLE_PER_CHUNK is not None and len(chunk) > SAMPLE_PER_CHUNK:
            chunk = chunk.sample(
                n=SAMPLE_PER_CHUNK,
                random_state=RANDOM_STATE + chunk_number,
            )
        chunk["source_file"] = file.name
        chunks.append(chunk)
        chunk_number += 1

raw = pd.concat(chunks, ignore_index=True)
print(f"Files combined: {len(csv_files)}")
print(f"Rows loaded: {len(raw):,}")
print(f"Columns loaded: {raw.shape[1]}")
raw.head()
""")

md("""
### Standardise column names

Government datasets often use spaces, punctuation and units in column names. Standard snake-case names make subsequent code more consistent.
""")

code("""
def clean_column_name(name):
    name = str(name).strip().lower()
    name = name.replace("$", "dollars").replace("%", "pct")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

df = raw.copy()
df.columns = [clean_column_name(c) for c in df.columns]

# Support small naming differences between official releases.
ALIASES = {
    "sale_dollars": "sale_dollars",
    "sale": "sale_dollars",
    "state_bottle_cost": "state_bottle_cost",
    "state_bottle_retail": "state_bottle_retail",
    "volume_sold_liters": "volume_sold_liters",
    "volume_sold_litres": "volume_sold_liters",
    "category_name": "category_name",
}
df = df.rename(columns={c: ALIASES[c] for c in df.columns if c in ALIASES})

print(df.shape)
df.head()
""")

code("""
df.info()
""")

code("""
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_pct": df.isna().mean().mul(100).round(2),
    "unique": df.nunique(dropna=True),
}).sort_values("missing_pct", ascending=False)

quality
""")

md("""
### Clean types and validate records

Identifiers such as store, item and postcode are treated as text. Measures are converted to numeric values, and dates are parsed explicitly. Records with missing dates or non-positive sales quantities cannot support the main analysis and are removed.
""")

code("""
numeric_candidates = [
    "pack", "bottle_volume_ml", "state_bottle_cost", "state_bottle_retail",
    "bottles_sold", "sale_dollars", "volume_sold_liters", "volume_sold_gallons"
]
id_candidates = ["store_number", "zip_code", "county_number", "category", "vendor_number", "item_number"]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce"
        )

for col in id_candidates:
    if col in df.columns:
        df[col] = df[col].astype("string").str.replace(r"\\.0$", "", regex=True).str.strip()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

duplicate_rows = int(df.duplicated().sum())
print(f"Exact duplicate rows: {duplicate_rows:,}")

required = ["date", "bottles_sold", "sale_dollars"]
before = len(df)
df = df.dropna(subset=[c for c in required if c in df.columns]).drop_duplicates()
df = df[(df["bottles_sold"] > 0) & (df["sale_dollars"] > 0)].copy()
print(f"Rows removed by validation: {before - len(df):,}")
print(f"Rows retained: {len(df):,}")
""")

md("""
### Feature engineering

Estimated gross profit uses the difference between the recorded state retail price and state cost. It is not net profit and excludes taxes, discounts and operating expenses.
""")

code("""
df["month"] = df["date"].dt.month_name()
df["month_number"] = df["date"].dt.month
df["day_name"] = df["date"].dt.day_name()
df["is_weekend"] = df["date"].dt.dayofweek >= 5
df["unit_sale_price"] = df["sale_dollars"] / df["bottles_sold"]

if {"state_bottle_retail", "state_bottle_cost", "bottles_sold"}.issubset(df.columns):
    df["estimated_gross_profit"] = (
        df["state_bottle_retail"] - df["state_bottle_cost"]
    ) * df["bottles_sold"]
    df["estimated_margin_pct"] = np.where(
        df["state_bottle_retail"] > 0,
        df["estimated_gross_profit"] / df["sale_dollars"] * 100,
        np.nan,
    )

df[["date", "month", "day_name", "unit_sale_price"]].head()
""")

code("""
metrics = {
    "Transactions": len(df),
    "Sales revenue": df["sale_dollars"].sum(),
    "Bottles sold": df["bottles_sold"].sum(),
    "Average transaction": df["sale_dollars"].mean(),
}
if "estimated_gross_profit" in df:
    metrics["Estimated gross profit"] = df["estimated_gross_profit"].sum()

pd.Series(metrics, name="value").to_frame()
""")

md("""
### Descriptive statistics and outliers

Retail orders are expected to be right-skewed: most orders are modest, while a smaller number of bulk orders are very large. Therefore, an outlier is a record requiring investigation, not automatically an error.
""")

code("""
measures = [c for c in [
    "bottles_sold", "sale_dollars", "volume_sold_liters",
    "unit_sale_price", "estimated_gross_profit"
] if c in df.columns]

df[measures].describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
""")

code("""
def iqr_summary(data, columns):
    rows = []
    for col in columns:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (data[col] < lower) | (data[col] > upper)
        rows.append({
            "variable": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "outlier_pct": round(mask.mean() * 100, 2),
        })
    return pd.DataFrame(rows)

outlier_report = iqr_summary(df, measures)
outlier_report
""")

code("""
fig, axes = plt.subplots(len(measures), 2, figsize=(13, 3.2 * len(measures)))
axes = np.atleast_2d(axes)

for i, col in enumerate(measures):
    upper_99 = df[col].quantile(0.99)
    sns.boxplot(x=df.loc[df[col] <= upper_99, col], ax=axes[i, 0], color=BLUE)
    sns.histplot(df.loc[df[col] <= upper_99, col], bins=40, ax=axes[i, 1], color=ORANGE)
    axes[i, 0].set_title(f"{col.replace('_', ' ').title()} (up to 99th percentile)")
    axes[i, 1].set_title(f"Distribution of {col.replace('_', ' ')}")

plt.tight_layout()
plt.show()
""")

md("""
## PACE: Construct

### Monthly sales performance
""")

code("""
monthly_metrics = (
    df.groupby(["month_number", "month"], as_index=False)
      .agg(
          revenue=("sale_dollars", "sum"),
          bottles=("bottles_sold", "sum"),
          transactions=("sale_dollars", "size"),
          average_transaction=("sale_dollars", "mean"),
      )
      .sort_values("month_number")
)
if "estimated_gross_profit" in df:
    profit = df.groupby(["month_number", "month"])["estimated_gross_profit"].sum().reset_index(name="estimated_profit")
    monthly_metrics = monthly_metrics.merge(profit, on=["month_number", "month"])

monthly_metrics
""")

code("""
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=monthly_metrics, x="month", y="revenue", marker="o", linewidth=2.5, color=BLUE, ax=ax)
ax.set_title("Monthly retail purchases reveal seasonal demand", loc="left", weight="bold")
ax.set_xlabel("")
ax.set_ylabel("Sales revenue ($)")
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(lambda x, pos: f"${x/1_000_000:.1f}M")
plt.tight_layout()
plt.show()
""")

md("""
### Categories, stores, vendors and counties
""")

code("""
def top_groups(data, group_col, n=10):
    return (
        data.groupby(group_col, dropna=False)
            .agg(revenue=("sale_dollars", "sum"), bottles=("bottles_sold", "sum"), transactions=("sale_dollars", "size"))
            .sort_values("revenue", ascending=False)
            .head(n)
            .reset_index()
    )

group_columns = [c for c in ["category_name", "vendor_name", "county", "store_name", "item_description"] if c in df]
top_tables = {col: top_groups(df, col) for col in group_columns}

for name, table in top_tables.items():
    print(f"Top {name.replace('_', ' ')}")
    display(table)
""")

code("""
plot_groups = [c for c in ["category_name", "county", "store_name"] if c in top_tables]
fig, axes = plt.subplots(1, len(plot_groups), figsize=(6 * len(plot_groups), 6))
axes = np.atleast_1d(axes)

for ax, col in zip(axes, plot_groups):
    plot_data = top_tables[col].sort_values("revenue")
    sns.barplot(data=plot_data, x="revenue", y=col, color=BLUE, ax=ax)
    ax.set_title(f"Top {col.replace('_', ' ')} by revenue", loc="left", weight="bold")
    ax.set_xlabel("Revenue ($)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, pos: f"${x/1_000_000:.0f}M")

plt.tight_layout()
plt.show()
""")

md("""
### Revenue concentration

Concentration measures whether a small group of stores or products accounts for a disproportionately large share of revenue.
""")

code("""
def concentration_table(data, group_col):
    result = data.groupby(group_col)["sale_dollars"].sum().sort_values(ascending=False).reset_index(name="revenue")
    result["revenue_share_pct"] = result["revenue"].div(result["revenue"].sum()).mul(100)
    result["cumulative_share_pct"] = result["revenue_share_pct"].cumsum()
    return result

store_concentration = concentration_table(df, "store_name")
store_concentration.head(10)
""")

code("""
top_10_share = store_concentration.head(10)["revenue_share_pct"].sum()
stores_for_80_pct = int((store_concentration["cumulative_share_pct"] < 80).sum() + 1)

print(f"Top 10 stores' share of sampled revenue: {top_10_share:.1f}%")
print(f"Stores required to reach 80% of sampled revenue: {stores_for_80_pct:,}")
""")

md("""
### Relationships between price, units and revenue
""")

code("""
corr_cols = [c for c in [
    "state_bottle_cost", "state_bottle_retail", "unit_sale_price",
    "bottles_sold", "sale_dollars", "volume_sold_liters", "estimated_gross_profit"
] if c in df]

plt.figure(figsize=(9, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1)
plt.title("Correlation between retail measures", loc="left", weight="bold")
plt.tight_layout()
plt.show()
""")

code("""
plot_sample = df.sample(min(10_000, len(df)), random_state=RANDOM_STATE)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_sample, x="bottles_sold", y="sale_dollars", alpha=.25, s=25, color=BLUE)
plt.xscale("log")
plt.yscale("log")
plt.title("Larger orders generally generate more revenue", loc="left", weight="bold")
plt.xlabel("Bottles sold (log scale)")
plt.ylabel("Sales revenue (log scale)")
plt.tight_layout()
plt.show()
""")

md("""
## PACE: Execute

### Automatically generated findings

The cell below turns the calculated results into a concise evidence-based summary. Because the source can be updated, values are generated dynamically rather than hard-coded.
""")

code("""
peak_month = monthly_metrics.loc[monthly_metrics["revenue"].idxmax()]
quiet_month = monthly_metrics.loc[monthly_metrics["revenue"].idxmin()]
top_category = top_tables.get("category_name", pd.DataFrame()).head(1)
top_county = top_tables.get("county", pd.DataFrame()).head(1)
top_store = top_tables.get("store_name", pd.DataFrame()).head(1)

print("EXECUTIVE SUMMARY")
print("-" * 70)
print(f"The analytical sample contains {len(df):,} valid retailer purchase records.")
print(f"Peak revenue occurred in {peak_month['month']}; the lowest occurred in {quiet_month['month']}.")
if not top_category.empty:
    print(f"The highest-revenue category was {top_category.iloc[0]['category_name']}.")
if not top_county.empty:
    print(f"The highest-revenue county was {top_county.iloc[0]['county']}.")
if not top_store.empty:
    print(f"The highest-revenue store was {top_store.iloc[0]['store_name']}.")
print(f"The ten leading stores generated {top_10_share:.1f}% of sampled revenue.")
print("IQR outliers were retained because large wholesale orders can be legitimate business activity.")
""")

md("""
### Recommendations

1. **Align stock planning with monthly demand.** Build inventory ahead of the strongest months and investigate whether quieter months reflect seasonality or controllable stock constraints.
2. **Prioritise high-value categories without relying on revenue alone.** Compare revenue, bottles and estimated profit because high sales do not always mean the strongest margin.
3. **Use county and store benchmarks.** Compare similar stores within the same county before treating high or low performance as exceptional.
4. **Investigate large orders instead of automatically deleting them.** Wholesale purchasing naturally creates right-skewed distributions. Confirm extreme records using price, bottle count and volume together.
5. **Monitor concentration risk.** Heavy dependence on a few stores, vendors or categories can make revenue vulnerable to local or supplier changes.

### Limitations

- A reproducible sample is used by default for memory efficiency, so totals are sample totals rather than official statewide totals.
- Purchases made by retailers are not the same as final consumer purchases.
- Estimated profit does not account for discounts, taxes, wages, rent or other operating costs.
- Correlation identifies association, not causation.
- Missing category or geographic fields may slightly affect group comparisons.

### Next steps

- Run the analysis on the complete 2024 file using a higher-memory environment.
- Combine multiple years to separate normal seasonality from longer-term change.
- Add county population to calculate per-capita measures.
- Build an interactive Power BI or Tableau dashboard with monthly, category and geographic filters.
""")

md("""
## Data source and licence

- [Data.gov catalogue entry](https://catalog.data.gov/dataset/iowa-liquor-sales-2024)
- [Official State of Iowa CSV](https://idh-be.iowa.gov/api/v1/datasets/1261/rows.csv)
- Dataset licence: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
""")

with (ROOT / "iowa_liquor_sales_eda.ipynb").open("w", encoding="utf-8") as file:
    json.dump(nb, file, indent=1, ensure_ascii=False)
print(f"Created {ROOT / 'iowa_liquor_sales_eda.ipynb'}")
