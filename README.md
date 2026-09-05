# Iowa Liquor Sales: Exploratory Data Analysis

An end-to-end retail EDA of spirits purchases made by Iowa Class E liquor licensees during 2024. The project follows the **PACE** framework used in the Google Advanced Data Analytics programme: Plan, Analyse, Construct and Execute.

## Business question

How can retailers use product, seasonal and geographic sales patterns to improve stock planning and understand store performance?

## What the project covers

- Reproducible loading of the official State of Iowa CSV
- Data-quality checks for missing values, duplicates and invalid records
- Feature engineering for month, weekday, unit price and estimated gross profit
- Distribution and IQR outlier analysis
- Monthly revenue and estimated profit trends
- Product-category, vendor, store and county comparisons
- Price-volume and cost-retail relationships
- Clear recommendations and an executive summary

## Repository structure

```text
iowa-liquor-sales-eda/
├── iowa_liquor_sales_eda.ipynb
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── README.md
└── images/
    └── .gitkeep
```

## Run in Google Colab

Open `iowa_liquor_sales_eda.ipynb` in Colab and select **Runtime > Run all**. The notebook reads the official 2024 CSV in chunks and keeps a reproducible sample so it remains practical on the free Colab runtime.

To analyse the complete file, change `SAMPLE_PER_CHUNK` to `None`. This needs substantially more memory.

## Key metrics

The notebook calculates:

- Sales revenue
- Bottles sold
- Sales volume in litres
- Estimated gross profit
- Estimated gross margin
- Average order value
- Revenue by month, county, store, vendor and product category

Estimated gross profit is calculated from the recorded state retail and state cost values. It is an analytical estimate and does not include store operating costs, discounts, taxes or other expenses.

## Data source

State of Iowa, Alcohol Operations Bureau, **Iowa Liquor Sales, 2024**:

- [Dataset catalogue](https://catalog.data.gov/dataset/iowa-liquor-sales-2024)
- [CSV download](https://idh-be.iowa.gov/api/v1/datasets/1261/rows.csv)

The dataset is public and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn and Jupyter/Google Colab.

## Important limitation

The records represent wholesale spirits purchases by licensed retailers, rather than purchases made by individual consumers. Results should therefore be interpreted as retailer purchasing patterns, not direct consumer behaviour.

