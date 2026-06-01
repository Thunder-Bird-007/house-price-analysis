# House Price Analysis — Exploratory Data Analysis

**Project 02 | Data Analysis Roadmap | May 2026**

---

## What This Project Is About

This is an exploratory data analysis of the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The central question I wanted to answer:

> **What factors most strongly predict house sale prices?**

I worked through 7 structured analysis questions, covering distribution analysis, correlation tests, feature engineering, and a full correlation heatmap. The goal wasn't just to get numbers — it was to understand *why* each method was the right tool for each question.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Kaggle — House Prices: Advanced Regression Techniques |
| File | `data/raw/train.csv` |
| Size | 1,460 houses · 81 columns |
| Target variable | `SalePrice` (USD) |

---

## Project Structure

```
house-price-analysis/
├── data/
│   └── raw/
│       └── train.csv
├── notebooks/
│   └── analysis.ipynb
├── visuals/
│   ├── saleprice_distribution.png
│   ├── saleprice_distribution_log.png
│   ├── grlivarea_vs_saleprice.png
│   ├── overallqual_vs_saleprice.png
│   ├── avg_saleprice_by_neighborhood.png
│   ├── median_saleprice_by_neighborhood.png
│   ├── yearbuilt_vs_saleprice.png
│   ├── totalSF_vs_saleprice.png
│   └── correlation_matrix.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Analysis Questions

| # | Question | Method | Stat Test |
|---|----------|--------|-----------|
| Q1 | What does the SalePrice distribution look like? | Histogram + log transform | Descriptive stats |
| Q2 | Does living area predict price? | Scatter plot | Pearson r |
| Q3 | How does quality rating affect price? | Box plot | Spearman r |
| Q4 | Which neighborhoods have the highest prices? | Horizontal bar chart | Descriptive (median) |
| Q5 | Has price changed over the years? | Scatter + regression line | Pearson r |
| Q6 | Does TotalSF predict better than GrLivArea alone? | Feature engineering + scatter | Pearson r comparison |
| Q7 | Which features correlate most with price? | Correlation heatmap | Pearson r matrix |

---

## Key Findings

**Top 5 features correlated with SalePrice:**

| Rank | Feature | Pearson r |
|------|---------|-----------|
| 1 | OverallQual | 0.791 |
| 2 | TotalSF *(engineered)* | 0.779 |
| 3 | GrLivArea | 0.709 |
| 4 | GarageCars | 0.640 |
| 5 | GarageArea | 0.623 |

A few things that stood out:

- **Quality beats size.** `OverallQual` (r = 0.791) outperforms raw square footage. Buyers are paying for quality rating more than square feet.
- **Feature engineering worked.** Creating `TotalSF = GrLivArea + TotalBsmtSF` improved correlation from 0.709 to 0.779 — a meaningful gain.
- **Location matters, but clusters.** Top neighborhoods (`NridgHt`, `NoRidge`, `StoneBr`) hit ~$300k median; bottom ones (`MeadowV`, `IDOTRR`) sit around $85k. But most neighborhoods cluster in the $150k–$170k range, where other features become the real differentiators.
- **The year-built analysis is flawed by inflation.** I flagged this in the notebook — comparing nominal prices across decades without CPI adjustment overstates the correlation. That's something to fix before using this for real.
- **Multicollinearity found:** `GarageCars` and `GarageArea` correlate at 0.882 with each other — they're measuring the same thing. Same with `GrLivArea` and `TotalSF` (0.880). These need to be deduplicated before any ML work.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/abrarawad/house-price-analysis.git
cd house-price-analysis

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/analysis.ipynb
```

> The notebook runs top to bottom. All visuals are saved automatically to `/visuals/`.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
```

---

## What's Next (If I Continue This)

- Adjust `YearBuilt` analysis for inflation using CPI data
- Try `TotalSF_weighted = GrLivArea + 0.5 * TotalBsmtSF` to see if basement area deserves less weight
- Encode `Neighborhood` as a categorical feature for ML
- Build a baseline regression model using the top features identified here

---

## License

MIT — see `LICENSE` for details.
