# House Price Notebook — Line by Line Code Explanations
### Every unique construct explained word by word

---

## Cell 1 — Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

All five lines are covered in the Project 01 file (Cell 3). Nothing new here.

---

## Cell 2 — Loading Data

```python
df = pd.read_csv('../data/raw/train.csv')
df.head()
```

**`df.head()`**

A pandas method that returns the first 5 rows of the DataFrame and displays them
as a formatted table in the notebook. No arguments needed — by default it shows
5 rows. If you want a different number, pass it as an argument: `df.head(10)`
shows 10 rows. Useful as a quick sanity check after loading — you can see the
column names, data types, and a sample of values before doing anything else.

The `pd.read_csv(...)` and `df =` assignment are covered in Project 01.

---

## Cell 3 — Quick Overview

```python
df.describe()
```

**`df.describe()`**

A pandas method that computes summary statistics for every numeric column in the
DataFrame simultaneously. For each column it returns: count (number of
non-missing values), mean, standard deviation, minimum, 25th percentile (Q1),
median (Q2 / 50th percentile), 75th percentile (Q3), and maximum. The output is
a new DataFrame where each row is one of these statistics and each column is a
feature from your data. Useful for quickly spotting outliers (huge max vs. mean
gap), scale differences between columns, and missing data (a count lower than
the total rows means missing values exist).

---

## Cell 4 — Q1: Create log_saleprice and Plot Distribution

```python
saleprice = df['SalePrice'].dropna()
df['log_saleprice'] = np.log1p(df['SalePrice'])
```

**`np.log1p(df['SalePrice'])`**

`np.log1p` is a numpy function that computes the natural logarithm of (1 + x)
for every value x in the input. The reason for adding 1 before taking the log is
safety: `log(0)` is mathematically undefined (negative infinity), so if any
value is zero, `np.log(x)` would produce `-inf` and break the column.
`np.log1p(0)` returns 0, which is safe. For house prices (which are never zero or
negative), the `+1` makes virtually no difference numerically, but it is the
standard habit to use `log1p` on count or price data.

`df['log_saleprice'] =` creates a brand new column in the DataFrame. If that
column name does not exist yet, pandas adds it. If it does exist, it overwrites it.

---

## Cell 5 — Q1: Plot Raw Distribution

```python
plt.figure(figsize=(8,5))
plt.hist(saleprice, bins=30, color='lightgreen', label='Sale Price')
plt.title("Sale Price Distribution")
plt.xlabel('Sale Price')
plt.ylabel('Number of Houses')
plt.legend()
plt.tight_layout()
plt.savefig("../visuals/saleprice_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean: {df['SalePrice'].mean():.2f}")
print(f"Median: {df['SalePrice'].median():.2f}")

if df['SalePrice'].mean() > df['SalePrice'].median():
    print("Here Mean > Median: The distribution is right-skewed.")
elif df['SalePrice'].mean() < df['SalePrice'].median():
    print("Here Mean < Median: The distribution is left-skewed.")
else:
    print("Here Mean = Median: The distribution is symmetric.")
```

`plt.hist()`, `plt.legend()`, `plt.savefig()` and the f-string formatting are all
covered in Project 01. The new constructs here are:

**`elif`**

Python allows chaining conditions together using `elif` (short for "else if").
The interpreter checks the `if` first. If that is False, it checks the `elif`. If
that is also False, it falls through to `else`. You can have as many `elif` blocks
as you need between `if` and `else`. Here you need three branches because
mean > median, mean < median, and mean == median are three mutually exclusive
outcomes.

---

## Cell 6 — Q1: Plot Log Distribution and Check Skewness

```python
plt.figure(figsize=(8,5))
plt.hist(df['log_saleprice'], bins=30, color='lightblue', label='Log(Sale Price)')
plt.title("Log(Sale Price) Distribution")
plt.xlabel('Log(Sale Price)')
plt.ylabel('Number of Houses')
plt.legend()
plt.tight_layout()
plt.savefig("../visuals/saleprice_distribution_log.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean of LogSalePrice: {df['log_saleprice'].mean():.3f}")
print(f"Median of LogSalePrice: {df['log_saleprice'].median():.3f}")
print(f"Skewness after log: {df['log_saleprice'].skew():.3f}")
```

**`:.3f` format code**

Same family as `:.2f` from Project 01, but three decimal places instead of two.
Used here because skewness values are often small (e.g. 0.121) and two decimal
places would round that to 0.12, losing useful precision.

**`.skew()`**

A pandas method that computes the skewness of a Series — a statistical measure
of how asymmetric the distribution is. A value of 0 means perfectly symmetric.
Positive values mean the distribution has a long right tail (right-skewed).
Negative values mean a long left tail (left-skewed). Values between -0.5 and 0.5
are generally considered close enough to symmetric for modeling purposes.

---

## Cell 7 — Q2: Scatter Plot

```python
living_price_df = df[['GrLivArea', 'SalePrice']].dropna()

plt.figure(figsize=(8,5))
sns.scatterplot(data=living_price_df, x='GrLivArea', y='SalePrice', color='orange', alpha=0.7)
plt.title("GrLivArea vs SalePrice")
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.tight_layout()
plt.savefig("../visuals/grlivarea_vs_saleprice.png", dpi=150, bbox_inches='tight')
plt.show()
```

**`sns.scatterplot(data=living_price_df, x='GrLivArea', y='SalePrice', color='orange', alpha=0.7)`**

`sns.scatterplot()` draws a scatter plot — one dot per row, with the x position
determined by one column and the y position by another.

`data=living_price_df` tells seaborn which DataFrame to use. When you provide
`data=`, you can refer to columns by name in the `x=` and `y=` arguments directly.

`x='GrLivArea'` specifies which column goes on the horizontal axis.

`y='SalePrice'` specifies which column goes on the vertical axis.

`color='orange'` sets the dot color. A single color string applies to all dots.

`alpha=0.7` controls transparency (0 = invisible, 1 = solid). Setting it below 1
means where dots overlap, the overlapping region appears darker. This is called
overplotting control — without it, a dense region of dots all look the same solid
color, and you cannot tell if 5 dots overlap there or 500.

---

## Cell 8 — Q2: Pearson Correlation

```python
r_value, p_value = stats.pearsonr(living_price_df['GrLivArea'], living_price_df['SalePrice'])
if p_value < 0.05:
    print(f"Significant correlation: r = {r_value:.3f}, p = {p_value:.3e}")
elif p_value >= 0.05:
    print(f"No significant correlation: r = {r_value:.3f}, p = {p_value:.3e}")
```

**`stats.pearsonr(x, y)`**

A scipy function that computes Pearson's r — the linear correlation coefficient
between two continuous variables. It measures how closely the two variables move
together in a straight line. Returns two values: r (the correlation coefficient,
ranging from -1 to +1) and p (the p-value for the hypothesis test that the true
correlation is zero).

`r_value, p_value =` is tuple unpacking — same pattern as in the chi-square and
t-test cells in Project 01.

**`:.3e` format code**

`e` means scientific notation. `.3` means three decimal places in the mantissa.
So a value like `0.000000000000000000000000000000000000000000000000000000000000000000000000
000000000000000000000000000000000000000000000000000000000000000000000000000000
000000000000000000000000000000000000000000000000000000000000000452` would display
as `4.520e-223`. Scientific notation is used here because p-values from large
datasets can be so small that standard decimal formatting would print dozens of
leading zeros, making them unreadable. `:.3e` compresses any p-value down to a
short readable form.

---

## Cell 9 — Q3: Box Plot

```python
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='OverallQual', y='log_saleprice', color='lightcoral')
plt.yscale('log')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title("Overall Quality vs SalePrice")
plt.xlabel('Overall Quality')
plt.ylabel('Sale Price(log scale)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../visuals/overallqual_vs_saleprice.png", dpi=150, bbox_inches='tight')
plt.show()
```

**`sns.boxplot(data=df, x='OverallQual', y='log_saleprice', color='lightcoral')`**

`sns.boxplot()` draws one box for each unique value in the `x=` column. Each box
shows the distribution of the `y=` column within that group: the box spans Q1 to
Q3, the line inside is the median, the whiskers extend to 1.5× the IQR, and dots
beyond that are outliers.

`x='OverallQual'` — the grouping variable. Since OverallQual has 10 unique values
(1 through 10), you get 10 boxes side by side.

`y='log_saleprice'` — the value to summarize within each group. Note that
`log_saleprice` is already a log-transformed column. Combined with `plt.yscale('log')`
below, this is a double log (a bug — see the issues noted in the project review).

`color='lightcoral'` — a single color for all boxes. Unlike `sns.scatterplot`
where `color` controls dot color, here it controls the box fill color.

`plt.yscale('log')` and `plt.grid(axis='y', linestyle='--', alpha=0.7)` are
covered in Project 01 (Cell 29).

---

## Cell 10 — Q3: Spearman Correlation

```python
r_value_2, p_value_2 = stats.spearmanr(df['OverallQual'], df['log_saleprice'])
if p_value_2 < 0.05:
    print(f"Significant correlation: r = {r_value_2:.3f}, p = {p_value_2:.3e}")
elif p_value_2 >= 0.05:
    print(f"No significant correlation: r = {r_value_2:.3f}, p = {p_value_2:.3e}")
```

**`stats.spearmanr(x, y)`**

Spearman's rank correlation. Where Pearson measures linear relationship between
two continuous variables, Spearman converts both variables to ranks first and then
measures how well those ranks agree. This makes it appropriate for ordinal
variables like OverallQual, where the numbers represent order (4 is better than 3)
but not equal spacing (the gap between 4 and 5 is not necessarily the same as the
gap between 8 and 9).

The return values are the same as `pearsonr`: (r, p). The r value has the same
interpretation — closer to ±1 means stronger monotonic relationship. Tuple
unpacking is the same pattern.

```python
print(df['OverallQual'].value_counts().sort_index())
```

**`.value_counts()`**

A pandas Series method that counts how many times each unique value appears.
Returns a new Series where the index is the unique values and the values are the
counts. By default it sorts by count descending (most common value first).

**`.sort_index()`**

Sorts the Series by its index (the unique values) instead of by count. Here that
means sorting by quality rating 1 through 10 in order, so you can read the sample
sizes for each rating level sequentially.

---

## Cell 11 — Q4: Neighborhood Bar Chart

```python
df['Neighborhood'].unique()
```

**`.unique()`**

A pandas Series method that returns an array of all distinct values in the column,
with no duplicates. Unlike `.value_counts()` which gives you counts, `.unique()`
just tells you what values exist. Useful for checking how many categories a column
has before deciding how to visualize or encode it.

```python
saleprice_by_neighborhood = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=saleprice_by_neighborhood.values, y=saleprice_by_neighborhood.index, palette='mako')
plt.xticks(rotation=0)
plt.title("Median Sale Price by Neighborhood")
plt.xlabel('Median Sale Price')
plt.ylabel('Neighborhood')
plt.tight_layout()
plt.savefig("../visuals/median_saleprice_by_neighborhood.png", dpi=150, bbox_inches='tight')
plt.show()
```

**`df.groupby('Neighborhood')['SalePrice'].median()`**

Same groupby pattern as Project 01 but using `.median()` instead of `.mean()`. As
covered in Q1, median is preferred here because SalePrice is right-skewed — a few
very expensive houses would pull the mean upward and misrepresent the typical price
for that neighborhood.

**`sns.barplot(x=saleprice_by_neighborhood.values, y=saleprice_by_neighborhood.index, palette='mako')`**

`sns.barplot()` draws a bar chart. The unusual thing here is that `x` and `y` are
swapped from what you might expect — `x` gets the numeric values and `y` gets the
category names. This produces a horizontal bar chart instead of a vertical one.
Seaborn determines the orientation from which axis has the numeric values: numbers
on x = horizontal bars, numbers on y = vertical bars.

`.values` extracts the underlying numpy array from the Series (just the numbers,
no index). `.index` extracts the index of the Series (the neighborhood names, as
an array). You need to pass them separately because `sns.barplot` cannot infer both
from a Series alone when you're specifying axes manually.

`palette='mako'` applies a sequential color palette from seaborn's built-in palette
collection. `'mako'` is a dark-to-light blue-green palette. Unlike passing a single
`color=` string, a `palette=` applies different shades across the bars based on
their position — useful when you have many categories and want visual variation.

---

## Cell 12 — Q5: Regression Plot

```python
year_saleprice_df = df[['YearBuilt', 'SalePrice']].dropna()

plt.figure(figsize=(8,5))
sns.regplot(data=year_saleprice_df, x='YearBuilt', y='SalePrice',
            line_kws={"color": "#F4991A"}, scatter_kws={"color": "#344F1F"})
plt.title("YearBuilt vs SalePrice")
plt.xlabel('Year Built')
plt.ylabel('Sale Price')
plt.tight_layout()
plt.savefig("../visuals/yearbuilt_vs_saleprice.png", dpi=150, bbox_inches='tight')
plt.show()
```

**`sns.regplot()`**

A seaborn function that draws a scatter plot and fits a linear regression line on
top of it in one call. The shaded band around the line is the 95% confidence
interval — it shows the uncertainty in the regression estimate. Narrower band =
more data, more confident. Wider band = less data or more spread.

**`line_kws={"color": "#F4991A"}`**

`line_kws` is short for "line keyword arguments." It is a dictionary of styling
options that get passed through to the underlying matplotlib line-drawing function.
`{"color": "#F4991A"}` sets the regression line color using a hex color code.
`#F4991A` is an orange tone.

**`scatter_kws={"color": "#344F1F"}`**

Same idea but for the scatter dots. `{"color": "#344F1F"}` is a dark green hex
code. `scatter_kws` and `line_kws` let you style the two parts of the plot
(dots and line) independently, since seaborn combines both into one function call.

**Hex color codes**

A hex color code starts with `#` followed by 6 hexadecimal digits. The digits come
in three pairs: `RR GG BB` — red, green, blue, each pair ranging from `00` (none)
to `FF` (maximum, which is 255 in decimal). `#F4991A` breaks down as: red=F4
(high), green=99 (medium), blue=1A (very low) — which produces orange. You can
look up any color's hex code using any color picker tool online.

---

## Cell 13 — Q5: Pearson Correlation (YearBuilt)

```python
r_value3, p_value3 = stats.pearsonr(year_saleprice_df['YearBuilt'], year_saleprice_df['SalePrice'])
if p_value3 < 0.05:
    print(f"Significant correlation: r = {r_value3:.3f}, p = {p_value3:.3e}")
elif p_value3 >= 0.05:
    print(f"No significant correlation: r = {r_value3:.3f}, p = {p_value3:.3e}")
```

Same `stats.pearsonr()` and f-string patterns as Cell 8. The only difference is
different variable names (`r_value3`, `p_value3`) to avoid overwriting the earlier
results. Using numbered suffixes (`r_value`, `r_value_2`, `r_value3`) is a common
but fragile habit — in a longer project you would usually use more descriptive
names like `r_living` and `r_yearbuilt`.

---

## Cell 14 — Q6: Feature Engineering

```python
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
```

**`df['TotalBsmtSF'].fillna(0)`**

`.fillna(value)` replaces every NaN in the Series with the given value. Here that
value is `0`, because a missing basement square footage means the house has no
basement — so 0 is semantically correct. This is covered in Project 01's
correlation heatmap cell, but the choice of replacement value is different: there
you used `.median()` for Age because "no age" means "unknown age." Here you use
literal `0` because "no basement square footage" means "no basement." The right
fill value always depends on what the missing data means, not a fixed rule.

**`df['GrLivArea'] + df['TotalBsmtSF']`**

Column arithmetic in pandas. When you add two Series together with `+`, pandas
aligns them by index and adds the values row by row. Every house gets its
`GrLivArea` value added to its `TotalBsmtSF` value, producing a new Series of the
same length. This is called vectorized arithmetic — you do not need a for loop.

`df['TotalSF'] =` assigns that result as a new column in the DataFrame.

---

## Cell 15 — Q6: Regression Plot (TotalSF)

```python
plt.figure(figsize=(8,5))
sns.regplot(data=df, x='TotalSF', y='SalePrice',
            line_kws={"color": "#F3C623"}, scatter_kws={"color": "#10375C"})
plt.title("TotalSF vs SalePrice")
plt.xlabel('Total Square Feet')
plt.ylabel('Sale Price')
plt.tight_layout()
plt.savefig("../visuals/totalSF_vs_saleprice.png", dpi=150, bbox_inches='tight')
plt.show()
```

Same `sns.regplot()` pattern as Cell 12. No new constructs — different column
names and hex color codes only.

---

## Cell 16 — Q7: Correlation Heatmap

```python
numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.drop(columns=['Id','log_saleprice'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
            fmt='.2f', linewidths=0.5, center=0, square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig("../visuals/correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
```

`df.select_dtypes()`, `df.drop(columns=[...])`, `numeric_df.corr()`, and
`sns.heatmap()` are all covered in detail in Project 01 (Cell 32). The differences
here are:

`drop(columns=['Id', 'log_saleprice'])` — dropping two columns at once by passing
a list. In Project 01 only `PassengerId` was dropped. The list can be as long as
you need; pandas removes all named columns in one call.

`annot=False` — this is the opposite of Project 01's `annot=True`. With 80+
numeric columns the heatmap cell count would be enormous, and printing a number
inside every cell would be completely unreadable. Turning annotations off means
you rely on the color alone to read the matrix.

---

## Cell 17 — Q7: Print Correlation Values

```python
print("=== Correlation Analysis Results ===\n")
print("Correlation with SalePrice:")
print(correlation_matrix['SalePrice'].sort_values(ascending=False))
print("\n")

corr_values = correlation_matrix.unstack().sort_values(ascending=False)
corr_values = corr_values[corr_values < 0.999]
print("Top Correlations (excluding 1.0):")
print(corr_values.head(5))
print("\n")

print("Top Negative Correlations:")
print(corr_values.tail(5))
```

`correlation_matrix.unstack()`, `.sort_values(ascending=False)`, `.head()`,
`.tail()`, and the `< 0.999` boolean filter are all covered in Project 01
(Cell 33). The only difference is `.head(5)` and `.tail(5)` instead of 10 —
this dataset has fewer features worth inspecting in detail.

---

## Quick Reference — New Patterns in This Notebook

**New data access patterns**

`df['NewCol'] = df['A'] + df['B']` creates a new column using arithmetic on two
existing columns. Pandas aligns by index and operates row by row — no loop needed.

`series.unique()` returns all distinct values with no duplicates. Different from
`.value_counts()` which returns value-count pairs.

**New aggregation patterns**

`df.groupby('Col')['Other'].median()` — same groupby structure as Project 01 but
using median instead of mean. Use median when the target column is skewed.

`series.skew()` measures distribution asymmetry. Zero = symmetric, positive = right
tail, negative = left tail.

**New transformation patterns**

`np.log1p(series)` applies log(1 + x) to every value. Preferred over `np.log()`
for price and count data because it handles zeros safely.

`series.fillna(0)` replaces NaNs with zero. Use when the missing value is
genuinely zero (no basement), not when it is unknown (unknown age — use median
instead).

**New statistical test patterns**

`stats.pearsonr(x, y)` measures linear correlation between two continuous
variables. Returns (r, p). Use when both variables are continuous and the
relationship appears linear.

`stats.spearmanr(x, y)` measures rank correlation between two variables. Returns
(r, p). Use when at least one variable is ordinal.

**New plotting patterns**

`sns.scatterplot(data=df, x='Col1', y='Col2', alpha=0.7)` — two continuous
variables, one dot per row. `alpha` controls overplotting.

`sns.boxplot(data=df, x='CatCol', y='NumCol')` — one box per category showing
the distribution of the numeric column within each group.

`sns.regplot(x='Col1', y='Col2', data=df, line_kws={...}, scatter_kws={...})` —
scatter plot with a fitted regression line. Style the line and dots separately
using their respective `_kws` dictionaries.

`sns.barplot(x=series.values, y=series.index, palette='mako')` — horizontal bar
chart when `x` gets the numeric values and `y` gets the category labels.

`plt.yscale('log')` switches the y-axis to log scale. Use when the data spans
multiple orders of magnitude.

`:.3e` in an f-string formats as scientific notation with 3 decimal places. Use
for very small p-values.

---

*House Price EDA Project 02 | May 2026*
