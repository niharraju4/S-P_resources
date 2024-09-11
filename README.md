
# Table of Contents
1. [Introduction](#introduction)
2. [Libraries](#libraries)
3. [Loading Data](#loading-data)
4. [Data Exploration](#data-exploration)
5. [Moving Averages](#moving-averages)
6. [Visualization](#visualization)
7. [Resampling Analysis](#resampling-analysis)
8. [Correlation Analysis](#correlation-analysis)
9. [Results](#results)

## Introduction
This documentation provides a detailed guide to analyze stock data for several tech companies using Python. The process involves loading the data, exploring the data, calculating moving averages, visualizing the data, performing resampling analysis, and conducting correlation analysis.

**Author: Nihar Raju**

## Libraries
The following libraries are used in this code:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib.pyplot`: For plotting.
- `seaborn`: For statistical data visualization.
- `glob`: For file path manipulation.
- `plotly.express`: For interactive plots.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import plotly.express as px
```

## Loading Data
The data files are loaded from the specified directory, and the data is concatenated into a single DataFrame.

```python
# List of company data files
company_list = [
    r'N:\\Personal_Projects\\Machine_learning_projects\\S&P_resources\\individual_stocks_5yr\\AAPL_data.csv',
    r'N:\\Personal_Projects\\Machine_learning_projects\\S&P_resources\\individual_stocks_5yr\\AMZN_data.csv',
    r'N:\\Personal_Projects\\Machine_learning_projects\\S&P_resources\\individual_stocks_5yr\\GOOG_data.csv',
    r'N:\\Personal_Projects\\Machine_learning_projects\\S&P_resources\\individual_stocks_5yr\\MNST_data.csv'
]

# Initialize an empty DataFrame
all_data = pd.DataFrame()

# Load and concatenate the data
for file in company_list:
    current_df = pd.read_csv(file)
    all_data = pd.concat([all_data, current_df], ignore_index=True)

# Display the shape and first few rows of the DataFrame
all_data.shape
all_data.head()
all_data.tail()
```

## Data Exploration
The data is explored to understand its structure, check for missing values, and convert the date column to datetime format.

```python
# Display unique company names
all_data['Name'].unique()

# Check for missing values
all_data.isnull()
all_data.isnull().sum()

# Display data types
all_data.dtypes

# Convert the 'date' column to datetime format
all_data['date'] = pd.to_datetime(all_data['date'])

# Display unique company names
tech_list = all_data['Name'].unique()
tech_list
```

## Moving Averages
Moving averages are calculated for the closing prices of the stocks.

```python
# Calculate moving averages
ma_day = [10, 20, 50, 100, 200]
new_data = all_data.copy()

for ma in ma_day:
    new_data['close_' + str(ma)] = new_data['close'].rolling(ma).mean()

# Display the first few rows of the DataFrame with moving averages
new_data.head(15)
new_data.tail(15)
```

## Visualization
The data is visualized using scatter plots and line plots.

```python
# Plot closing prices for each company
plt.figure(figsize=(20, 12))

for index, company in enumerate(tech_list, 1):
    plt.subplot(2, 2, index)
    filter1 = new_data['Name'] == company
    df = new_data[filter1]
    plt.plot(df['date'], df['close'])
    plt.title(company)

# Plot moving averages for each company
plt.figure(figsize=(20, 12))

for index, company in enumerate(tech_list, 1):
    plt.subplot(2, 2, index)
    filter1 = new_data['Name'] == company
    df = new_data[filter1]
    df[['close_10', 'close_20', 'close_50']].plot(ax=plt.gca())
    plt.title(company)
```

## Resampling Analysis
Resampling analysis is performed on the closing prices of Apple stock.

```python
# Load Apple stock data
apple = pd.read_csv(r'N:\\Personal_Projects\\Machine_learning_projects\\S&P_resources\\individual_stocks_5yr\\AAPL_data.csv')

# Convert the 'date' column to datetime format
apple['date'] = pd.to_datetime(apple['date'])

# Set the 'date' column as the index
apple.set_index('date', inplace=True)

# Calculate daily return
apple['Daily return(in %)'] = apple['close'].pct_change() * 100

# Plot daily return
px.line(apple, x="date", y="Daily return(in %)")

# Resample closing prices
apple['close'].resample('M').mean().plot()  # Monthly
apple['close'].resample('Y').mean().plot()  # Yearly
apple['close'].resample('Q').mean().plot()  # Quarterly
```

## Correlation Analysis
Correlation analysis is performed between the closing prices of the tech companies.

```python
# Load data for each company
appl = pd.read_csv(company_list[0])
amz = pd.read_csv(company_list[1])
goog = pd.read_csv(company_list[2])
msft = pd.read_csv(company_list[3])

# Create a DataFrame with closing prices
closing_price = pd.DataFrame()
closing_price['apple_close'] = appl['close']
closing_price['amazon_close'] = amz['close']
closing_price['google_close'] = goog['close']
closing_price['microsoft_close'] = msft['close']

# Display the first few rows of the DataFrame
closing_price.head()

# Plot pairplot
sns.pairplot(closing_price)

# Calculate correlation matrix
corr_matrix = closing_price.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True)

# Calculate daily return for each company
for col in closing_price.columns:
    closing_price[col + '_pct_change'] = (closing_price[col] - closing_price[col].shift(1)) / closing_price[col].shift(1) * 100

# Create a DataFrame with daily returns
closing_pct_change = closing_price[['apple_close_pct_change', 'amazon_close_pct_change', 'google_close_pct_change', 'microsoft_close_pct_change']]

# Plot pairplot for daily returns
g = sns.PairGrid(data=closing_pct_change)
g.map_diag(sns.histplot)
g.map_lower(sns.scatterplot)
g.map_upper(sns.kdeplot)

# Calculate correlation matrix for daily returns
corr_matrix_pct_change = closing_pct_change.corr()

# Display the correlation matrix for daily returns
corr_matrix_pct_change
```

## Results
The results include the visualizations and correlation matrices for the closing prices and daily returns of the tech companies.

### Visualizations
- **Closing Prices**: Line plots showing the closing prices of each company.
- **Moving Averages**: Line plots showing the moving averages of the closing prices for each company.
- **Daily Return**: Line plot showing the daily return of Apple stock.
- **Resampling Analysis**: Line plots showing the monthly, yearly, and quarterly resampled closing prices of Apple stock.
- **Pairplot**: Pairplot showing the relationships between the closing prices of the tech companies.
- **Heatmap**: Heatmap showing the correlation matrix for the closing prices of the tech companies.
- **Pairplot for Daily Returns**: Pairplot showing the relationships between the daily returns of the tech companies.

### Correlation Matrices
- **Closing Prices**: Correlation matrix showing the correlations between the closing prices of the tech companies.
- **Daily Returns**: Correlation matrix showing the correlations between the daily returns of the tech companies.


**Author: Nihar Raju**
