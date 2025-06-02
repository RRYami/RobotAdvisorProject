# Get the data from sqllite database and calculate the CAPM model

import sqlite3

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Connect to the database
conn = sqlite3.connect('db/database.db')

# Get the data from the database
try:
    df = pd.read_sql_query("SELECT * FROM marketData", conn)
    # Close the connection
    conn.close()
except Exception as e:
    print(e)

# Transform the data
print("Transforming the data")
cleaned_df = df.copy()[['Symbol', 'Date', 'Adjusted_close']]

# slit the data into multiple dataframes based on the symbol
print("Splitting the data based on the symbol")
symbols = cleaned_df['Symbol'].unique()
print(symbols)
dfs = []
for symbol in symbols:
    dfs.append(cleaned_df[cleaned_df['Symbol'] == symbol].copy())

# Calculate the returns
print("Calculating the log returns")
for df in dfs:
    df['log_return'] = np.log(df['Adjusted_close'] / df['Adjusted_close'].shift(1))
    df.dropna(inplace=True)

# Average historical returns
print("Calculating the average historical returns")
returns = []
for df in dfs:
    returns.append(df['log_return'].mean()*252)

returns = pd.DataFrame(returns, columns=['returns'], index=symbols)
print(returns)

# Calculate the market returns
print("Calculating the market returns")
market = dfs[-1].copy()
market.rename(columns={'log_return': 'market_log_return'}, inplace=True)
market.drop(columns=['Symbol'], inplace=True)
