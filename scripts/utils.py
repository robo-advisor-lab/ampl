
import gymnasium as gym
from gymnasium import spaces
# from stable_baselines3 import PPO
# from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib

import random
import plotly.io as pio
# import cvxpy as cp
# import matplotlib.pyplot as plt
import datetime as dt
# from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
# from stable_baselines3.common.vec_env import DummyVecEnv
# import torch
from flipside import Flipside
from dune_client.client import DuneClient

import os
from dotenv import load_dotenv

import tensorflow

import datetime as dt
from datetime import timedelta
import pytz  # Import pytz if using timezones

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import tensorflow as tf
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import joblib

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")
dune_api_key = os.getenv('DUNE_API_KEY')
fred_api_key = os.getenv("FRED_API_KEY")
dune = DuneClient(dune_api_key)

def flipside_api_results(api_key,query=None,query_result_set=None):
  
  flipside_api_key = api_key
  flipside = Flipside(flipside_api_key, "https://api-v2.flipsidecrypto.xyz")

  if query_result_set == None:
    query_result_set = flipside.query(query)
  # what page are we starting on?
  current_page_number = 1

  # How many records do we want to return in the page?
  page_size = 1000

  # set total pages to 1 higher than the `current_page_number` until
  # we receive the total pages from `get_query_results` given the 
  # provided `page_size` (total_pages is dynamically determined by the API 
  # based on the `page_size` you provide)

  total_pages = 2


  # we'll store all the page results in `all_rows`
  all_rows = []

  while current_page_number <= total_pages:
    results = flipside.get_query_results(
      query_result_set.query_id,
      page_number=current_page_number,
      page_size=page_size
    )

    total_pages = results.page.totalPages
    if results.records:
        all_rows = all_rows + results.records
    
    current_page_number += 1

  return pd.DataFrame(all_rows)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensorflow.random.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_time(df):
    time_cols = ['date','dt','hour','time','day','month','year','week','timestamp','date(utc)','block_timestamp']
    for col in df.columns:
        if col.lower() in time_cols and col.lower() != 'timestamp':
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
        elif col.lower() == 'timestamp':
            df[col] = pd.to_datetime(df[col], unit='ms')
            df.set_index(col, inplace=True)
    print(df.index)
    return df 

def clean_prices(prices_df):
    print('cleaning prices')
    # Pivot the dataframe
    prices_df = prices_df.drop_duplicates(subset=['DT', 'SYMBOL'])
    prices_df_pivot = prices_df.pivot(
        index='DT',
        columns='SYMBOL',
        values='PRICE'
    )
    prices_df_pivot = prices_df_pivot.reset_index()

    # Rename the columns by combining 'symbol' with a suffix
    prices_df_pivot.columns = ['DT'] + [f'{col}_Price' for col in prices_df_pivot.columns[1:]]
    
    print(f'cleaned prices: {prices_df_pivot}')
    return prices_df_pivot

def calculate_cumulative_return(portfolio_values_df):
    """
    Calculate the cumulative return for each column in the portfolio.
    
    Parameters:
    portfolio_values_df (pd.DataFrame): DataFrame with columns representing portfolio values
    
    Returns:
    pd.DataFrame: DataFrame with cumulative returns for each column
    """
    cumulative_returns = {}

    for col in portfolio_values_df.columns:
        print(f'col:{col}')
        initial_value = portfolio_values_df[col].iloc[0]
        print(f'initial_value: {initial_value}')
        final_value = portfolio_values_df[col].iloc[-1]
        print(f'final_value: {final_value}')
        cumulative_return = (final_value / initial_value) - 1
        cumulative_returns[col] = cumulative_return

    # Convert the dictionary to a DataFrame
    cumulative_returns_df = pd.DataFrame(cumulative_returns, index=['Cumulative_Return'])
    
    return cumulative_returns_df

def calculate_cagr(history):
    print(f'cagr history: {history}')
    #print(f'cagr history {history}')
    initial_value = history.iloc[0]
    #print(f'cagr initial value {initial_value}')
    final_value = history.iloc[-1]
    #print(f'cagr final value {final_value}')
    number_of_hours = (history.index[-1] - history.index[0]).total_seconds() / 3600
    #print(f'cagr number of hours {number_of_hours}')
    number_of_years = number_of_hours / (365.25 * 24)  # Convert hours to years
    #print(f'cagr number of years {number_of_years}')

    if number_of_years == 0:
        return 0

    cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
    cagr_percentage = cagr * 100
    return cagr

def calculate_beta(data, columnx, columny):
    X = data[f'{columnx}'].pct_change().dropna().values.reshape(-1, 1)  
    Y = data[f'{columny}'].pct_change().dropna().values
  
    # Check if X and Y are not empty
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Input arrays X and Y must have at least one sample each.")
        return 0

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, Y)

    # Output the beta
    beta = model.coef_[0]
    return beta

def fetch_and_process_tbill_data(api_url, api_key, data_key, date_column, value_column, date_format='datetime'):
    api_url_with_key = f"{api_url}&api_key={api_key}"

    response = requests.get(api_url_with_key)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[data_key])
        
        if date_format == 'datetime':
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
        df[value_column] = df[value_column].astype(float)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
    
def set_global_seed(env, seed=20):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env.seed(seed)
    env.action_space.seed(seed)

def data_processing(df,dropna=True):
    clean_df = clean_prices(df)
    clean_df = to_time(clean_df)
    if dropna == True:
        clean_df = clean_df.dropna(axis=1, how='any')

    if '__row_index' in clean_df.columns:
        clean_df.drop(columns=['__row_index'], inplace=True)

    return clean_df

def pull_data(api=False):
    crypto_tickers = ['BTC-USD','ETH-USD','PAXG-USD']
    trad_tickers = ['^GSPC']

    vol_path = 'data/vol.csv'
    prices_path = 'data/prices.csv'
    ampl_vol_path = 'data/ampl_vol.csv'
    target_data_path = 'data/target.csv'
    rebase_rates_path = 'data/rebase_rates.csv'
    usdc_vol_path = 'data/usdc_vol.csv'
    usdt_vol_path = 'data/usdt_vol.csv'
    dai_vol_path = 'data/dai_vol.csv'
    stablecoin_volume_path = 'data/hourly_stable_vol.csv'
    # dune_prices_path = 'data/dune_prices.csv'
    
    if api==False:
        hourly_dex_vol = pd.read_csv(vol_path).dropna()
        hourly_prices = pd.read_csv(prices_path).dropna()
        hourly_ampl_vol = pd.read_csv(ampl_vol_path).dropna()
        target_data = pd.read_csv(target_data_path).dropna()
        rebase_rates_df = pd.read_csv(rebase_rates_path).dropna()
        usdc_vol = pd.read_csv('data/usdc_vol.csv').dropna()
        usdt_vol = pd.read_csv('data/usdt_vol.csv').dropna()
        dai_vol = pd.read_csv('data/dai_vol.csv').dropna()
        stablecoin_volume = pd.read_csv('data/hourly_stable_vol.csv').dropna()

        yfinance_data = {}
        for asset in trad_tickers+crypto_tickers:
            yfinance_data[asset] = pd.read_csv(f'data/{asset}_data.csv')
        
    else:
        target_data = dune_api_results(4162876,True,target_data_path)
        rebase_rates_df = dune_api_results(4162239, True, rebase_rates_path)
        # hourly_prices = dune_api_results(4166429, True, dune_prices_path)
        hourly_prices = pd.read_csv(prices_path).dropna()
        hourly_dex_vol = pd.read_csv(vol_path).dropna()
        hourly_ampl_vol = pd.read_csv(ampl_vol_path).dropna()
        usdc_vol = pd.read_csv('data/usdc_vol.csv').dropna()
        usdt_vol = pd.read_csv('data/usdt_vol.csv').dropna()
        dai_vol = pd.read_csv('data/dai_vol.csv').dropna()
        stablecoin_volume = pd.read_csv('data/hourly_stable_vol.csv').dropna()
        
        yfinance_data = {}
        for asset in trad_tickers+crypto_tickers:
            asset_df = get_prices(asset, period='10y')
            asset_df[f'{asset}_price'] = asset_df['Close']
            asset_df[f'{asset}_price'].to_csv(f'data/{asset}_data.csv')
            yfinance_data[asset] = asset_df[f'{asset}_price']
    
    data_struct = {
        'hourly_dex_vol': hourly_dex_vol,
        'hourly_prices': hourly_prices,
        'hourly_ampl_vol':hourly_ampl_vol,
        'target_data': target_data,
        'rebase_rates_df': rebase_rates_df,
        'yfinance_data': yfinance_data,
        'usdc_vol': usdc_vol,
        'usdt_vol': usdt_vol,
        'dai_vol': dai_vol,
        'stablecoin_volume': stablecoin_volume
    }

    return data_struct

def get_prices(asset,period):
    y_ob = yf.Ticker(asset)
    return y_ob.history(period=period)
     
def dune_api_results(query_num, save_csv=False, csv_path=None):
    results = dune.get_latest_result(query_num)
    df = pd.DataFrame(results.result.rows)

    if save_csv and csv_path:
        df.to_csv(csv_path, index=False)
    return df

def train_ridge_model(test_data,target,features):

    top_corr = features

    X = test_data[top_corr]  # Features
    y = test_data[target]  # Target variable

    X = X.select_dtypes(include=['float64', 'int64'])  # Keep only numeric columns

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=20)

    # Initialize the model
    model = Ridge()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')

    print(f'Saving Model...')
    joblib.dump(model, 'pkl/ridge_model.pkl')

def plot_simulation(simulation_results, actual_values):
    # Create the figure
    fig = go.Figure()

    # Add the simulation (predicted) results
    fig.add_trace(go.Scatter(x=simulation_results.index, 
                             y=simulation_results['Simulated AMPL Price'], 
                             mode='lines', name='Predicted Price', 
                             line=dict(color='blue')))

    # Add the actual values
    fig.add_trace(go.Scatter(x=actual_values.index, 
                             y=actual_values['Historical AMPL Price'], 
                             mode='lines', name='Actual Price', 
                             line=dict(color='orange')))

    # Calculate metrics
    r2, mae, rmse = calculate_metrics(simulation_results, actual_values)

    # Add metrics as annotations
    fig.add_annotation(text=f"R²: {r2:.2f}", xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False, font=dict(size=12))
    fig.add_annotation(text=f"MAE: {mae:.2f}", xref="paper", yref="paper", x=0.05, y=0.90, showarrow=False, font=dict(size=12))
    fig.add_annotation(text=f"RMSE: {rmse:.2f}", xref="paper", yref="paper", x=0.05, y=0.85, showarrow=False, font=dict(size=12))

    # Update layout
    fig.update_layout(title='Simulation vs Actual Price',
                      xaxis_title='Time',
                      yaxis_title='Price',
                      legend=dict(x=0.1, y=1.1),
                      width=900, height=500)

    fig.show()

def calculate_metrics(simulation_results, actual_values):
    # Extract the predicted and actual values
    y_pred = simulation_results[simulation_results.columns[0]].values
    y_actual = actual_values.values

    # Calculate metrics
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    return r2, mae, rmse
    