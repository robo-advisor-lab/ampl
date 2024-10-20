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

import joblib

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta
import pytz  # Import pytz if using timezones

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.express as px

from scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data,set_random_seed,data_processing,train_ridge_model,plot_simulation,calculate_metrics
from sql_queries.queries import token_prices, volume
from scripts.data_processing import process_data
from models.simulation import RebaseSim

def get_dataset():
    features = ['AMPL_Price', 'rebase_rate_30d_rolling avg', 'rebase_rate_30_d_std_dev',
        'rebase_rate_7d_rolling avg', 'rebase_rate_7_d_std_dev',
        'active_addresses_7d_rolling avg', 'supply_30_d_std_dev',
        'active_addresses', 'active_addresses_30d_rolling avg',
        'active_addresses_7_d_lag', 'rebase_rate', 'rebase_rate_7_d_lag',
        'active_addresses_30_d_std_dev', 'active_addresses_30_d_lag',
        'active_addresses_7_d_std_dev', 'rebase_rate_30_d_lag',
        'target_price_30d_rolling avg', 'target_price_7d_rolling avg',
        'target_price', 'target_price_7_d_lag', 'supply_7_d_std_dev',
        'target_price_30_d_lag', 'STABLECOIN_VOLUME_7d_rolling avg',
        'STABLECOIN_VOLUME_30d_rolling avg', 'supply', 'supply_7d_rolling avg',
        'supply_7_d_lag', 'supply_30d_rolling avg', 'supply_30_d_lag']

    dataset,features = process_data(api=False, features=features)

    return dataset, features

def main(test_data,target,top_corr,train_model=False,start_date=None,end_date=None,actions=None):
    if train_model == True:
        train_ridge_model(test_data,target,top_corr)
    
    ridge_model = joblib.load('pkl/ridge_model.pkl')

    sim = RebaseSim(df=test_data, target=target, features=top_corr, model=ridge_model,
                start_date=start_date, end_date=end_date)
    
    sim.run_simulation(cycle=1,actions=actions)

    pred_df = sim.get_prediction_df()

    pred_set = set(pred_df.columns.to_list())
    test_set = set(test_data[[target]+top_corr.to_list()].columns.to_list())

    # Find columns in pred_df but not in test_data
    missing_in_test = pred_set - test_set

    # Find columns in test_data but not in pred_df
    missing_in_pred = test_set - pred_set

    print("Columns in pred_df but not in test_data:")
    print(missing_in_test)

    print("\nColumns in test_data but not in pred_df:")
    print(missing_in_pred)

    actual_rolling = test_data[['rebase_rate_30d_rolling avg','rebase_rate']][(test_data.index>=pred_df.index.min())&(test_data.index<=pred_df.index.max())]
    actual_values = test_data[target][(test_data.index>=pred_df.index.min())&(test_data.index<=pred_df.index.max())].to_frame('Historical AMPL Price')
    actual_rebase = test_data['rebase_rate'][(test_data.index>=pred_df.index.min())&(test_data.index<=pred_df.index.max())].to_frame('Historical Rebase Rates')

    pred_df.rename(columns={"AMPL_Price":"Simulated AMPL Price"},inplace=True)
    test_data_filtered = test_data[(test_data.index>=pred_df.index.min())&(test_data.index<=pred_df.index.max())]

    combined_df = pd.DataFrame({
    'Model Rebase Rate 30d': pred_df['rebase_rate_30d_rolling avg'],
    'Actual Rebase Rate 30d': test_data_filtered['rebase_rate_30d_rolling avg']
    })

    # Reset the index to have a column for time if needed
    # combined_df.reset_index(inplace=True)

    # Plot using Plotly Express
    fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns,
                title='Rebase Rate Comparison 30d avg',
                labels={'index': 'Date', 'value': 'Rate', 'variable': 'Series'},
                markers=True)

    # Show the plot
    fig.show()

    combined_df = pd.DataFrame({
    'Model Rebase Rate': pred_df['rebase_rate'],
    'Actual Rebase Rate': test_data_filtered['rebase_rate']
    })

    # Reset the index to have a column for time if needed
    # combined_df.reset_index(inplace=True)

    # Plot using Plotly Express
    fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns,
                title='Rebase Rate Comparison',
                labels={'index': 'Date', 'value': 'Rate', 'variable': 'Series'},
                markers=True)

    # Show the plot
    fig.show()

    combined_df = pd.DataFrame({
    'Pred Supply': pred_df['supply'],
    'Actual Supply': test_data_filtered['supply']
    })

    # Reset the index to have a column for time if needed
    # combined_df.reset_index(inplace=True)

    # Plot using Plotly Express
    fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns,
                title='Supply Comparison',
                labels={'index': 'Date', 'value': 'Supply', 'variable': 'Series'},
                markers=True)

    # Show the plot
    fig.show()

    combined_df = pd.DataFrame({
    'Pred Supply 7d Rolling Avg': pred_df['supply_7d_rolling avg'],
    'Actual Supply 7d Rolling Avg': test_data_filtered['supply_7d_rolling avg']
    })

    # Reset the index to have a column for time if needed
    # combined_df.reset_index(inplace=True)

    # Plot using Plotly Express
    fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns,
                title='Supply 7-day Rolling Average Comparison',
                labels={'index': 'Date', 'value': 'Supply', 'variable': 'Series'},
                markers=True)

    # Show the plot
    fig.show()

    # Check which indices in actual_values are not in pred_df
    missing_in_pred_df = actual_values.index.difference(pred_df[['Simulated AMPL Price']].index)

    # Check which indices in pred_df are not in actual_values
    missing_in_actual_values = pred_df[['Simulated AMPL Price']].index.difference(actual_values.index)

    print("Indices in actual_values but not in pred_df:", missing_in_pred_df)
    print("Indices in pred_df but not in actual_values:", missing_in_actual_values)

    plot_simulation(pred_df, actual_values)

if __name__ == "__main__":
    target = 'AMPL_Price'
    dataset, features = get_dataset()
    main(test_data=dataset,target=target,top_corr=features,
         train_model=False,
          start_date=None,
           end_date='2020-06-01 00:00:00',
           actions=1.5) # pass action as %; on chart viewed as percent (need to fix that)



