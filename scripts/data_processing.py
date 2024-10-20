
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

from scripts.utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data,set_random_seed,data_processing,pull_data
from sql_queries.queries import token_prices, volume


def process_data(api=False,features=None):
    data = pull_data(api=api)

    hourly_dex_vol = data['hourly_dex_vol']
    hourly_prices = data['hourly_prices']
    hourly_ampl_vol = data['hourly_ampl_vol']
    target_data = data['target_data']
    rebase_rates_df = data['rebase_rates_df']
    yfinance_data =data['yfinance_data']
    usdc_vol = data['usdc_vol']
    usdt_vol = data['usdt_vol']
    dai_vol = data['dai_vol']
    stablecoin_volume = data['stablecoin_volume']

    #2022-05-19 is when we don't have nan in dataset
    target_data = target_data[['time','active_addresses','price','supply','target_price','mkap']]
    target_data = target_data[target_data['time']>='2020-05-19 03:00:00']   
    target_data.drop_duplicates(inplace=True)
    target_data['time'] = pd.to_datetime(target_data['time']).dt.tz_localize(None)
    target_data.set_index('time',inplace=True)
    target_data.rename(columns={'price':'AMPL_Price'},inplace=True)

    # hourly_prices.rename(columns={"hour":"DT",'symbol':'SYMBOL'},inplace=True)
    hourly_ampl_vol.rename(columns={"VOLUME":"AMPL Volume"},inplace=True)
    hourly_dex_vol.rename(columns={"VOLUME":"DEX Volume"},inplace=True)
    hourly_dex_vol = to_time(hourly_dex_vol)
    hourly_ampl_vol = to_time(hourly_ampl_vol)
    usdc_vol = to_time(usdc_vol)
    usdt_vol = to_time(usdt_vol)
    dai_vol = to_time(dai_vol)
    stablecoin_volume = to_time(stablecoin_volume)
    # hourly_prices_df = data_processing(hourly_prices, dropna=False)
    # print(F'hourly_prices_df:{hourly_prices_df}')
    # hourly_prices_df.drop(columns=['AMPL_Price'],inplace=True)

    rebase_rates_df['rebaseTime'] = pd.to_datetime(rebase_rates_df['rebaseTime'])
    rebase_rates_df.rename(columns={"rebaseTime":"DT"},inplace=True)
    rebase_rates_df["DT"] = rebase_rates_df["DT"].dt.strftime('%Y-%m-%d %H:00:00') 
    rebase_rates_df.set_index('DT',inplace=True)
    rebase_rates_df.index = pd.to_datetime(rebase_rates_df.index).tz_localize(None)
    rebase_rates_df = rebase_rates_df[rebase_rates_df.index >= '2020-05-19 03:00:00']

    # print(f'hourly_prices_df{hourly_prices_df.index}')
    print(f'hourly_dex_vol: \n{hourly_dex_vol[hourly_dex_vol.index=="2024-10-10 02:00:00"]}')
    print(f'hourly_ampl_vol: \n{hourly_ampl_vol[hourly_ampl_vol.index=="2024-10-10 02:00:00"]}')
    # combined_volume[combined_volume.index=="2024-10-10 02:00:00"]

    # combined_prices = pd.merge(
    #     hourly_prices_df,
    #     hourly_dex_vol,
    #     left_index=True,
    #     right_index=True,
    #     how='inner'

    # )

    combined_volume = pd.merge(
        hourly_dex_vol,
        hourly_ampl_vol,
        left_index=True,
        right_index=True,
        how='outer'
    )

    combined_volume['AMPL Volume'] = combined_volume['AMPL Volume'].fillna(0)

    print(f'combined_volume: {combined_volume[combined_volume.index=="2024-10-10 02:00:00"]}')

    combined_volume = combined_volume.merge(
        usdc_vol,
        left_index=True,
        right_index=True,
        how='outer'
    )

    print(f'combined_volume: {combined_volume[combined_volume.index=="2024-10-10 02:00:00"]}')

    combined_volume = combined_volume.merge(
        usdt_vol,
        left_index=True,
        right_index=True,
        how='outer'
    )

    print(f'combined_volume: {combined_volume[combined_volume.index=="2024-10-10 02:00:00"]}')

    combined_volume = combined_volume.merge(
        dai_vol,
        left_index=True,
        right_index=True,
        how='outer'
    )

    print(f'combined_volume: {combined_volume[combined_volume.index=="2024-10-10 02:00:00"]}')

    combined_volume = combined_volume.merge(
        stablecoin_volume,
        left_index=True,
        right_index=True,
        how='outer'
    )

    print(f'combined_volume: {combined_volume[combined_volume.index=="2024-10-10 02:00:00"]}')

    print(f'rebase_rates_df{rebase_rates_df.index}')
    print(f'target_data{target_data.index}')

    combined_data = pd.merge(
        rebase_rates_df,
        target_data,
        left_index=True,
        right_index=True,
        how='right'
    )

    print(f'combined_data: {combined_data[combined_data.index=="2024-10-10 02:00:00"]}')

    combined_data['rebase_rate'] = combined_data['rebase_rate'].fillna(0)

    print(f'combined_data: {combined_data[combined_data.index=="2024-10-10 02:00:00"]}')
    print(f'combined_volume: {combined_volume[combined_volume.index=="2024-10-10 02:00:00"]}')

    combined_data = combined_data.merge(combined_volume,
                                        left_index=True,
                                        right_index=True,
                                        how='left')
    
    combined_data = combined_data.fillna(0)
    
    combined_data['excess_over_target'] = combined_data['AMPL_Price'] - combined_data['target_price']
    # combined_data.drop(columns=['AMPL_Price'],inplace=True)

    # return combined_data

    combined_data['price_deviation_abs'] = combined_data['AMPL_Price'] - combined_data['target_price']
    combined_data['price_deviation_pct'] = (combined_data['AMPL_Price'] - combined_data['target_price']) / combined_data['target_price'] * 100

    target = 'AMPL_Price'

    core = ['rebase_rate','active_addresses',
                   'supply','target_price','STABLECOIN_VOLUME']

    test_data = combined_data[core+[target]].copy()

    windows=[7,30]

    # windows=[168,720]

    for col in core:
        for window in windows:
            if col != target:
                test_data[f'{col}_{window}d_rolling avg'] = test_data[col].rolling(window=window, min_periods=1).mean().fillna(0)
                test_data[f'{col}_{window}_d_std_dev'] = test_data[col].rolling(window=window, min_periods=1).std().fillna(0)
                test_data[f'{col}_{window}_d_lag'] = test_data[col].shift(window).fillna(0)

    test_data.reset_index(inplace=True)

    # Extract temporal features from the 'ds' column
    test_data['month'] = test_data['time'].dt.month
    test_data['day_of_week'] = test_data['time'].dt.dayofweek
    test_data['day_of_year'] = test_data['time'].dt.dayofyear

    # Calculate Fourier features
    test_data['fourier_sin'] = np.sin(2 * np.pi * test_data['time'].dt.dayofyear / 365.25)
    test_data['fourier_cos'] = np.cos(2 * np.pi * test_data['time'].dt.dayofyear / 365.25)

    # Optionally, you can set the 'ds' column back as the index if needed
    test_data.set_index('time', inplace=True)

    if features == None:

        target_correlations = test_data.corr()[target].sort_values(ascending=False)
        target_correlations = target_correlations.to_frame('correlations_to_ampl_usd')

        top_corr = target_correlations[abs(target_correlations['correlations_to_ampl_usd'])>0.1].index
    else:
        top_corr = pd.Index(features)

    print(f'top_corr: {top_corr}')

    #These are the features as of 10-18; in case I pull api and the correlations change

    # top_corr = pd.Index(['AMPL_Price', 'rebase_rate_30d_rolling avg', 'rebase_rate_30_d_std_dev',
    #    'rebase_rate_7d_rolling avg', 'rebase_rate_7_d_std_dev',
    #    'active_addresses_7d_rolling avg', 'supply_30_d_std_dev',
    #    'active_addresses', 'active_addresses_30d_rolling avg',
    #    'active_addresses_7_d_lag', 'rebase_rate', 'rebase_rate_7_d_lag',
    #    'active_addresses_30_d_std_dev', 'active_addresses_30_d_lag',
    #    'active_addresses_7_d_std_dev', 'rebase_rate_30_d_lag',
    #    'target_price_30d_rolling avg', 'target_price_7d_rolling avg',
    #    'target_price', 'target_price_7_d_lag', 'supply_7_d_std_dev',
    #    'target_price_30_d_lag', 'STABLECOIN_VOLUME_7d_rolling avg',
    #    'STABLECOIN_VOLUME_30d_rolling avg', 'supply', 'supply_7d_rolling avg',
    #    'supply_7_d_lag', 'supply_30d_rolling avg', 'supply_30_d_lag'])

    if 'AMPL_Price' in top_corr:
        top_corr = top_corr.drop('AMPL_Price')

    return test_data, top_corr





    