#w/ Actions

#window was 360, scale factor at 1

import pandas as pd
from datetime import timedelta
import numpy as np

class RebaseSim:
    def __init__(self, df, target, features, model, window=360, scale_factor=1, start_date=None, end_date=None):
        # Set default start and end dates
        if start_date is None:
            start_date = df.index.min()
        if end_date is None:
            end_date = df.index.max()

        print(f'features: {features}')

        # Filter the dataframe to the simulation period
        self.df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        self.start_date = start_date
        self.end_date = end_date
        self.target = target
        self.features = features.to_list() 
        self.model = model
        self.results = pd.DataFrame()
        self.window = window
        self.scale_factor = scale_factor
        self.prediction_df = pd.DataFrame(columns=[self.target] + self.features)
        self.prediction_df.index.name = 'time'  # New DataFrame for predictions

    def run_simulation(self, cycle=1, actions=None):
      # current_time = pd.to_datetime(self.start_date) + timedelta(hours=1)
      current_time = pd.to_datetime(self.start_date)
      end_time = pd.to_datetime(self.end_date)

      while current_time <= end_time:
          predictions = []

          # Check if current time is 02:00:00 to apply the rebase action
          if current_time.time() == pd.Timestamp('02:00:00').time():
              print(f'current_time: {current_time.time()}')
              if actions is not None:
                  self.apply_action(actions, current_time) # Applies to orig df

          hours_until_end = (end_time - current_time).total_seconds() // 3600
          effective_cycle = min(cycle, int(hours_until_end))

          print(f'start_Date: {self.start_date} \nend_date: {self.end_date} \ndf: {self.df}')

          for _ in range(effective_cycle):
              try:
                  # Modify this line to ensure we're using the latest day
                  # print(f'current_time: {current_time}')
                  # print(f'df: {self.df[self.df.index <= current_time]}')
                  latest_day = self.df[self.df.index <= current_time].index[-1]
                  self.prediction_df.loc[current_time] = self.df[[self.target] + self.features].loc[current_time].copy()

                  # print(f'self.prediction_df["supply"]: {self.prediction_df["supply"]}')

                  self.recalculate_temporal_columns(current_time=current_time)

                  # Select the last available data for features
                  X_test = self.prediction_df[self.features].loc[[current_time]]
                  X_test.dropna(inplace=True)

                  if X_test.empty:
                      print(f"No features available for prediction at {latest_day}. Ending simulation.")
                      return  # Exit if there's no data to predict

                  print(f"Current features for prediction: {X_test}")

              except KeyError:
                  print(f"Warning: Features not available for current time: {current_time}. Skipping.")
                  break

              # Calculate historical volatility using only the data up to the current time
              volatilities = self.calculate_historical_volatility(current_time)
              prediction = self.forecast(X_test, volatilities)
              predictions.append(prediction[0])  # Store the prediction

              # Update self.prediction_df and move to the next hour
              self.prediction_df.loc[current_time] = [*prediction, *X_test.values.flatten()]
              current_time += timedelta(hours=1)

              # self.recalculate_temporal_columns(current_time=current_time)

          if predictions:
              future_index = pd.date_range(start=current_time - timedelta(hours=effective_cycle), periods=effective_cycle, freq='H', inclusive='both')
              self.update_prediction_df(future_index, predictions)
              print(f"Cycle completed. Forecast for {future_index[-1]} done. Predictions: {predictions}")
          else:
              print("No predictions made in this cycle. Exiting simulation.")
              break  # Exit if no predictions were made

          # Check for NaNs in results after simulation
          if self.results.isnull().values.any():
              print("Warning: NaN values detected in results DataFrame.")


    def apply_action(self, action, current_time):
      # Apply the rebase action to 'supply'
      if action is not None and 'supply' in self.df.columns and 'rebase_rate' in self.df.columns:
          original_supply = self.df.loc[current_time, "supply"] if not self.df['supply'].empty else 0
          # print(f'action: {action}')
          # print(f'original_supply: {original_supply}')

          new_supply = original_supply * (1 + action / 100)  # Adjust supply by the action percentage
          # print(f'new_supply: {new_supply}')

          # Update the supply for the current time
          self.df.loc[current_time, "supply"] = new_supply
          self.df.loc[current_time, "rebase_rate"] = action
          print(f"Adjusted supply by {action}% from {original_supply} to {new_supply}")

          # Fill forward the new supply value for the next 24 hours
          next_rebase_time = current_time + timedelta(hours=24)
          print(f'next_rebase_time: {next_rebase_time}')
          self.df.loc[current_time:next_rebase_time, "supply"] = new_supply

            # Recalculate temporal columns after applying the action using data up to now
            # self.recalculate_temporal_columns(current_time=current_time)  # Pass the latest index

    def recalculate_temporal_columns(self, current_time):
        # Use data up to the current time to calculate temporal features
        data_up_to_now = self.prediction_df[self.prediction_df.index <= current_time]

        # Check if there's enough data to compute rolling features
        if data_up_to_now.empty:
            return  # No data to calculate

        # 7-day rolling average for supply
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'supply_7d_rolling avg'] = data_up_to_now['supply'].rolling(window=7, min_periods=1).mean().fillna(0)
        # 7-day standard deviation for supply
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'supply_7_d_std_dev'] = data_up_to_now['supply'].rolling(window=7, min_periods=1).std().fillna(0)
        # 7-day lag for supply
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'supply_7_d_lag'] = data_up_to_now['supply'].shift(7).fillna(0)

        # 30-day rolling average for supply
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'supply_30d_rolling avg'] = data_up_to_now['supply'].rolling(window=30, min_periods=1).mean().fillna(0)
        # 30-day standard deviation for supply
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'supply_30_d_std_dev'] = data_up_to_now['supply'].rolling(window=30, min_periods=1).std().fillna(0)
        # 30-day lag for supply
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'supply_30_d_lag'] = data_up_to_now['supply'].shift(30).fillna(0)

        # 7-day rolling average for rebase_rate
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'rebase_rate_7d_rolling avg'] = data_up_to_now['rebase_rate'].rolling(window=7, min_periods=1).mean().fillna(0)
        # 7-day standard deviation for rebase_rate
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'rebase_rate_7_d_std_dev'] = data_up_to_now['rebase_rate'].rolling(window=7, min_periods=1).std().fillna(0)
        # 7-day lag for rebase_rate
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'rebase_rate_7_d_lag'] = data_up_to_now['rebase_rate'].shift(7).fillna(0)

        # 30-day rolling average for rebase_rate
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'rebase_rate_30d_rolling avg'] = data_up_to_now['rebase_rate'].rolling(window=30, min_periods=1).mean().fillna(0)
        # 30-day standard deviation for rebase_rate
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'rebase_rate_30_d_std_dev'] = data_up_to_now['rebase_rate'].rolling(window=30, min_periods=1).std().fillna(0)
        # 30-day lag for rebase_rate
        self.prediction_df.loc[self.prediction_df.index <= current_time, 'rebase_rate_30_d_lag'] = data_up_to_now['rebase_rate'].shift(30).fillna(0)

        # self.prediction_df.fillna(0, inplace=True)  # Fill NaN with zeros

        # print(f'self.prediction_df after temporal update at {current_time}: {self.prediction_df[self.features]}')
        print("Recalculated the temporal columns for supply and rebase_rate.")

    def forecast(self, X, volatilities):
        # Make predictions using the model
        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)  # Ensure predictions are non-negative

        # Check for NaN in predictions
        if np.isnan(predictions).any():
            print("Predictions contain NaN values!")

        # # Apply stochastic noise based on historical volatility
        # if np.isnan(volatilities):
        #     print("Volatility is NaN. Adjusting prediction with zero noise.")
        #     noise = np.zeros_like(predictions)  # Use zero noise if volatility is NaN
        # else:
        #     noise = np.random.normal(0, volatilities * self.scale_factor, predictions.shape)

        # # Set a minimum value for predictions
        # minimum_value = 0.12 * self.df[self.target].mean()  # Adjust this as needed
        # adjusted_predictions = np.maximum(predictions + noise, minimum_value)

        adjusted_predictions = predictions

        # Check for NaN in adjusted predictions
        if np.isnan(adjusted_predictions).any():
            print("Adjusted predictions contain NaN values!")
            print("Original Predictions:", predictions)
            # print("Noise:", noise)

        return adjusted_predictions

    def update_prediction_df(self, future_index, predictions):
      new_data = pd.DataFrame(predictions, index=future_index, columns=[self.target])
      new_data.index.name = 'time'

      # print(f'new_data: {new_data}')
      # print(f'new_data index: {new_data.index}')

      # # Ensure time is the index from the start
      # if 'time' not in self.prediction_df.columns:
      #     self.prediction_df['time'] = self.prediction_df.index  # Add 'time' column if missing
      # print(f'self.prediction_df before concat:{self.prediction_df}')
      # Concatenate new predictions with the existing DataFrame
      self.prediction_df = pd.concat([self.prediction_df, new_data],axis=0)
      # print(f'self.prediction_df after concat:{self.prediction_df}')

      self.prediction_df = self.prediction_df.reset_index()
      # print(f'self.prediction_df after reset:{self.prediction_df}')

      # print(f'self.prediction_df time col: {self.prediction_df["time"]}')

      # Drop duplicates based on both 'time' (index) and target
      self.prediction_df = self.prediction_df.drop_duplicates(subset=[self.target, 'time'], keep='first')
      # print(f'self.prediction_df after dropdupe:{self.prediction_df.index}')
      # Set the index back to 'time'
      self.prediction_df = self.prediction_df.set_index('time')
      # print(f'self.prediction_df:{self.prediction_df.index}')

      print(f"Updated prediction_df with predictions. Latest entry for {future_index[0]}: {self.prediction_df[self.target].iloc[-1]}")

    def get_results(self):
        return self.results

    def get_df(self):
        return self.df

    def get_prediction_df(self):
        return self.prediction_df  # Add this method to access the prediction DataFrame

    def calculate_historical_volatility(self, current_time):
        window = self.window
        # print(f'current_time: {current_time}')
        # print(f'self.prediction_df[self.prediction_df.index < current_time]: {self.prediction_df[self.prediction_df.index <= current_time]}')
        # Filter the DataFrame to include only the data up to the current time
        data_up_to_now = self.prediction_df[self.prediction_df.index <= current_time]
        print(f'data_up_to_now: {data_up_to_now.index}')

        # Calculate percentage change
        daily_returns = data_up_to_now[self.target].pct_change()
        daily_returns = daily_returns.dropna()

        # Ensure we have enough data to compute volatility
        if len(daily_returns) < window:
            print(f"Not enough data to compute volatility at {current_time}. Returning 0.")
            return 0  # Return a fallback value

        # Calculate historical volatility
        volatility = daily_returns.rolling(window=window, min_periods=1).std().iloc[-1]

        if np.isnan(volatility):
            print(f"Volatility is NaN at {current_time}. Returning 0.")
            return 0  # Return a fallback value if volatility is NaN

        return volatility


