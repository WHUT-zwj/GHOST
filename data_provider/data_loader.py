import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='',
                 target='close_qfq', scale=True, timeenc=0, freq='d', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # Information
        if size == None:
            self.seq_len = 60  # Adjust these values as needed
            self.label_len = 30
            self.pred_len = 10
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # Initialize
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        # Path under WSL environment, mounted under Linux, so can't use Windows format
        self.sentiment_path = "/mnt/d/PycharmProjects/MyFIinal/dataset/CHN_NEWS_sentiment.csv"

        # Add file existence check
        if not os.path.exists(self.sentiment_path):
            raise FileNotFoundError(f"Sentiment data file not found: {self.sentiment_path}")
        self.__read_data__()



    def __read_data__(self):

        data_folder = os.path.join(self.root_path, self.data_path)
        if not os.path.isdir(data_folder):
            raise ValueError(f"Error: {data_folder} is not a valid directory")
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]


        all_data = []
        stock_codes = []

        for file in csv_files:
            file_path = os.path.join(data_folder, file)
            # print(f"Reading file: {file_path}")

            # Read CSV file
            df = pd.read_csv(file_path)

            # Ensure date column is properly parsed
            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # Get stock code
            stock_code = df['ts_code'].iloc[0]
            stock_codes.append(stock_code)

            # Sort by date
            df = df.sort_values('trade_date')

            all_data.append(df)

        self.stock_codes = np.array(stock_codes)

        # Merge all data
        df_raw = pd.concat(all_data, ignore_index=True)

        # Sort by date and stock code
        df_raw = df_raw.sort_values(['trade_date', 'ts_code'])

        # Data preprocessing
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('trade_date')
        cols.remove('ts_code')
        df_raw = df_raw[['trade_date', 'ts_code'] + cols + [self.target]]

        # Modify this part of the code
        unique_dates = df_raw['trade_date'].unique()
        unique_dates = np.sort(unique_dates)  # Use np.sort instead of .sort()

        # Calculate the number of dates in each time period
        total_dates = len(unique_dates)
        num_train = int(total_dates * 0.6)
        num_test = int(total_dates * 0.2)
        num_vali = total_dates - num_train - num_test

        # Determine date boundaries
        train_dates = unique_dates[:num_train]
        val_dates = unique_dates[num_train:num_train + num_vali]
        test_dates = unique_dates[num_train + num_vali:]
        # Save complete dataframe for later use in retrieving training data
        df_raw_full = df_raw.copy()

        # Select corresponding date range based on set type
        if self.set_type == 0:  # train
            selected_dates = train_dates
        elif self.set_type == 1:  # validation
            selected_dates = val_dates
        else:  # test
            selected_dates = test_dates

        # Filter data for the selected time period
        df_raw = df_raw[df_raw['trade_date'].isin(selected_dates)]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        feature_columns = [
            # Basic market data (3)
            'open_qfq', 'high_qfq', 'low_qfq',

            # Bias indicators (2)
            'bias1_qfq', 'bias3_qfq',

            # Bollinger Bands (1)
            'boll_mid_qfq',  # Only keep the middle band, remove upper and lower bands

            # BRAR indicators (2)
            'brar_ar_qfq', 'brar_br_qfq',

            # Other individual technical indicators (3)
            'cci_qfq', 'cr_qfq', 'dfma_dif_qfq',

            # DMI indicators (3)
            'dmi_adx_qfq', 'dmi_mdi_qfq', 'dmi_pdi_qfq',  # Keep trend strength and direction features

            # DPO indicator (1)
            'dpo_qfq',  # Keep core momentum indicator

            # EMA indicators (3)
            'ema_qfq_10', 'ema_qfq_30', 'ema_qfq_250',  # Keep short-term, medium-term, and long-term moving averages

            # EMV indicator (1)
            'emv_qfq',  # Keep core volatility indicator

            # EXPMA indicator (1)
            'expma_12_qfq',  # Keep short-term exponential moving average

            # KDJ indicators (2)
            'kdj_d_qfq', 'kdj_k_qfq',  # Keep key indicators

            # MA indicators (3)
            'ma_qfq_10', 'ma_qfq_30', 'ma_qfq_250',  # Keep short-term, medium-term, and long-term moving averages

            # MACD indicators (2)
            'macd_dif_qfq', 'macd_dea_qfq',  # Keep fast line and signal line

            # MASS indicator (1)
            'mass_qfq',  # Keep mass index

            # Other technical indicators (4)
            'mtm_qfq', 'obv_qfq', 'psy_qfq', 'roc_qfq',  # Keep momentum and volume indicators

            # RSI indicator (1)
            'rsi_qfq_12',  # Select medium-term RSI

            # TAQ indicator (1)
            'taq_mid_qfq',  # Keep middle band

            # Other indicators (2)
            'wr_qfq', 'trma_qfq',  # Keep Williams indicator and trend indicator

            # XSII indicators (2)
            'xsii_td1_qfq', 'xsii_td4_qfq' ,'close_qfq' # Select representative indicators
        ]

        df_data = df_raw[feature_columns]


        # Data normalization
        data_normalized = []
        self.scalers = {}

        for stock_code in self.stock_codes:
            # Get current stock data for this set
            stock_data = df_data[df_raw['ts_code'] == stock_code].values
            # print(f"Data shape before normalization: {stock_data.shape}")
            if self.scale:
                scaler = StandardScaler()
                if self.set_type == 0:  # Training set
                    if len(stock_data) > 0:
                        scaler.fit(stock_data)
                    else:
                        print(f"Warning: No training data found for stock {stock_code}")
                        continue
                else:  # Validation or test set
                    # Get training data from complete dataset
                    train_data = df_raw_full[
                        (df_raw_full['trade_date'].isin(train_dates)) &
                        (df_raw_full['ts_code'] == stock_code)
                        ][feature_columns].values

                    if len(train_data) > 0:
                        scaler.fit(train_data)
                    else:
                        print(f"Warning: No training data found for stock {stock_code}")
                        continue

                # Transform data for current dataset
                if len(stock_data) > 0:
                    stock_data = scaler.transform(stock_data)
                else:
                    print(f"Warning: No data found for stock {stock_code} in current set")
                    continue

                self.scalers[stock_code] = scaler

            data_normalized.append(stock_data)

        # Check if there is any data
        if not data_normalized:
            raise ValueError("No data available after processing")

        data = np.stack(data_normalized, axis=1)  # shape: (time_steps, num_stocks, features)

        # Process timestamps
        df_stamp = df_raw[df_raw['ts_code'] == self.stock_codes[0]][['trade_date']]  # Extract dates from the first stock code
        df_stamp['trade_date'] = pd.to_datetime(df_stamp.trade_date)  # Convert timestamp format

        # Save actual dates
        self.dates = df_stamp['trade_date'].values  # New code

        # print(f"   11111111111111datas: {self.dates.shape}")

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.trade_date.dt.month
            df_stamp['day'] = df_stamp.trade_date.dt.day
            df_stamp['weekday'] = df_stamp.trade_date.dt.weekday
            df_stamp['year'] = df_stamp.trade_date.dt.year
            data_stamp = df_stamp.drop(['trade_date'], 1).values
            # Rows represent time steps, columns represent year, month, day, weekday (4 time features)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['trade_date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data[:, :, -1:]  # Only take the last feature (close_qfq)
        self.data_stamp = data_stamp


        # Read sentiment data
        sentiment_df = pd.read_csv(self.sentiment_path)
        sentiment_df['trade_date'] = pd.to_datetime(sentiment_df['trade_date'])

        # Select sentiment features to use
        sentiment_features = [
            'total_news',
            'avg_tone',
            'positive_ratio',
            'news_ma_5',
            'news_std_5',
            'news_zscore_5',
            'news_ma_10',
            'news_std_10',
            'news_zscore_10',
            'news_ma_20',
            'news_std_20',
            'news_zscore_20',
            'news_daily_change',
            'news_weekly_change',
            'tone_ma_3',
            'tone_volatility_3',
            'tone_ma_5',
            'tone_volatility_5',
            'tone_ma_10',
            'tone_volatility_10',
            'tone_change',
            'tone_momentum',
            'pos_ratio_ma_5',
            'pos_ratio_ma_10',
            'pos_ratio_ma_20',
            'pos_ratio_change',
            'pos_ratio_acc',
            'weighted_tone',
            'quality_tone',
            'combined_sentiment'
        ]

        # Normalize sentiment features
        sentiment_scaler = StandardScaler()

        if self.scale:
            # Only process date range for current dataset
            current_sentiment = sentiment_df[
                sentiment_df['trade_date'].isin(selected_dates)
            ][sentiment_features].values

            if self.set_type == 0:  # Training set
                # Fit and transform training data
                sentiment_scaler.fit(current_sentiment)
                sentiment_data = sentiment_scaler.transform(current_sentiment)
            else:  # Validation or test set
                # Fit scaler with training data
                train_sentiment = sentiment_df[
                    sentiment_df['trade_date'].isin(train_dates)
                ][sentiment_features].values
                sentiment_scaler.fit(train_sentiment)
                # Transform current data with trained scaler
                sentiment_data = sentiment_scaler.transform(current_sentiment)
        else:
            # No normalization
            sentiment_data = sentiment_df[
                sentiment_df['trade_date'].isin(selected_dates)
            ][sentiment_features].values


        # Reshape sentiment data to match stock data dimensions
        self.sentiment_data = np.repeat(
            sentiment_data[:, np.newaxis, :],
            len(self.stock_codes),
            axis=1
        )

        # Save sentiment data scaler for later inverse transformation
        self.sentiment_scaler = sentiment_scaler

        if self.set_type == 2:  # Test set
            self.raw_open_prices = df_raw[['trade_date', 'ts_code', 'open_qfq']].copy()
        # Sort by date to ensure order matches the dataset
            self.raw_open_prices = self.raw_open_prices.sort_values(['trade_date', 'ts_code'])


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # Add corresponding sentiment data
        sentiment_x = self.sentiment_data[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, sentiment_x
        # Input sequence (seq_x), target sequence (seq_y), input timestamp (seq_x_mark) and target timestamp (seq_y_mark).

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, preds, trues):
        """
        Shape of preds and trues: [num_samples, 1, num_stocks, num_features] i.e. (517, 1, 189, 1)
        """
        print("Dimensions before inverse transform:")
        print(f"preds shape: {preds.shape}")

        # Save original shape
        original_shape = preds.shape

        # Create result arrays
        preds_inverse = np.zeros_like(preds)
        trues_inverse = np.zeros_like(trues)

        # Inverse transform for each stock separately
        for i, stock_code in enumerate(self.stock_codes):
            scaler = self.scalers[stock_code]

            # Extract data for this stock
            stock_pred = preds[:, :, i, :]  # shape: (517, 1, 1)
            stock_true = trues[:, :, i, :]

            # Reshape to 2D: [num_samples, num_features]
            stock_pred_2d = stock_pred.reshape(-1, 1)  # shape: (517, 1)
            stock_true_2d = stock_true.reshape(-1, 1)

            # Manual inverse transform
            close_price_mean = scaler.mean_[-1]  # Get mean of closing price
            close_price_std = scaler.scale_[-1]  # Get std of closing price

            # Manually perform inverse transformation
            stock_pred_inverse = (stock_pred_2d * close_price_std) + close_price_mean
            stock_true_inverse = (stock_true_2d * close_price_std) + close_price_mean

            # Reshape back to original dimensions
            preds_inverse[:, :, i, :] = stock_pred_inverse.reshape(original_shape[0], 1, 1)
            trues_inverse[:, :, i, :] = stock_true_inverse.reshape(original_shape[0], 1, 1)

        return preds_inverse, trues_inverse

    def inverse_transform_sentiment(self, sentiment_data):
        """Inverse transform sentiment data"""
        if self.scale:
            return self.sentiment_scaler.inverse_transform(sentiment_data)
        return sentiment_data