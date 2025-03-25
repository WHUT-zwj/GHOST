from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import pandas as pd

# plt.rcParams['font.sans-serif'] = ['SimHei']  # Choose SimHei font
# plt.rcParams['axes.unicode_minus'] = False  # Solve the negative sign display problem

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):  # Initialize experiment parameters
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):  # Build model
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model  # Code for building the model

    def _get_data(self, flag):  # Get dataset
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader  # Code for getting data

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim  # Code for selecting optimizer

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion  # Code for selecting loss function

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_sentiment) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_sentiment = batch_sentiment.float().to(self.device)  # Add sentiment data
                # Create a zero tensor with the same shape as batch_y's last pred_len time steps
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :, :]).float()
                # Concatenate batch_y's first label_len time steps with the created zero tensor
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_sentiment)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, :, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                # Only print information for the first batch to avoid too much output
                if i == 0:
                    break
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # Get training, validation and test data
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # Create directory for saving checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # Initialize early stopping mechanism
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # Select optimizer and loss function
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # Main training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_sentiment) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)  # Input sequence
                # [batch_size, seq_len, num_stocks, features]
                batch_y = batch_y.float().to(self.device)  # Target sequence
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_sentiment = batch_sentiment.float().to(self.device)  # Add sentiment data

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, :, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))  # Weight saving code

        return self.model

    def calculate_labels(self, preds):
        """
        Calculate price change labels for each time point
        preds: predicted price sequence shape: (517, 189, 1)
        """
        num_days, num_stocks, _ = preds.shape
        labels = np.zeros((num_days, num_stocks))  # Store price change labels

        # Calculate price change over pred_len days
        for i in range(num_days - self.args.pred_len):
            current_prices = preds[i, :, 0]  # Current prices
            future_prices = preds[i + self.args.pred_len, :, 0]  # Prices after pred_len days
            labels[i] = (future_prices - current_prices) / current_prices

        print("\n====== Price Change Calculation Monitor ======")
        print(f"Prediction data shape: {preds.shape}")
        print(f"Label data shape: {labels.shape}")

        # Verify reasonableness of price changes
        valid_labels = labels[~np.isnan(labels) & ~np.isinf(labels)]
        print("\nPrice change statistics:")
        print(f"Maximum increase: {np.max(valid_labels):.4f}")
        print(f"Minimum increase: {np.min(valid_labels):.4f}")
        print(f"Average increase: {np.mean(valid_labels):.4f}")
        print(f"Standard deviation: {np.std(valid_labels):.4f}")

        return labels

    def backtest(self, preds, real_prices, dates, stock_codes, open_prices, k=20,
                 initial_capital=10000000, transaction_cost=0.0015):
        """
        Backtest function - Rebalance every pred_len days (A-share T+1 rule adapted version)
        :param preds: predicted values, shape (n_days, n_stocks, 1)
        :param real_prices: actual closing prices, shape (n_days, n_stocks, 1)
        :param dates: date list, shape (n_days,)
        :param stock_codes: stock code list, shape (n_stocks,)
        :param open_prices: opening price data, shape (n_days, n_stocks)
        :param k: number of stocks to select each time
        :param initial_capital: initial capital
        :param transaction_cost: transaction cost
        :return: annualized return, Sharpe ratio, maximum drawdown
        """
        num_days, num_stocks, _ = real_prices.shape
        capital = initial_capital
        daily_returns = []  # Returns for each trading day
        capital_history = [initial_capital]  # Record capital changes
        trading_dates = []  # Record trading dates
        pred_len = self.args.pred_len

        # Track recorded dates to avoid duplicates
        recorded_dates = set()

        print("\n====== Backtest Started ======")
        print(f"Initial capital: {initial_capital}")
        print(f"Backtest days: {num_days}")
        print(f"Number of stocks selected: {k}")
        print(f"Transaction cost: {transaction_cost * 100}%")
        print(f"Rebalancing period: {pred_len} days")

        # Ensure input data alignment
        assert (open_prices.index == dates).all(), "Opening price data not aligned with dates"

        # Backtest according to pred_len cycles
        for day in range(0, num_days - pred_len - 1, pred_len):
            try:
                # 1. Determine buy/sell dates
                buy_day = day + 1  # T+1 buy
                sell_day = min(day + 1 + pred_len, num_days - 1)  # T+1 sell

                # 2. Stock selection logic
                current_pred = preds[day, :, 0]
                future_pred = preds[sell_day, :, 0]
                price_changes = (future_pred - current_pred) / current_pred
                top_k_indices = np.argsort(price_changes)[-k:]

                # 3. Get actual prices
                # Use T+1 day's buying price as T day's closing price
                buy_prices = real_prices[buy_day - 1, top_k_indices, 0]  # Previous day's closing price (T day closing as T+1 buying price)
                sell_prices = real_prices[sell_day, top_k_indices, 0]  # T+1 selling closing price

                # 4. Calculate actual returns after transaction costs
                adjusted_buy = buy_prices * (1 + transaction_cost)  # Add transaction cost when buying
                adjusted_sell = sell_prices * (1 - transaction_cost)  # Subtract transaction cost when selling
                stock_returns = (adjusted_sell - adjusted_buy) / adjusted_buy
                period_return = np.mean(stock_returns)

                # 5. Calculate daily returns during holding period
                prev_prices = buy_prices.copy()  # Use unadjusted prices during holding period
                for t in range(buy_day, sell_day + 1):
                    # Check if this date has already been recorded to avoid duplicates
                    if dates[t] in recorded_dates:
                        continue

                    # Record this date to prevent future duplication
                    recorded_dates.add(dates[t])

                    current_prices = real_prices[t, top_k_indices, 0]

                    # If it's the last day, consider selling transaction cost
                    if t == sell_day:
                        # Use adjusted selling price (adjusted_sell), don't deduct cost again
                        current_return = np.mean((adjusted_sell - prev_prices) / prev_prices)
                    else:
                        current_return = np.mean((current_prices - prev_prices) / prev_prices)

                    # Update capital and records
                    capital *= (1 + current_return)
                    daily_returns.append(current_return)
                    capital_history.append(capital)
                    trading_dates.append(dates[t])
                    prev_prices = current_prices.copy()

                print(f"\nTrading period: {dates[day]} to {dates[sell_day]}")
                print(f"Selected stocks: {[stock_codes[i] for i in top_k_indices]}")
                print(f"Predicted returns: {price_changes[top_k_indices]}")
                print(f"Actual total return: {period_return:.4f}")
                print(f"Current capital: {capital:.2f}")

            except Exception as e:
                print(f"Error in trading period {dates[day]}: {str(e)}")
                continue

        # Calculate result metrics (same as before)
        daily_returns = np.array(daily_returns)
        cumulative_returns = np.cumprod(1 + daily_returns)

        # 1. Calculate total return
        total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0

        # 2. Calculate annualized metrics
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0
        annual_vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0

        # 3. Calculate Sharpe ratio
        risk_free_rate = 0.02  # Annual risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1  # Convert to daily risk-free rate

        # Calculate excess returns
        excess_returns = daily_returns - daily_rf

        # Directly calculate Sharpe ratio (more accurate method)
        if len(excess_returns) > 0 and np.std(excess_returns, ddof=1) != 0:
            sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 4. Calculate maximum drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        print("\n====== Backtest Results ======")
        print(f"Total return: {total_return:.4f}")
        print(f"Annualized return: {annual_return:.4f}")
        print(f"Annualized volatility: {annual_vol:.4f}")
        print(f"Sharpe ratio: {sharpe_ratio:.4f}")
        print(f"Maximum drawdown: {max_drawdown:.4f}")

        # Save trading results (same as before)
        returns_data = pd.DataFrame({
            'Date': trading_dates,
            'Daily_Return': daily_returns,
            'Portfolio_Value': capital_history[1:],  # Exclude initial value
            'Cumulative_Return': cumulative_returns - 1
        })

        try:
            output_dir = "trading_results"
            os.makedirs(output_dir, exist_ok=True)
            returns_data.to_excel(os.path.join(output_dir, 'trading_results.xlsx'), index=False)

            plt.figure(figsize=(10, 6))
            plt.plot(trading_dates, cumulative_returns - 1)
            plt.title('Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Return')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'))
            plt.close()
        except Exception as e:
            print(f"Error saving results: {e}")

        return annual_return, sharpe_ratio, max_drawdown
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        self.stock_codes = test_data.stock_codes

        # Extract opening price data
        raw_open_prices_df = test_data.raw_open_prices  # Get opening price data from test_data
        # Convert opening price data to wide format
        raw_open_prices_wide = raw_open_prices_df.pivot(
            index='trade_date',  # Row index is date
            columns='ts_code',  # Column index is stock code
            values='open_qfq'  # Value is opening price
        )

        # Sort by date to ensure correct order
        raw_open_prices_wide = raw_open_prices_wide.sort_index()
        # Calculate actual effective prediction length, considering prediction window
        total_length = len(test_data.dates)
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len

        # Calculate effective prediction days
        valid_length = total_length - seq_len - pred_len + 1
        # Save corresponding dates, ensure date alignment
        self.test_dates = test_data.dates[seq_len + pred_len - 1:total_length]

        # Extract opening price data corresponding to test set dates
        aligned_open_prices = raw_open_prices_wide.loc[self.test_dates]

        if test:
            print('Loading model...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_sentiment) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_sentiment = batch_sentiment.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_sentiment)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -pred_len:, :, :]
                batch_y = batch_y[:, -pred_len:, :, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # Only take the last day of pred_len days prediction
                pred = outputs[:, 0:1, :, f_dim:]  # Take the first day, maintain 4D
                true = batch_y[:, 0:1, :, f_dim:]  # Corresponding actual value is also the first day


                preds.append(pred)
                trues.append(true)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Inverse transform
        if test_data.scale:
            preds, trues = test_data.inverse_transform(preds, trues)
        # If subsequent processing requires 3D data, remove seq_len dimension here
        preds = preds.squeeze(1)  # Only use when necessary
        trues = trues.squeeze(1)

        # Ensure data length matches
        if len(self.test_dates) > preds.shape[0] + pred_len:
            print("Adjusting date range to match prediction data...")
            self.test_dates = self.test_dates[:preds.shape[0] + pred_len]

        # Process prediction results
        results_with_labels = self.calculate_labels(preds)  # Use new label calculation function

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        real_close_prices = trues

        # When calling backtest in test function, pass aligned_open_prices
        annual_return, sharpe_ratio, max_drawdown = self.backtest(
            preds,  # Use original predicted prices, not labels
            real_close_prices,
            self.test_dates,
            self.stock_codes,
            open_prices=aligned_open_prices  # Add opening price data
        )
        # Calculate evaluation metrics
        mae, mse, rmse, mape, mspe, rse, corr, accuracy, precision, recall, f1, sharpe = metric(preds, trues,
                                                                                                pred_len=self.args.pred_len)

        print("\n====== Price Movement Prediction Results ======")
        print(f'Accuracy: {accuracy:.6f}')
        print(f'Precision: {precision:.6f}')
        print(f'Recall: {recall:.6f}')
        print(f'F1: {f1:.6f}')

        return mae, mse, rmse, mape, mspe, rse, corr, accuracy, precision, recall, f1, sharpe