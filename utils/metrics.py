import numpy as np
from sklearn.metrics import f1_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def calculate_stock_metrics(pred, true, threshold=0.005):
    """
    Calculate prediction evaluation metrics
    pred: predicted values, shape is (time_steps, num_stocks)
    true: actual values, shape is (time_steps, num_stocks)
    threshold: threshold for determining price movements
    """
    # Calculate price movements
    pred_diff = pred[1:, :] - pred[:-1, :]
    true_diff = true[1:, :] - true[:-1, :]

    num_timesteps = true_diff.shape[0]

    # Store prediction and true labels for each time step
    all_pred_direction = []
    all_true_direction = []

    # Process each time step separately
    for t in range(num_timesteps):
        # Get indices of top k stocks with highest actual returns at current time step
        topk_indices = np.argsort(true_diff[t])[-20:]

        # Get predicted and actual movements for these k stocks at current time step
        pred_diff_topk = pred_diff[t, topk_indices]
        true_diff_topk = true_diff[t, topk_indices]

        # Binary classification: up=1, down=0
        pred_direction = (pred_diff_topk > threshold).astype(int)
        true_direction = (true_diff_topk > threshold).astype(int)

        # Add current time step results to the list
        all_pred_direction.extend(pred_direction)
        all_true_direction.extend(true_direction)

    # Convert to numpy arrays
    pred_direction = np.array(all_pred_direction)
    true_direction = np.array(all_true_direction)

    # Calculate sample weights
    total_samples = len(true_direction)
    weights = {
        'up': np.sum(true_direction == 1) / total_samples,
        'down': np.sum(true_direction == 0) / total_samples,
        'flat': 0  # Keep dictionary structure, but no longer used
    }

    # Initialize results dictionary
    metrics = {
        'accuracy': {'up': 0, 'down': 0, 'flat': 0, 'overall': 0},
        'precision': {'up': 0, 'down': 0, 'flat': 0, 'overall': 0},
        'recall': {'up': 0, 'down': 0, 'flat': 0, 'overall': 0},
        'f1': {'up': 0, 'down': 0, 'flat': 0, 'overall': 0}
    }

    # Calculate confusion matrix
    TP = np.sum((pred_direction == 1) & (true_direction == 1))
    TN = np.sum((pred_direction == 0) & (true_direction == 0))
    FP = np.sum((pred_direction == 1) & (true_direction == 0))
    FN = np.sum((pred_direction == 0) & (true_direction == 1))

    # Calculate metrics for 'up'
    precision_up = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_up = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_up = 2 * (precision_up * recall_up) / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0

    # Calculate metrics for 'down'
    precision_down = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_down = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_down = 2 * (precision_down * recall_down) / (precision_down + recall_down) if (
                                                                                                 precision_down + recall_down) > 0 else 0

    # Fill metrics dictionary
    metrics['precision']['up'] = precision_up
    metrics['precision']['down'] = precision_down
    metrics['recall']['up'] = recall_up
    metrics['recall']['down'] = recall_down
    metrics['f1']['up'] = f1_up
    metrics['f1']['down'] = f1_down
    metrics['accuracy']['up'] = recall_up
    metrics['accuracy']['down'] = recall_down

    # Calculate overall accuracy
    metrics['accuracy']['overall'] = (TP + TN) / (TP + TN + FP + FN)

    # Calculate weighted average for overall metrics
    for metric in ['precision', 'recall', 'f1']:
        metrics[metric]['overall'] = (
                metrics[metric]['up'] * weights['up'] +
                metrics[metric]['down'] * weights['down']
        )

    return metrics


def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio
    :param returns: numpy array of daily (or periodic) returns
    :param risk_free_rate: risk-free rate, default is 0
    :return: Sharpe ratio
    """
    try:
        # Ensure input is a numpy array
        returns = np.array(returns)

        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]

        # Ensure data is one-dimensional
        returns = returns.ravel()

        # Calculate excess returns
        excess_returns = returns - risk_free_rate

        # Calculate Sharpe ratio
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

        return sharpe
    except Exception as e:
        print(f"Error in sharpe_ratio calculation: {e}")
        print(f"returns shape: {returns.shape}")
        print(f"returns statistics: mean={np.mean(returns)}, std={np.std(returns)}")
        return 0  # Return default value


def metric(pred, true, pred_len, threshold=0.005):
    """
    Comprehensively calculate all evaluation metrics

    Parameters:
    pred: predicted values
    true: actual values
    pred_len: prediction length
    threshold: threshold for determining price movements

    Returns:
    Tuple containing all metrics: mae, mse, rmse, mape, mspe, rse, corr,
                     accuracy, precision, recall, f1, sharpe
    """
    # Basic error metrics
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    # Calculate daily returns
    daily_returns = (pred[1:] - pred[:-1]) / pred[:-1]
    sharpe = sharpe_ratio(daily_returns)

    # Calculate price movement related metrics
    stock_metrics = calculate_stock_metrics(pred, true, threshold)

    # Get overall metrics
    accuracy = stock_metrics['accuracy']['overall']
    precision = stock_metrics['precision']['overall']
    recall = stock_metrics['recall']['overall']
    f1 = stock_metrics['f1']['overall']

    return mae, mse, rmse, mape, mspe, rse, corr, accuracy, precision, recall, f1, sharpe