"""State-of-the-art baseline methods for CGM imputation."""

from __future__ import annotations

import numpy as np
from scipy import signal, stats
from scipy.interpolate import UnivariateSpline, Akima1DInterpolator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
import warnings

warnings.filterwarnings("ignore")


def spline_cubic(
    series: np.ndarray, obs_mask: np.ndarray, smoothing: float = 0.1
) -> np.ndarray:
    """Cubic spline interpolation with smoothing.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        smoothing: Smoothing factor (0=interpolating, higher=smoother)

    Returns:
        Imputed series
    """
    idx = np.arange(len(series))
    if not obs_mask.any():
        return np.zeros_like(series)

    obs_idx = idx[obs_mask.astype(bool)]
    obs_vals = series[obs_mask.astype(bool)]

    if len(obs_idx) < 4:  # Fall back to linear for too few points
        return np.interp(idx, obs_idx, obs_vals).reshape(-1, 1).astype(np.float32)

    try:
        # Use UnivariateSpline with smoothing
        spl = UnivariateSpline(obs_idx, obs_vals, s=smoothing * len(obs_idx), k=3)
        imputed = spl(idx)
        return imputed.reshape(-1, 1).astype(np.float32)
    except:
        # Fallback to linear interpolation
        return np.interp(idx, obs_idx, obs_vals).reshape(-1, 1).astype(np.float32)


def akima_interp(series: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Akima spline interpolation - less oscillatory than cubic splines.

    Reference:
        Akima, H. (1970). "A New Method of Interpolation and Smooth Curve
        Fitting Based on Local Procedures." Journal of the ACM, 17(4), 589-602.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)

    Returns:
        Imputed series
    """
    idx = np.arange(len(series))
    if not obs_mask.any():
        return np.zeros_like(series)

    obs_idx = idx[obs_mask.astype(bool)]
    obs_vals = series[obs_mask.astype(bool)].flatten()

    if len(obs_idx) < 2:
        return series.copy()
    elif len(obs_idx) < 5:  # Akima needs at least 5 points
        return np.interp(idx, obs_idx, obs_vals).reshape(-1, 1).astype(np.float32)

    try:
        interp = Akima1DInterpolator(obs_idx, obs_vals)
        imputed = interp(idx)
        # Handle extrapolation with nearest neighbor
        imputed[: obs_idx[0]] = obs_vals[0]
        imputed[obs_idx[-1] :] = obs_vals[-1]
        return imputed.reshape(-1, 1).astype(np.float32)
    except:
        return np.interp(idx, obs_idx, obs_vals).reshape(-1, 1).astype(np.float32)


def savitzky_golay(
    series: np.ndarray,
    obs_mask: np.ndarray,
    window_length: int = 31,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay filter for smooth imputation.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        window_length: Window size (must be odd)
        polyorder: Polynomial order

    Returns:
        Imputed series
    """
    # First do linear interpolation
    idx = np.arange(len(series))
    if not obs_mask.any():
        return np.zeros_like(series)

    obs_idx = idx[obs_mask.astype(bool)]
    obs_vals = series[obs_mask.astype(bool)].flatten()

    if len(obs_idx) < 2:
        return series.copy()

    # Linear interpolation first
    interp_vals = np.interp(idx, obs_idx, obs_vals)

    # Apply Savitzky-Golay smoothing
    window_length = min(window_length, len(series))
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, polyorder + 1)

    try:
        smoothed = signal.savgol_filter(interp_vals, window_length, polyorder)
        return smoothed.reshape(-1, 1).astype(np.float32)
    except:
        return interp_vals.reshape(-1, 1).astype(np.float32)


def ewma_impute(
    series: np.ndarray, obs_mask: np.ndarray, alpha: float = 0.3
) -> np.ndarray:
    """Exponentially Weighted Moving Average imputation.

    Combines forward and backward EWMA for bidirectional smoothing.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        alpha: Smoothing factor (0-1, higher=less smoothing)

    Returns:
        Imputed series
    """
    out = series.copy().astype(np.float32)
    obs_bool = obs_mask.astype(bool).flatten()

    # Forward pass
    ewma_val = None
    for i in range(len(out)):
        if obs_bool[i]:
            if ewma_val is None:
                ewma_val = out[i, 0]
            else:
                ewma_val = alpha * out[i, 0] + (1 - alpha) * ewma_val
        else:
            if ewma_val is not None:
                out[i, 0] = ewma_val

    # Backward pass
    ewma_val = None
    result = out.copy()
    for i in range(len(out) - 1, -1, -1):
        if obs_bool[i]:
            if ewma_val is None:
                ewma_val = series[i, 0]
            else:
                ewma_val = alpha * series[i, 0] + (1 - alpha) * ewma_val
        else:
            if ewma_val is not None:
                # Blend forward and backward passes
                if out[i, 0] != 0:  # Has forward fill
                    result[i, 0] = 0.5 * (out[i, 0] + ewma_val)
                else:
                    result[i, 0] = ewma_val

    return result


def local_mean_impute(
    series: np.ndarray, obs_mask: np.ndarray, window: int = 48
) -> np.ndarray:
    """Local mean imputation using nearby observed values.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        window: Window size for local averaging

    Returns:
        Imputed series
    """
    result = series.copy()
    obs_bool = obs_mask.astype(bool).flatten()

    for i in range(len(series)):
        if not obs_bool[i]:
            # Find nearby observed values
            start = max(0, i - window // 2)
            end = min(len(series), i + window // 2 + 1)

            local_obs = series[start:end][obs_mask[start:end].astype(bool)]

            if len(local_obs) > 0:
                result[i] = np.mean(local_obs)
            else:
                # No local observations, use global mean
                global_obs = series[obs_bool]
                if len(global_obs) > 0:
                    result[i] = np.mean(global_obs)
                else:
                    result[i] = 0

    return result.astype(np.float32)


def seasonal_decompose_impute(
    series: np.ndarray, obs_mask: np.ndarray, period: int = 288
) -> np.ndarray:
    """Seasonal decomposition-based imputation.

    Decomposes into trend + seasonal + residual, imputes each separately.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        period: Seasonal period (288 = daily for 5-min CGM)

    Returns:
        Imputed series
    """
    from scipy.ndimage import uniform_filter1d

    # First fill with linear interpolation
    idx = np.arange(len(series))
    obs_bool = obs_mask.astype(bool).flatten()

    if not obs_bool.any():
        return np.zeros_like(series)

    obs_idx = idx[obs_bool]
    obs_vals = series[obs_bool].flatten()

    if len(obs_idx) < period:
        # Not enough data for decomposition
        return np.interp(idx, obs_idx, obs_vals).reshape(-1, 1).astype(np.float32)

    # Initial interpolation
    filled = np.interp(idx, obs_idx, obs_vals)

    # Extract trend using moving average
    trend = uniform_filter1d(filled, size=period, mode="nearest")

    # Extract seasonal component
    detrended = filled - trend
    seasonal = np.zeros(len(filled))

    # Average seasonal pattern
    for i in range(period):
        seasonal_vals = detrended[i::period]
        if len(seasonal_vals) > 0:
            seasonal[i::period] = np.mean(seasonal_vals)

    # Residual
    residual = filled - trend - seasonal

    # Impute missing values using decomposition
    result = series.copy().astype(np.float32)
    for i in range(len(series)):
        if not obs_bool[i]:
            result[i, 0] = trend[i] + seasonal[i]

    return result


def lstm_ar_impute(
    series: np.ndarray, obs_mask: np.ndarray, lookback: int = 12
) -> np.ndarray:
    """Simple LSTM-like autoregressive imputation.

    Uses a weighted combination of recent values with learned-like weights.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        lookback: Number of previous timesteps to consider

    Returns:
        Imputed series
    """
    result = series.copy().astype(np.float32)
    obs_bool = obs_mask.astype(bool).flatten()

    # Create exponentially decaying weights
    weights = np.exp(-np.arange(lookback) * 0.2)
    weights /= weights.sum()

    for i in range(len(series)):
        if not obs_bool[i]:
            # Look back for observed values
            history = []
            for j in range(max(0, i - lookback), i):
                if obs_bool[j]:
                    history.append(series[j, 0])

            if len(history) > 0:
                # Use weighted average of recent values
                recent = history[-len(weights) :]
                w = weights[-len(recent) :]
                w = w / w.sum()
                result[i, 0] = np.sum(recent * w)
            else:
                # No recent history, try forward looking
                future = []
                for j in range(i + 1, min(len(series), i + lookback + 1)):
                    if obs_bool[j]:
                        future.append(series[j, 0])

                if len(future) > 0:
                    result[i, 0] = future[0]  # Use nearest future value
                else:
                    # Use global mean as last resort
                    global_obs = series[obs_bool]
                    if len(global_obs) > 0:
                        result[i, 0] = np.mean(global_obs)

    return result


def matrix_factorization_impute(
    series: np.ndarray, obs_mask: np.ndarray, rank: int = 5, window: int = 288
) -> np.ndarray:
    """Matrix factorization-based imputation using local windows.

    Treats the time series as overlapping windows forming a matrix,
    then uses low-rank approximation for imputation.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        rank: Rank for low-rank approximation
        window: Window size for creating matrix

    Returns:
        Imputed series
    """
    L = len(series)

    if L < window:
        # Series too short, fall back to linear
        idx = np.arange(L)
        obs_bool = obs_mask.astype(bool).flatten()
        if obs_bool.any():
            return (
                np.interp(idx, idx[obs_bool], series[obs_bool].flatten())
                .reshape(-1, 1)
                .astype(np.float32)
            )
        else:
            return np.zeros_like(series)

    # Create Hankel-like matrix from time series
    n_windows = L - window + 1
    matrix = np.zeros((window, n_windows))
    mask_matrix = np.zeros((window, n_windows), dtype=bool)

    for i in range(n_windows):
        matrix[:, i] = series[i : i + window, 0]
        mask_matrix[:, i] = obs_mask[i : i + window, 0].astype(bool)

    # Fill missing values with column means initially
    for j in range(n_windows):
        col_obs = matrix[mask_matrix[:, j], j]
        if len(col_obs) > 0:
            matrix[~mask_matrix[:, j], j] = np.mean(col_obs)
        else:
            # Use neighboring columns
            if j > 0:
                matrix[:, j] = matrix[:, j - 1]

    # Low-rank approximation using SVD
    try:
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        s[rank:] = 0  # Keep only top 'rank' components
        matrix_approx = U @ np.diag(s) @ Vt

        # Average overlapping predictions
        result = np.zeros(L)
        counts = np.zeros(L)

        for i in range(n_windows):
            result[i : i + window] += matrix_approx[:, i]
            counts[i : i + window] += 1

        result = (result / np.maximum(counts, 1)).reshape(-1, 1)

        # Keep observed values
        result[obs_mask.astype(bool)] = series[obs_mask.astype(bool)]

        return result.astype(np.float32)
    except:
        # SVD failed, fall back to linear
        idx = np.arange(L)
        obs_bool = obs_mask.astype(bool).flatten()
        if obs_bool.any():
            return (
                np.interp(idx, idx[obs_bool], series[obs_bool].flatten())
                .reshape(-1, 1)
                .astype(np.float32)
            )
        else:
            return np.zeros_like(series)


def ensemble_ml_impute(
    series: np.ndarray, obs_mask: np.ndarray, n_estimators: int = 50
) -> np.ndarray:
    """Ensemble machine learning imputation using Random Forest and Gradient Boosting.

    Creates features from local context and uses ensemble to predict missing values.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)
        n_estimators: Number of trees in ensemble

    Returns:
        Imputed series
    """
    L = len(series)
    obs_bool = obs_mask.astype(bool).flatten()

    if not obs_bool.any() or obs_bool.sum() < 10:
        # Not enough data for ML
        return np.zeros_like(series)

    # Create features: position, local stats, etc.
    features = []
    targets = []

    window = 24  # Look at 2 hours of context (24 * 5min)

    for i in range(L):
        if obs_bool[i]:
            feat = []

            # Position features
            feat.append(i / L)  # Relative position
            feat.append(np.sin(2 * np.pi * i / 288))  # Daily cycle
            feat.append(np.cos(2 * np.pi * i / 288))

            # Local context
            start = max(0, i - window)
            end = min(L, i + window + 1)
            local_data = series[start:end, 0]
            local_obs = obs_mask[start:end, 0].astype(bool)

            if local_obs.any():
                local_vals = local_data[local_obs]
                feat.extend(
                    [
                        np.mean(local_vals),
                        np.std(local_vals) if len(local_vals) > 1 else 0,
                        np.min(local_vals),
                        np.max(local_vals),
                    ]
                )
            else:
                feat.extend([0, 0, 0, 0])

            # Distance to nearest observed
            distances_back = np.where(obs_bool[:i])[0]
            dist_back = (i - distances_back[-1]) / L if len(distances_back) > 0 else 1.0
            feat.append(dist_back)

            distances_forward = np.where(obs_bool[i + 1 :])[0]
            dist_forward = (
                (distances_forward[0] + 1) / L if len(distances_forward) > 0 else 1.0
            )
            feat.append(dist_forward)

            features.append(feat)
            targets.append(series[i, 0])

    if len(features) < 10:
        # Not enough training data
        idx = np.arange(L)
        return (
            np.interp(idx, idx[obs_bool], series[obs_bool].flatten())
            .reshape(-1, 1)
            .astype(np.float32)
        )

    X_train = np.array(features)
    y_train = np.array(targets)

    # Train ensemble
    rf = RandomForestRegressor(
        n_estimators=n_estimators // 2, max_depth=10, random_state=42
    )
    gb = GradientBoostingRegressor(
        n_estimators=n_estimators // 2, max_depth=5, random_state=42
    )

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Predict missing values
    result = series.copy().astype(np.float32)

    for i in range(L):
        if not obs_bool[i]:
            feat = []

            # Same feature extraction as training
            feat.append(i / L)
            feat.append(np.sin(2 * np.pi * i / 288))
            feat.append(np.cos(2 * np.pi * i / 288))

            start = max(0, i - window)
            end = min(L, i + window + 1)
            local_data = result[start:end, 0]  # Use partially imputed data
            local_obs = obs_mask[start:end, 0].astype(bool)

            if local_obs.any():
                local_vals = local_data[local_obs]
                feat.extend(
                    [
                        np.mean(local_vals),
                        np.std(local_vals) if len(local_vals) > 1 else 0,
                        np.min(local_vals),
                        np.max(local_vals),
                    ]
                )
            else:
                feat.extend([0, 0, 0, 0])

            distances_back = np.where(obs_bool[:i])[0]
            dist_back = (i - distances_back[-1]) / L if len(distances_back) > 0 else 1.0
            feat.append(dist_back)

            distances_forward = np.where(obs_bool[i + 1 :])[0]
            dist_forward = (
                (distances_forward[0] + 1) / L if len(distances_forward) > 0 else 1.0
            )
            feat.append(dist_forward)

            X_test = np.array(feat).reshape(1, -1)

            # Ensemble prediction
            pred_rf = rf.predict(X_test)[0]
            pred_gb = gb.predict(X_test)[0]
            result[i, 0] = 0.5 * (pred_rf + pred_gb)

    return result


def bayesian_regression_impute(series: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Bayesian regression imputation with uncertainty estimation.

    Uses Bayesian Ridge Regression to impute with uncertainty.

    Args:
        series: Input time series (L, 1)
        obs_mask: Observation mask (L, 1)

    Returns:
        Imputed series
    """
    L = len(series)
    obs_bool = obs_mask.astype(bool).flatten()

    if not obs_bool.any() or obs_bool.sum() < 5:
        return np.zeros_like(series)

    # Create polynomial features
    idx = np.arange(L)
    degree = min(5, obs_bool.sum() // 3)  # Adaptive degree

    obs_idx = idx[obs_bool]
    obs_vals = series[obs_bool, 0]

    # Create polynomial features
    X_train = np.vander(obs_idx / L, degree + 1, increasing=True)

    # Fit Bayesian Ridge
    br = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, n_iter=300
    )
    br.fit(X_train, obs_vals)

    # Predict all points
    X_all = np.vander(idx / L, degree + 1, increasing=True)
    predictions = br.predict(X_all)

    # Keep observed values
    result = predictions.reshape(-1, 1).astype(np.float32)
    result[obs_bool] = series[obs_bool]

    return result
