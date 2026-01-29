import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Literal, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


class OutlierHandler:
    METHODS = ['iqr', 'zscore', 'isolation_forest', 'lof', 'winsorize', 'clip']
    ACTIONS = ['remove', 'cap', 'nan', 'flag']

    def __init__(
        self,
        method: Literal['iqr', 'zscore', 'isolation_forest', 'lof', 'winsorize', 'clip'] = 'iqr',
        action: Literal['remove', 'cap', 'nan', 'flag'] = 'cap',
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        contamination: float = 0.05,
        winsorize_limits: Tuple[float, float] = (0.05, 0.05),
        clip_quantiles: Tuple[float, float] = (0.01, 0.99),
        random_state: int = 42
    ):
        self.method = method
        self.action = action
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.winsorize_limits = winsorize_limits
        self.clip_quantiles = clip_quantiles
        self.random_state = random_state

        self._bounds = {}
        self._numerical_cols = None
        self._fitted = False

    def _identify_numerical_cols(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        exclude_cols = exclude_cols or []
        return [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
        ]

    def fit(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> 'OutlierHandler':
        self._numerical_cols = self._identify_numerical_cols(df, exclude_cols)

        if self.method == 'iqr':
            self._fit_iqr(df)
        elif self.method == 'zscore':
            self._fit_zscore(df)
        elif self.method == 'clip':
            self._fit_clip(df)
        elif self.method == 'winsorize':
            self._fit_winsorize(df)

        self._fitted = True
        return self

    def _fit_iqr(self, df: pd.DataFrame):
        for col in self._numerical_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self._bounds[col] = (lower, upper)

    def _fit_zscore(self, df: pd.DataFrame):
        for col in self._numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            lower = mean - self.zscore_threshold * std
            upper = mean + self.zscore_threshold * std
            self._bounds[col] = (lower, upper)

    def _fit_clip(self, df: pd.DataFrame):
        for col in self._numerical_cols:
            lower = df[col].quantile(self.clip_quantiles[0])
            upper = df[col].quantile(self.clip_quantiles[1])
            self._bounds[col] = (lower, upper)

    def _fit_winsorize(self, df: pd.DataFrame):
        for col in self._numerical_cols:
            lower = df[col].quantile(self.winsorize_limits[0])
            upper = df[col].quantile(1 - self.winsorize_limits[1])
            self._bounds[col] = (lower, upper)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted and self.method not in ['isolation_forest', 'lof']:
            raise RuntimeError("OutlierHandler must be fitted before transform")

        result = df.copy()

        if self.method in ['iqr', 'zscore', 'clip', 'winsorize']:
            result = self._transform_bounded(result)
        elif self.method == 'isolation_forest':
            result = self._transform_isolation_forest(result)
        elif self.method == 'lof':
            result = self._transform_lof(result)

        return result

    def _transform_bounded(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        outlier_mask = pd.Series(False, index=df.index)

        for col in self._numerical_cols:
            lower, upper = self._bounds[col]
            col_outliers = (df[col] < lower) | (df[col] > upper)
            outlier_mask = outlier_mask | col_outliers

            if self.action == 'cap':
                result[col] = result[col].clip(lower, upper)
            elif self.action == 'nan':
                result.loc[col_outliers, col] = np.nan
            elif self.action == 'flag':
                result[f'{col}_outlier'] = col_outliers.astype(int)

        if self.action == 'remove':
            result = result[~outlier_mask]

        return result

    def _transform_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if not self._numerical_cols:
            return result

        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        predictions = iso_forest.fit_predict(df[self._numerical_cols].fillna(0))
        outlier_mask = predictions == -1

        if self.action == 'remove':
            result = result[~outlier_mask]
        elif self.action == 'nan':
            result.loc[outlier_mask, self._numerical_cols] = np.nan
        elif self.action == 'flag':
            result['isolation_forest_outlier'] = outlier_mask.astype(int)
        elif self.action == 'cap':
            for col in self._numerical_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                result.loc[outlier_mask, col] = result.loc[outlier_mask, col].clip(lower, upper)

        return result

    def _transform_lof(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if not self._numerical_cols:
            return result

        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            n_jobs=-1
        )
        predictions = lof.fit_predict(df[self._numerical_cols].fillna(0))
        outlier_mask = predictions == -1

        if self.action == 'remove':
            result = result[~outlier_mask]
        elif self.action == 'nan':
            result.loc[outlier_mask, self._numerical_cols] = np.nan
        elif self.action == 'flag':
            result['lof_outlier'] = outlier_mask.astype(int)
        elif self.action == 'cap':
            for col in self._numerical_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                result.loc[outlier_mask, col] = result.loc[outlier_mask, col].clip(lower, upper)

        return result

    def fit_transform(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        return self.fit(df, exclude_cols).transform(df)

    def get_config(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'action': self.action,
            'iqr_multiplier': self.iqr_multiplier,
            'zscore_threshold': self.zscore_threshold,
            'contamination': self.contamination,
            'winsorize_limits': self.winsorize_limits,
            'clip_quantiles': self.clip_quantiles,
            'random_state': self.random_state
        }

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._bounds.copy()


def handle_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    action: str = 'cap',
    exclude_cols: List[str] = None,
    **kwargs
) -> pd.DataFrame:
    handler = OutlierHandler(method=method, action=action, **kwargs)
    return handler.fit_transform(df, exclude_cols=exclude_cols)