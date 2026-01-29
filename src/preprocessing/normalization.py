import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Literal
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer
)


class DataNormalizer:
    METHODS = ['zscore', 'minmax', 'robust', 'maxabs', 'quantile', 'yeo_johnson', 'box_cox']

    def __init__(
        self,
        method: Literal['zscore', 'minmax', 'robust', 'maxabs', 'quantile', 'yeo_johnson', 'box_cox'] = 'zscore',
        feature_range: tuple = (0, 1),
        n_quantiles: int = 1000,
        output_distribution: str = 'uniform',
        random_state: int = 42
    ):
        self.method = method
        self.feature_range = feature_range
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.random_state = random_state

        self._scaler = None
        self._numerical_cols = None
        self._fitted = False

    def _identify_numerical_cols(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        exclude_cols = exclude_cols or []
        return [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
        ]

    def _create_scaler(self):
        if self.method == 'zscore':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == 'robust':
            return RobustScaler()
        elif self.method == 'maxabs':
            return MaxAbsScaler()
        elif self.method == 'quantile':
            return QuantileTransformer(
                n_quantiles=self.n_quantiles,
                output_distribution=self.output_distribution,
                random_state=self.random_state
            )
        elif self.method == 'yeo_johnson':
            return PowerTransformer(method='yeo-johnson')
        elif self.method == 'box_cox':
            return PowerTransformer(method='box-cox')
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> 'DataNormalizer':
        self._numerical_cols = self._identify_numerical_cols(df, exclude_cols)

        if self._numerical_cols:
            self._scaler = self._create_scaler()

            data_to_fit = df[self._numerical_cols]
            if self.method == 'box_cox':
                min_val = data_to_fit.min().min()
                if min_val <= 0:
                    data_to_fit = data_to_fit - min_val + 1e-6
                    self._box_cox_shift = min_val - 1e-6
                else:
                    self._box_cox_shift = 0

            self._scaler.fit(data_to_fit)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        result = df.copy()

        if self._numerical_cols:
            data_to_transform = result[self._numerical_cols]

            if self.method == 'box_cox' and hasattr(self, '_box_cox_shift'):
                data_to_transform = data_to_transform - self._box_cox_shift

            result[self._numerical_cols] = self._scaler.transform(data_to_transform)

        return result

    def fit_transform(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        return self.fit(df, exclude_cols).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        result = df.copy()

        if self._numerical_cols:
            result[self._numerical_cols] = self._scaler.inverse_transform(
                result[self._numerical_cols]
            )

            if self.method == 'box_cox' and hasattr(self, '_box_cox_shift'):
                result[self._numerical_cols] = result[self._numerical_cols] + self._box_cox_shift

        return result

    def get_config(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'feature_range': self.feature_range,
            'n_quantiles': self.n_quantiles,
            'output_distribution': self.output_distribution,
            'random_state': self.random_state
        }


def normalize_data(
    df: pd.DataFrame,
    method: str = 'zscore',
    exclude_cols: List[str] = None,
    **kwargs
) -> pd.DataFrame:
    normalizer = DataNormalizer(method=method, **kwargs)
    return normalizer.fit_transform(df, exclude_cols=exclude_cols)