import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Literal
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class DataImputer:
    NUMERICAL_METHODS = ['mean', 'median', 'knn', 'mice', 'interpolate']
    CATEGORICAL_METHODS = ['mode', 'constant', 'knn_categorical']

    def __init__(
        self,
        numerical_strategy: Literal['mean', 'median', 'knn', 'mice', 'interpolate'] = 'median',
        categorical_strategy: Literal['mode', 'constant', 'knn_categorical'] = 'mode',
        knn_neighbors: int = 5,
        mice_max_iter: int = 10,
        constant_fill_value: str = 'MISSING',
        random_state: int = 42
    ):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.knn_neighbors = knn_neighbors
        self.mice_max_iter = mice_max_iter
        self.constant_fill_value = constant_fill_value
        self.random_state = random_state

        self._numerical_imputer = None
        self._categorical_imputer = None
        self._numerical_cols = None
        self._categorical_cols = None
        self._fitted = False

    def _identify_column_types(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> tuple:
        exclude_cols = exclude_cols or []
        numerical_cols = []
        categorical_cols = []

        for col in df.columns:
            if col in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)

        return numerical_cols, categorical_cols

    def _create_numerical_imputer(self):
        if self.numerical_strategy == 'mean':
            return SimpleImputer(strategy='mean')
        elif self.numerical_strategy == 'median':
            return SimpleImputer(strategy='median')
        elif self.numerical_strategy == 'knn':
            return KNNImputer(n_neighbors=self.knn_neighbors)
        elif self.numerical_strategy == 'mice':
            return IterativeImputer(
                max_iter=self.mice_max_iter,
                random_state=self.random_state
            )
        elif self.numerical_strategy == 'interpolate':
            return None  # handled separately
        else:
            raise ValueError(f"Unknown numerical strategy: {self.numerical_strategy}")

    def _create_categorical_imputer(self):
        if self.categorical_strategy == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif self.categorical_strategy == 'constant':
            return SimpleImputer(strategy='constant', fill_value=self.constant_fill_value)
        elif self.categorical_strategy == 'knn_categorical':
            return None  # handled separately
        else:
            raise ValueError(f"Unknown categorical strategy: {self.categorical_strategy}")

    def fit(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> 'DataImputer':
        self._numerical_cols, self._categorical_cols = self._identify_column_types(df, exclude_cols)

        if self._numerical_cols and self.numerical_strategy != 'interpolate':
            self._numerical_imputer = self._create_numerical_imputer()
            self._numerical_imputer.fit(df[self._numerical_cols])

        if self._categorical_cols and self.categorical_strategy != 'knn_categorical':
            self._categorical_imputer = self._create_categorical_imputer()
            self._categorical_imputer.fit(df[self._categorical_cols])

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Imputer must be fitted before transform")

        result = df.copy()

        if self._numerical_cols:
            if self.numerical_strategy == 'interpolate':
                result[self._numerical_cols] = result[self._numerical_cols].interpolate(
                    method='linear', limit_direction='both'
                )
                result[self._numerical_cols] = result[self._numerical_cols].fillna(
                    result[self._numerical_cols].median()
                )
            else:
                result[self._numerical_cols] = self._numerical_imputer.transform(
                    result[self._numerical_cols]
                )

        if self._categorical_cols:
            if self.categorical_strategy == 'knn_categorical':
                result = self._knn_categorical_impute(result)
            else:
                result[self._categorical_cols] = self._categorical_imputer.transform(
                    result[self._categorical_cols]
                )

        return result

    def fit_transform(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        return self.fit(df, exclude_cols).transform(df)

    def _knn_categorical_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import LabelEncoder

        result = df.copy()

        for col in self._categorical_cols:
            if result[col].isna().any():
                le = LabelEncoder()
                non_null_mask = result[col].notna()

                if non_null_mask.sum() == 0:
                    result[col] = self.constant_fill_value
                    continue

                result.loc[non_null_mask, col + '_encoded'] = le.fit_transform(
                    result.loc[non_null_mask, col].astype(str)
                )

                if self._numerical_cols:
                    features_for_knn = self._numerical_cols + [col + '_encoded']
                else:
                    features_for_knn = [col + '_encoded']

                temp_df = result[features_for_knn].copy()
                knn = KNNImputer(n_neighbors=self.knn_neighbors)
                imputed = knn.fit_transform(temp_df)

                imputed_col = imputed[:, -1].round().astype(int)
                imputed_col = np.clip(imputed_col, 0, len(le.classes_) - 1)

                result[col] = le.inverse_transform(imputed_col)
                result.drop(columns=[col + '_encoded'], inplace=True)

        return result

    def get_config(self) -> Dict[str, Any]:
        return {
            'numerical_strategy': self.numerical_strategy,
            'categorical_strategy': self.categorical_strategy,
            'knn_neighbors': self.knn_neighbors,
            'mice_max_iter': self.mice_max_iter,
            'constant_fill_value': self.constant_fill_value,
            'random_state': self.random_state
        }


def impute_data(
    df: pd.DataFrame,
    numerical_strategy: str = 'median',
    categorical_strategy: str = 'mode',
    exclude_cols: List[str] = None,
    **kwargs
) -> pd.DataFrame:
    imputer = DataImputer(
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy,
        **kwargs
    )
    return imputer.fit_transform(df, exclude_cols=exclude_cols)