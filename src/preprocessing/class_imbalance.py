import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Literal, Tuple
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTETomek


class ClassBalancer:
    METHODS = [
        'random_undersample', 'random_oversample',
        'smote', 'smote_tomek', 'adasyn',
        'tomek_links', 'edited_nn', 'cluster_centroids'
    ]

    def __init__(
            self,
            method: str = 'smote',
            sampling_strategy: str = 'auto',
            random_state: int = 42,
            k_neighbors: int = 5,
            n_jobs: int = -1
    ):
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

        self._sampler = None
        self._fitted = False

    def _create_sampler(self):
        if self.method == 'random_undersample':
            return RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'random_oversample':
            return RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'smote':
            return SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
                n_jobs=self.n_jobs
            )
        elif self.method == 'smote_tomek':
            return SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif self.method == 'adasyn':
            return ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=self.k_neighbors,
                n_jobs=self.n_jobs
            )
        elif self.method == 'tomek_links':
            return TomekLinks(
                sampling_strategy=self.sampling_strategy,
                n_jobs=self.n_jobs
            )
        elif self.method == 'edited_nn':
            return EditedNearestNeighbours(
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.k_neighbors,
                n_jobs=self.n_jobs
            )
        elif self.method == 'cluster_centroids':
            return ClusterCentroids(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def resample(
            self,
            df: pd.DataFrame,
            target_col: str,
            feature_cols: List[str] = None
    ) -> pd.DataFrame:
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        original_dtypes = X.dtypes.to_dict()

        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        sampler = self._create_sampler()
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        result = pd.DataFrame(X_resampled, columns=X.columns)
        result[target_col] = y_resampled

        return result

    def get_config(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'sampling_strategy': self.sampling_strategy,
            'random_state': self.random_state,
            'k_neighbors': self.k_neighbors,
            'n_jobs': self.n_jobs
        }


def balance_classes(
        df: pd.DataFrame,
        target_col: str,
        method: str = 'smote',
        feature_cols: List[str] = None,
        **kwargs
) -> pd.DataFrame:
    balancer = ClassBalancer(method=method, **kwargs)
    return balancer.resample(df, target_col, feature_cols)


def get_class_distribution(df: pd.DataFrame, target_col: str) -> Dict[Any, int]:
    return dict(Counter(df[target_col]))


def calculate_imbalance_ratio(df: pd.DataFrame, target_col: str) -> float:
    distribution = get_class_distribution(df, target_col)
    counts = list(distribution.values())
    return max(counts) / min(counts) if min(counts) > 0 else float('inf')
