import pandas as pd
from typing import Dict, Any, List
from aerial import discretization as aerial_disc


class DataDiscretizer:
    UNSUPERVISED_METHODS = [
        'equal_width', 'equal_frequency', 'kmeans',
        'quantile', 'zscore', 'custom_bins'
    ]
    SUPERVISED_METHODS = ['entropy', 'decision_tree', 'chimerge']
    ALL_METHODS = UNSUPERVISED_METHODS + SUPERVISED_METHODS

    def __init__(
            self,
            method: str = 'entropy',
            n_bins: int = 5,
            target_col: str = None,
            custom_bins: Dict[str, List[float]] = None,
            random_state: int = 42
    ):
        self.method = method
        self.n_bins = n_bins
        self.target_col = target_col
        self.custom_bins = custom_bins
        self.random_state = random_state

        if method in self.SUPERVISED_METHODS and target_col is None:
            raise ValueError(f"Supervised method '{method}' requires target_col")

    def _convert_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                df[col] = df[col].replace({0: 'False', 1: 'True', 0.0: 'False', 1.0: 'True', True: 'True', False: 'False'})
        return df

    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._convert_binary_columns(df)
        return self._discretize_impl(df)

    def _discretize_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method == 'equal_width':
            return aerial_disc.equal_width_discretization(df, n_bins=self.n_bins)

        elif self.method == 'equal_frequency':
            return aerial_disc.equal_frequency_discretization(df, n_bins=self.n_bins)

        elif self.method == 'kmeans':
            return aerial_disc.kmeans_discretization(
                df, n_bins=self.n_bins, random_state=self.random_state
            )

        elif self.method == 'quantile':
            return aerial_disc.quantile_discretization(df, n_bins=self.n_bins)

        elif self.method == 'zscore':
            return aerial_disc.zscore_discretization(df)

        elif self.method == 'custom_bins':
            if self.custom_bins is None:
                raise ValueError("custom_bins must be provided for custom_bins method")
            return aerial_disc.custom_bins_discretization(df, bins_dict=self.custom_bins)

        elif self.method == 'entropy':
            return aerial_disc.entropy_based_discretization(
                df, target_col=self.target_col, n_bins=self.n_bins
            )

        elif self.method == 'decision_tree':
            return aerial_disc.decision_tree_discretization(
                df, target_col=self.target_col, max_depth=self.n_bins
            )

        elif self.method == 'chimerge':
            return aerial_disc.chimerge_discretization(
                df, target_col=self.target_col, max_bins=self.n_bins
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_config(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'n_bins': self.n_bins,
            'target_col': self.target_col,
            'random_state': self.random_state
        }


def discretize_data(
        df: pd.DataFrame,
        method: str = 'entropy',
        n_bins: int = 5,
        target_col: str = None,
        **kwargs
) -> pd.DataFrame:
    discretizer = DataDiscretizer(
        method=method,
        n_bins=n_bins,
        target_col=target_col,
        **kwargs
    )
    return discretizer.discretize(df)


def discretize_per_label(
    df: pd.DataFrame,
    label_cols: List[str],
    method: str = 'entropy',
    n_bins: int = 5,
    exclude_features_per_label: Dict[str, List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Discretize data separately for each label column.

    Useful for supervised discretization methods where each label
    may require different bin boundaries.

    Args:
        df: Input DataFrame
        label_cols: List of label column names
        method: Discretization method (must be supervised: entropy, decision_tree, chimerge)
        n_bins: Number of bins
        exclude_features_per_label: Dict mapping label to features to exclude

    Returns:
        Dict mapping label column to discretized DataFrame
    """
    results = {}
    exclude_features_per_label = exclude_features_per_label or {}

    feature_cols = [c for c in df.columns if c not in label_cols]

    for label_col in label_cols:
        excluded = exclude_features_per_label.get(label_col, [])
        features_to_use = [f for f in feature_cols if f not in excluded]

        cols_to_keep = features_to_use + [label_col]
        label_df = df[cols_to_keep].copy()

        discretizer = DataDiscretizer(
            method=method,
            n_bins=n_bins,
            target_col=label_col
        )
        results[label_col] = discretizer.discretize(label_df)

    return results
