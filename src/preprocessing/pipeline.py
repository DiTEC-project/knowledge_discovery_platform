import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .imputation import DataImputer
from .outlier_handling import OutlierHandler
from .normalization import DataNormalizer
from .discretization import DataDiscretizer, discretize_per_label
from .class_imbalance import ClassBalancer


class PreprocessingPipeline:
    def __init__(self, name: str = None):
        self.name = name or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._steps = []
        self._configs = {}
        self._fitted = False

    def add_imputation(
        self,
        numerical_strategy: str = 'median',
        categorical_strategy: str = 'mode',
        exclude_cols: List[str] = None,
        **kwargs
    ) -> 'PreprocessingPipeline':
        imputer = DataImputer(
            numerical_strategy=numerical_strategy,
            categorical_strategy=categorical_strategy,
            **kwargs
        )
        self._steps.append(('imputation', imputer, {'exclude_cols': exclude_cols}))
        self._configs['imputation'] = imputer.get_config()
        return self

    def add_outlier_handling(
        self,
        method: str = 'iqr',
        action: str = 'cap',
        exclude_cols: List[str] = None,
        **kwargs
    ) -> 'PreprocessingPipeline':
        handler = OutlierHandler(method=method, action=action, **kwargs)
        self._steps.append(('outlier_handling', handler, {'exclude_cols': exclude_cols}))
        self._configs['outlier_handling'] = handler.get_config()
        return self

    def add_normalization(
        self,
        method: str = 'zscore',
        exclude_cols: List[str] = None,
        **kwargs
    ) -> 'PreprocessingPipeline':
        normalizer = DataNormalizer(method=method, **kwargs)
        self._steps.append(('normalization', normalizer, {'exclude_cols': exclude_cols}))
        self._configs['normalization'] = normalizer.get_config()
        return self

    def add_discretization(
        self,
        method: str = 'entropy',
        n_bins: int = 5,
        target_col: str = None,
        per_label: bool = False,
        target_labels: List[str] = None,
        exclude_features_per_label: Dict[str, List[str]] = None,
        **kwargs
    ) -> 'PreprocessingPipeline':
        """
        Add discretization step.

        Args:
            method: Discretization method
            n_bins: Number of bins
            target_col: Target column for supervised methods (single-label mode)
            per_label: If True, discretize separately for each label
            target_labels: Labels to discretize for (per_label mode)
            exclude_features_per_label: Features to exclude per label
        """
        if per_label:
            # Per-label discretization - store config only, actual discretization in fit_transform
            self._steps.append(('discretization_per_label', None, {
                'method': method,
                'n_bins': n_bins,
                'target_labels': target_labels,
                'exclude_features_per_label': exclude_features_per_label or {}
            }))
            self._configs['discretization'] = {
                'method': method,
                'n_bins': n_bins,
                'per_label': True,
                'target_labels': target_labels
            }
        else:
            discretizer = DataDiscretizer(
                method=method,
                n_bins=n_bins,
                target_col=target_col,
                **kwargs
            )
            self._steps.append(('discretization', discretizer, {}))
            self._configs['discretization'] = discretizer.get_config()
        return self

    def add_class_balancing(
        self,
        method: str = 'smote',
        target_col: str = None,
        feature_cols: List[str] = None,
        **kwargs
    ) -> 'PreprocessingPipeline':
        balancer = ClassBalancer(method=method, **kwargs)
        self._steps.append(('class_balancing', balancer, {
            'target_col': target_col,
            'feature_cols': feature_cols
        }))
        self._configs['class_balancing'] = balancer.get_config()
        return self

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fit and transform the data through all pipeline steps.

        Returns:
            - pd.DataFrame if using standard discretization
            - Dict[str, pd.DataFrame] if using per_label discretization
        """
        result = df.copy()

        for step_name, transformer, params in self._steps:
            if step_name == 'imputation':
                result = transformer.fit_transform(result, **params)
            elif step_name == 'outlier_handling':
                result = transformer.fit_transform(result, **params)
            elif step_name == 'normalization':
                result = transformer.fit_transform(result, **params)
            elif step_name == 'discretization':
                result = transformer.discretize(result)
            elif step_name == 'discretization_per_label':
                # Per-label discretization returns Dict[label, DataFrame]
                target_labels = params.get('target_labels')
                if target_labels is None:
                    target_labels = [c for c in result.columns if c.startswith('label_')]
                result = discretize_per_label(
                    df=result,
                    label_cols=target_labels,
                    method=params['method'],
                    n_bins=params['n_bins'],
                    exclude_features_per_label=params.get('exclude_features_per_label')
                )
            elif step_name == 'class_balancing':
                result = transformer.resample(result, **params)

        self._fitted = True
        return result

    def get_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'steps': [step[0] for step in self._steps],
            'configs': self._configs
        }

    def save_config(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.get_config(), f, indent=2)

    def get_output_filename(self, base_name: str = 'data') -> str:
        parts = [base_name]

        if 'imputation' in self._configs:
            imp_cfg = self._configs['imputation']
            parts.append(f"imp_{imp_cfg['numerical_strategy']}")

        if 'outlier_handling' in self._configs:
            out_cfg = self._configs['outlier_handling']
            parts.append(f"out_{out_cfg['method']}_{out_cfg['action']}")

        if 'normalization' in self._configs:
            norm_cfg = self._configs['normalization']
            parts.append(f"norm_{norm_cfg['method']}")

        if 'discretization' in self._configs:
            disc_cfg = self._configs['discretization']
            parts.append(f"disc_{disc_cfg['method']}_{disc_cfg['n_bins']}bins")

        if 'class_balancing' in self._configs:
            bal_cfg = self._configs['class_balancing']
            parts.append(f"bal_{bal_cfg['method']}")

        return '_'.join(parts)


def run_preprocessing(
    df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    imputation: Dict[str, Any] = None,
    outlier_handling: Dict[str, Any] = None,
    normalization: Dict[str, Any] = None,
    discretization: Dict[str, Any] = None,
    class_balancing: Dict[str, Any] = None,
    label_cols: List[str] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run preprocessing pipeline and save results.

    Returns:
        - pd.DataFrame if using standard discretization
        - Dict[str, pd.DataFrame] if using per_label discretization
    """
    label_cols = label_cols or [c for c in df.columns if c.startswith('label_')]

    pipeline = PreprocessingPipeline()

    if imputation:
        pipeline.add_imputation(exclude_cols=label_cols, **imputation)

    if outlier_handling:
        pipeline.add_outlier_handling(exclude_cols=label_cols, **outlier_handling)

    if normalization:
        pipeline.add_normalization(exclude_cols=label_cols, **normalization)

    if discretization:
        pipeline.add_discretization(**discretization)

    if class_balancing:
        pipeline.add_class_balancing(**class_balancing)

    result = pipeline.fit_transform(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = pipeline.get_output_filename(dataset_name)

    # Handle per-label discretization output
    if isinstance(result, dict):
        for label, label_df in result.items():
            label_name = label.replace('label_', '')
            label_df.to_csv(output_path / f"{filename}_{label_name}.csv", index=False)
    else:
        result.to_csv(output_path / f"{filename}.csv", index=False)

    pipeline.save_config(output_path / f"{filename}_config.json")

    return result