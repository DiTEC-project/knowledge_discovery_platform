from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class DataConfig:
    path: str
    name: str
    label_prefix: str = "label_"

    def get_label_cols(self, df) -> List[str]:
        return [c for c in df.columns if c.startswith(self.label_prefix)]

    def get_feature_cols(self, df) -> List[str]:
        label_cols = self.get_label_cols(df)
        return [c for c in df.columns if c not in label_cols]


@dataclass
class PreprocessingConfig:
    imputation: Optional[Dict[str, Any]] = None
    outlier_handling: Optional[Dict[str, Any]] = None
    normalization: Optional[Dict[str, Any]] = None
    discretization: Optional[Dict[str, Any]] = None
    class_balancing: Optional[Dict[str, Any]] = None

    # Discretization options:
    #   method: 'equal_width', 'equal_frequency', 'kmeans', 'quantile', 'zscore',
    #           'custom_bins', 'entropy', 'decision_tree', 'chimerge'
    #   n_bins: number of bins
    #   target_col: required for supervised methods (entropy, decision_tree, chimerge)
    #   per_label: if True, discretize separately for each label (supervised methods only)
    #   exclude_features_per_label: Dict[label, List[features]] to exclude per label

    @classmethod
    def default(cls) -> 'PreprocessingConfig':
        return cls(
            imputation={'numerical_strategy': 'median', 'categorical_strategy': 'mode'},
            outlier_handling={'method': 'iqr', 'action': 'cap'},
            normalization=None,
            discretization={'method': 'kmeans', 'n_bins': 10},
            class_balancing=None
        )

    @classmethod
    def minimal(cls) -> 'PreprocessingConfig':
        return cls(
            imputation={'numerical_strategy': 'median', 'categorical_strategy': 'mode'},
            discretization={'method': 'kmeans', 'n_bins': 5}
        )

    @classmethod
    def supervised_per_label(cls, target_labels: List[str] = None) -> 'PreprocessingConfig':
        return cls(
            imputation={'numerical_strategy': 'median', 'categorical_strategy': 'mode'},
            outlier_handling={'method': 'iqr', 'action': 'cap'},
            discretization={
                'method': 'entropy',
                'n_bins': 5,
                'per_label': True,
                'target_labels': target_labels
            }
        )


@dataclass
class AerialConfig:
    epochs: int = 4
    max_items: int = 2
    ant_similarity: float = 0.01
    cons_similarity: float = 0.8
    batch_size: int = 64
    layer_dims: List[int] = field(default_factory=lambda: [16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epochs': self.epochs,
            'max_items': self.max_items,
            'ant_similarity': self.ant_similarity,
            'cons_similarity': self.cons_similarity,
            'batch_size': self.batch_size,
            'layer_dims': self.layer_dims
        }


@dataclass
class NiaARMConfig:
    algorithm: str = 'HHO'
    population: int = 50
    max_evals: int = 10000
    metrics: List[str] = field(default_factory=lambda: ['support', 'confidence'])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'population': self.population,
            'max_evals': self.max_evals,
            'metrics': self.metrics
        }


@dataclass
class MLxtendConfig:
    algorithm: str = 'fpgrowth'
    min_support: float = 0.1
    min_confidence: float = 0.5
    max_items: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'max_items': self.max_items
        }


@dataclass
class FilterConfig:
    metric: str
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {'metric': self.metric, 'threshold': self.threshold}


@dataclass
class RuleMiningConfig:
    miner_type: str  # 'aerial', 'niaarm', 'mlxtend'
    miner_config: Any  # AerialConfig, NiaARMConfig, or MLxtendConfig
    mode: str = 'rules'  # 'rules', 'itemsets', 'both'
    filters: List[FilterConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'miner_type': self.miner_type,
            'miner_config': self.miner_config.to_dict(),
            'mode': self.mode,
            'filters': [f.to_dict() for f in self.filters]
        }


@dataclass
class ClassificationConfig:
    classifier_type: str  # 'corels', 'cba'
    n_folds: int = 5
    random_state: int = 42
    # CORELS params
    regularization: float = 0.015
    max_cardinality: int = 2
    # CBA params
    support: float = 0.1
    confidence: float = 0.5
    maxlen: int = 10
    cba_algorithm: str = 'm1'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'classifier_type': self.classifier_type,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'regularization': self.regularization,
            'max_cardinality': self.max_cardinality,
            'support': self.support,
            'confidence': self.confidence,
            'maxlen': self.maxlen,
            'cba_algorithm': self.cba_algorithm
        }


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    preprocessing: PreprocessingConfig
    output_dir: str = "./out"

    def get_output_path(self, suffix: str = "") -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
