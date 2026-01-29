import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.preprocessing.pipeline import PreprocessingPipeline
from src.rule_mining.aerial_miner import AerialMiner
from src.rule_mining.niaarm_miner import NiaARMMiner, HHOMiner, GWOMiner, SCAMiner, MFOMiner
from src.rule_mining.mlxtend_miner import MLxtendMiner
from src.postprocessing.rule import filter_rules, filter_itemsets
from src.utils.excel_io import save_rule_mining_results, save_classification_results

from .config import (
    DataConfig, PreprocessingConfig, RuleMiningConfig,
    AerialConfig, NiaARMConfig, MLxtendConfig, FilterConfig
)


def load_data(config: DataConfig) -> pd.DataFrame:
    path = Path(config.path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def preprocess_data(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    label_cols: List[str] = None
) -> pd.DataFrame:
    if label_cols is None:
        label_cols = [c for c in df.columns if c.startswith('label_')]

    pipeline = PreprocessingPipeline()

    if config.imputation:
        pipeline.add_imputation(exclude_cols=label_cols, **config.imputation)

    if config.outlier_handling:
        pipeline.add_outlier_handling(exclude_cols=label_cols, **config.outlier_handling)

    if config.normalization:
        pipeline.add_normalization(exclude_cols=label_cols, **config.normalization)

    if config.discretization:
        pipeline.add_discretization(**config.discretization)

    if config.class_balancing:
        pipeline.add_class_balancing(**config.class_balancing)

    return pipeline.fit_transform(df)


def create_miner(config: RuleMiningConfig, target_classes: List[Dict] = None):
    miner_type = config.miner_type.lower()

    if miner_type == 'aerial':
        cfg = config.miner_config
        return AerialMiner(
            epochs=cfg.epochs,
            max_items=cfg.max_items,
            ant_similarity=cfg.ant_similarity,
            cons_similarity=cfg.cons_similarity,
            batch_size=cfg.batch_size,
            layer_dims=cfg.layer_dims,
            target_class=target_classes
        )

    elif miner_type == 'niaarm':
        cfg = config.miner_config
        algo_map = {
            'hho': HHOMiner,
            'gwo': GWOMiner,
            'sca': SCAMiner,
            'mfo': MFOMiner
        }
        algo_name = cfg.algorithm.lower()
        if algo_name not in algo_map:
            return NiaARMMiner(
                algorithm=cfg.algorithm,
                population=cfg.population,
                max_evals=cfg.max_evals,
                metrics=cfg.metrics
            )
        return algo_map[algo_name](
            population=cfg.population,
            max_evals=cfg.max_evals
        )

    elif miner_type == 'mlxtend':
        cfg = config.miner_config
        return MLxtendMiner(
            algorithm=cfg.algorithm,
            min_support=cfg.min_support,
            min_confidence=cfg.min_confidence,
            max_items=cfg.max_items
        )

    else:
        raise ValueError(f"Unknown miner type: {miner_type}")


def apply_filters(
    data: List[Dict],
    filters: List[FilterConfig],
    mode: str = 'rules'
) -> List[Dict]:
    if not filters:
        return data

    result = data
    filter_func = filter_rules if mode == 'rules' else filter_itemsets

    for f in filters:
        result = filter_func(result, criterion=f.metric, threshold=f.threshold)

    return result


def run_rule_mining(
    data: pd.DataFrame,
    config: RuleMiningConfig,
    target_classes: List[Dict] = None
) -> Tuple[List[Dict], Dict[str, Any]]:
    miner = create_miner(config, target_classes)
    mode = config.mode

    results = []
    stats = {}

    if mode in ['itemsets', 'both'] and hasattr(miner, 'mine_itemsets'):
        itemsets, itemset_stats = miner.mine_itemsets(data)
        itemsets = apply_filters(itemsets, config.filters, mode='itemsets')
        results.extend(itemsets)
        stats['itemsets'] = itemset_stats
        stats['itemsets']['count'] = len(itemsets)

    if mode in ['rules', 'both'] and hasattr(miner, 'mine_rules'):
        rules, rule_stats = miner.mine_rules(data)
        rules = apply_filters(rules, config.filters, mode='rules')
        results.extend(rules)
        stats['rules'] = rule_stats
        stats['rules']['count'] = len(rules)

    return results, stats


def generate_output_filename(
    experiment_name: str,
    miner_type: str,
    mode: str,
    dataset_name: str
) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{experiment_name}_{miner_type}_{mode}_{dataset_name}"