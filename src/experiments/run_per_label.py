"""
Per-Label Mining Experiment

Runs rule mining separately for each label, optionally excluding features
that define each label (to avoid trivial rules).
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import json

import aerial
import pandas as pd

from src.experiments.config import (
    DataConfig, PreprocessingConfig, RuleMiningConfig,
    AerialConfig, NiaARMConfig, FilterConfig
)
from src.experiments.base import load_data, preprocess_data, run_rule_mining
from src.utils.excel_io import save_combined_results

aerial.setup_logging(logging.INFO)


def load_feature_exclusions(path: str) -> Dict[str, Set[str]]:
    """
    Load feature exclusions per label from JSON file.

    Expected format:
    [
        {"label_column": "label_anemia", "conditions": [{"feature": "HGB"}, ...]},
        ...
    ]

    Returns:
        Dict mapping label column to set of features to exclude
    """
    with open(path) as f:
        rules = json.load(f)

    exclusions = {}
    for rule in rules:
        label = rule['label_column']
        if label not in exclusions:
            exclusions[label] = set()
        for condition in rule.get('conditions', []):
            exclusions[label].add(condition['feature'])

    return exclusions


def run_experiment(
    data_config: DataConfig,
    preprocessing_config: PreprocessingConfig,
    mining_config: RuleMiningConfig,
    output_dir: str = "./out/per_label_rules",
    feature_exclusions: Dict[str, Set[str]] = None,
    labels_to_process: List[str] = None
):
    """
    Run per-label mining experiment.

    Args:
        data_config: Data loading configuration
        preprocessing_config: Preprocessing configuration
        mining_config: Rule mining configuration
        output_dir: Output directory
        feature_exclusions: Dict of {label: set of features to exclude}
        labels_to_process: Specific labels to process (None = all)
    """
    print("=" * 70)
    print("PER-LABEL MINING EXPERIMENT")
    print("=" * 70)
    print(f"Dataset: {data_config.name}")
    print(f"Miner: {mining_config.miner_type}")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = load_data(data_config)
    print(f"  Shape: {df.shape}")

    label_cols = data_config.get_label_cols(df)
    feature_cols = data_config.get_feature_cols(df)

    if labels_to_process:
        label_cols = [l for l in label_cols if l in labels_to_process]

    print(f"  Labels to process: {len(label_cols)}")

    # Preprocess full dataset
    print("\n[2] Preprocessing...")
    processed_df = preprocess_data(df, preprocessing_config, label_cols)
    print(f"  Processed shape: {processed_df.shape}")

    # Process each label
    print("\n[3] Mining per label...")
    all_results = []
    feature_exclusions = feature_exclusions or {}

    for label_col in label_cols:
        print(f"\n  [{label_col}]")

        # Get features to exclude for this label
        excluded = feature_exclusions.get(label_col, set())
        if excluded:
            print(f"    Excluding {len(excluded)} features")

        # Filter columns
        cols_to_use = [c for c in processed_df.columns
                       if c not in excluded or c == label_col]
        label_data = processed_df[cols_to_use].copy()

        # Set target class
        target_classes = [{label_col: True}, {label_col: 1}, {label_col: "1"}]

        try:
            results, stats = run_rule_mining(label_data, mining_config, target_classes)
            print(f"    Mined: {len(results)} items")

            all_results.append({
                'label': label_col,
                'rules': results,
                'stats': stats,
                'excluded_features': list(excluded),
                'num_features': len(cols_to_use) - 1
            })

        except Exception as e:
            print(f"    Error: {e}")
            all_results.append({
                'label': label_col,
                'rules': [],
                'stats': {'error': str(e)}
            })

    # Save results
    print("\n[4] Saving results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{mining_config.miner_type}_per_label_{data_config.name}"

    save_combined_results(
        all_results=all_results,
        output_path=output_path / filename,
        parameters=mining_config.to_dict(),
        metadata={
            'dataset': data_config.name,
            'preprocessing': str(preprocessing_config),
            'total_labels': len(label_cols),
            'timestamp': datetime.now().isoformat()
        }
    )

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    total_rules = sum(len(r['rules']) for r in all_results)
    print(f"Total labels: {len(all_results)}")
    print(f"Total rules: {total_rules}")
    print("\nPer-label summary:")
    for r in all_results:
        print(f"  {r['label']}: {len(r['rules'])} rules")
    print("=" * 70)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

if __name__ == '__main__':
    # Data configuration
    DATA = DataConfig(
        path="./data/raw/bloodcounts_labeled.csv",
        name="bloodcounts"
    )

    # Preprocessing configuration
    PREPROCESSING = PreprocessingConfig(
        imputation={
            'numerical_strategy': 'median',
            'categorical_strategy': 'mode'
        },
        outlier_handling={
            'method': 'iqr',
            'action': 'cap'
        },
        discretization={
            'method': 'kmeans',
            'n_bins': 10
        }
    )

    # Mining configuration
    MINING = RuleMiningConfig(
        miner_type='aerial',
        miner_config=AerialConfig(
            epochs=2,
            max_items=3,
            min_rule_frequency=0.01,
            min_rule_strength=0.8,
            batch_size=64,
            layer_dims=[8]
        ),
        mode='rules',
        filters=[
            FilterConfig('confidence', 0.8)
        ]
    )

    # Optional: Load feature exclusions from labelling rules
    # FEATURE_EXCLUSIONS = load_feature_exclusions("./data/labelling_rules/comprehensive_label_rules.json")
    FEATURE_EXCLUSIONS = None

    # Optional: Process specific labels only
    # LABELS = ["label_anemia", "label_inflammation"]
    LABELS = None  # Process all

    # Output directory
    OUTPUT_DIR = "./out/per_label_rules"

    # Run experiment
    run_experiment(
        data_config=DATA,
        preprocessing_config=PREPROCESSING,
        mining_config=MINING,
        output_dir=OUTPUT_DIR,
        feature_exclusions=FEATURE_EXCLUSIONS,
        labels_to_process=LABELS
    )