"""
Classification Experiment

Loads data, applies preprocessing, mines rules/itemsets, builds classifiers,
and evaluates with cross-validation. Saves results to Excel.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import aerial
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.experiments.config import (
    DataConfig, PreprocessingConfig, RuleMiningConfig, ClassificationConfig,
    AerialConfig, FilterConfig
)
from src.experiments.base import load_data, preprocess_data, run_rule_mining
from src.classification.corels_wrapper import CorelsClassifier
from src.classification.cba_wrapper import CBAClassifier
from src.utils.excel_io import save_classification_results

aerial.setup_logging(logging.INFO)


def run_cross_validation(
        classifier_type: str,
        rules_or_itemsets: List[Dict],
        X: pd.DataFrame,
        y: pd.Series,
        config: ClassificationConfig,
        dataset_name: str
) -> Dict[str, Any]:
    """
    Run k-fold cross-validation with the specified classifier.

    Args:
        classifier_type: 'corels' or 'cba'
        rules_or_itemsets: Mined rules or itemsets
        X: Feature DataFrame
        y: Target Series
        config: Classification configuration
        dataset_name: Name for CORELS file output

    Returns:
        Dictionary with fold results and summary statistics
    """
    n_folds = config.n_folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)

    fold_results = []
    train_scores = []
    test_scores = []
    exec_times = []
    all_model_rules = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"    Fold {fold_idx}/{n_folds}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        try:
            if classifier_type == 'corels':
                clf = CorelsClassifier(n_jobs=1)
                clf.fit(rules_or_itemsets, X_train, y_train, dataset_name=dataset_name)
                exec_time = clf.execution_time
                model_rules = clf.get_model_rules()
            else:  # cba
                clf = CBAClassifier(
                    support=config.support,
                    confidence=config.confidence,
                    maxlen=config.maxlen,
                    algorithm=config.cba_algorithm
                )
                clf.fit(rules_or_itemsets, X_train, y_train)
                exec_time = 0
                model_rules = None

            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)

            train_scores.append(train_acc)
            test_scores.append(test_acc)
            exec_times.append(exec_time)
            if model_rules:
                all_model_rules.append(model_rules)

            print(f"      Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")

            fold_results.append({
                'fold': fold_idx,
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc),
                'execution_time': exec_time,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })

        except Exception as e:
            print(f"      Error: {e}")
            fold_results.append({'fold': fold_idx, 'error': str(e)})

    # Calculate summary statistics
    summary = {
        'classifier': classifier_type.upper(),
        'cv_method': f'{n_folds}-fold',
        'train_accuracy_mean': float(np.mean(train_scores)) if train_scores else 0,
        'train_accuracy_std': float(np.std(train_scores)) if train_scores else 0,
        'test_accuracy_mean': float(np.mean(test_scores)) if test_scores else 0,
        'test_accuracy_std': float(np.std(test_scores)) if test_scores else 0,
        'execution_time_mean': float(np.mean(exec_times)) if exec_times else 0
    }

    print(f"\n    Summary: Test {summary['test_accuracy_mean']:.2f}% ± {summary['test_accuracy_std']:.2f}%")

    return {
        'fold_results': fold_results,
        'summary': summary,
        'model_rules': all_model_rules
    }


def run_experiment(
        data_config: DataConfig,
        preprocessing_config: PreprocessingConfig,
        mining_config: RuleMiningConfig,
        classification_config: ClassificationConfig,
        target_label: str,
        output_dir: str = "./out/classifiers"
):
    """
    Run classification experiment.

    Args:
        data_config: Data loading configuration
        preprocessing_config: Preprocessing configuration
        mining_config: Rule mining configuration
        classification_config: Classification configuration
        target_label: Target label column for classification
        output_dir: Output directory
    """
    print("=" * 70)
    print("CLASSIFICATION EXPERIMENT")
    print("=" * 70)
    print(f"Dataset: {data_config.name}")
    print(f"Target: {target_label}")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = load_data(data_config)
    print(f"  Shape: {df.shape}")

    label_cols = data_config.get_label_cols(df)

    if target_label not in df.columns:
        print(f"  Error: Target label '{target_label}' not found in data")
        return

    # Preprocess
    print("\n[2] Preprocessing...")
    processed_df = preprocess_data(df, preprocessing_config, label_cols)
    print(f"  Processed shape: {processed_df.shape}")

    # Prepare features and target
    X = processed_df
    y = processed_df[target_label]

    print(f"  Target distribution: {y.value_counts().to_dict()}")

    # Mine rules/itemsets
    print("\n[3] Mining rules/itemsets...")
    target_classes = [{target_label: True}]
    rules_or_itemsets, mining_stats = run_rule_mining(
        processed_df, mining_config, target_classes
    )
    print(f"  Mined: {len(rules_or_itemsets)} items")

    if not rules_or_itemsets:
        print("  Error: No rules/itemsets mined. Cannot proceed with classification.")
        return

    # Filter out itemsets containing target label (for CORELS)
    if mining_config.mode == 'itemsets':
        rules_or_itemsets = [
            item for item in rules_or_itemsets
            if target_label not in item.get('items', {})
        ]
        print(f"  After filtering target label: {len(rules_or_itemsets)} items")

    # Run classification
    print("\n[4] Running classification...")
    classifier_type = classification_config.classifier_type

    cv_results = run_cross_validation(
        classifier_type=classifier_type,
        rules_or_itemsets=rules_or_itemsets,
        X=X,
        y=y,
        config=classification_config,
        dataset_name=data_config.name
    )

    # Save results
    print("\n[5] Saving results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{classifier_type}_{mining_config.miner_type}_{data_config.name}_{target_label.replace('label_', '')}"

    save_classification_results(
        fold_results=cv_results['fold_results'],
        summary_stats=cv_results['summary'],
        output_path=output_path / filename,
        parameters={
            **classification_config.to_dict(),
            **mining_config.to_dict()
        },
        model_rules=cv_results.get('model_rules'),
        metadata={
            'dataset': data_config.name,
            'target_label': target_label,
            'preprocessing': str(preprocessing_config),
            'timestamp': datetime.now().isoformat()
        }
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(
        f"Test accuracy: {cv_results['summary']['test_accuracy_mean']:.2f}% ± {cv_results['summary']['test_accuracy_std']:.2f}%")
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

    # Mining configuration (itemsets for CORELS, rules for CBA)
    MINING = RuleMiningConfig(
        miner_type='aerial',
        miner_config=AerialConfig(
            epochs=4,
            max_items=2,
            batch_size=64
        ),
        mode='itemsets',  # 'itemsets' for CORELS, 'rules' for CBA
        filters=[FilterConfig('support', 0.05)]
    )

    # Classification configuration
    CLASSIFICATION = ClassificationConfig(
        classifier_type='corels',  # 'corels' or 'cba'
        n_folds=5,
        random_state=42
    )

    # Target label
    TARGET_LABEL = "label_anemia"

    # Output directory
    OUTPUT_DIR = "./out/classifiers"

    # Run experiment
    run_experiment(
        data_config=DATA,
        preprocessing_config=PREPROCESSING,
        mining_config=MINING,
        classification_config=CLASSIFICATION,
        target_label=TARGET_LABEL,
        output_dir=OUTPUT_DIR
    )
