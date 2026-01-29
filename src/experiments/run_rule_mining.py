"""
Rule Mining Experiment: Deficiency Labels

Discretizes data with kmeans (3, 5, 10 bins) and mines rules for
deficiency-related labels, excluding ferritin, b12, and folate features.
"""
import logging
from pathlib import Path
from datetime import datetime

import aerial
import pandas as pd

from src.preprocessing.pipeline import PreprocessingPipeline
from src.rule_mining.aerial_miner import AerialMiner
from src.postprocessing.rule import filter_rules
from src.utils.excel_io import save_experiment_results, format_rule_for_excel

aerial.setup_logging(logging.INFO)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "../../data/raw/bloodcounts_labeled.csv"
OUTPUT_DIR = "../../out/deficiency_rules"

# Labels to mine rules for (RHS)
TARGET_LABELS = [
    'label_iron_deficiency',
    'label_b12_deficiency',
    'label_folate_deficiency',
    'label_possible_b12_deficiency',
    'label_possible_iron_deficiency_anemia',
    'label_iron_deficiency_anemia'
]

# Features to exclude for all deficiency labels
EXCLUDED_FEATURES = []

# Discretization configs
DISCRETIZATION_METHOD = 'entropy'  # 'kmeans', 'equal_width', 'equal_frequency', 'entropy'
BIN_SIZES = [3, 5, 10]

# Aerial config
AERIAL_CONFIG = {
    'epochs': 2,
    'max_antecedents': 3,
    'ant_similarity': 0.01,
    'cons_similarity': 0.8,
    'batch_size': 64,
    'layer_dims': [16]
}

# Filter thresholds
MIN_CONFIDENCE = 0.8
MIN_ZHANGS = 0.001


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("DEFICIENCY RULE MINING EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
    print(f"  Shape: {df.shape}")

    # Identify columns
    label_cols = [c for c in df.columns if c.startswith('label_')]
    feature_cols = [c for c in df.columns if c not in label_cols]

    # Find excluded feature columns (case-insensitive match)
    excluded_cols = []
    for feat in EXCLUDED_FEATURES:
        matches = [c for c in feature_cols if feat.lower() in c.lower()]
        excluded_cols.extend(matches)
    print(f"  Excluding features: {excluded_cols}")

    # Verify target labels exist
    available_targets = [l for l in TARGET_LABELS if l in df.columns]
    print(f"  Target labels found: {len(available_targets)}/{len(TARGET_LABELS)}")

    all_results = []

    # Run for each bin size
    for n_bins in BIN_SIZES:
        print(f"\n{'=' * 70}")
        print(f"DISCRETIZATION: {DISCRETIZATION_METHOD} with {n_bins} bins")
        print("=" * 70)

        # Mine rules for each target label
        for target_label in available_targets:
            # Preprocess with this bin size
            labeled_df = df[df[target_label].notna()].copy()
            pipeline = PreprocessingPipeline(name=f"{DISCRETIZATION_METHOD}_{n_bins}bins")
            pipeline.add_discretization(
                method=DISCRETIZATION_METHOD,
                n_bins=n_bins,
                target_col=target_label
            )

            processed_df = pipeline.fit_transform(labeled_df)
            print(f"  Processed shape: {processed_df.shape}")

            print(f"\n  [{target_label}] Mining rules...")

            # Prepare data: exclude features and other labels
            cols_to_use = [c for c in processed_df.columns
                           if c not in excluded_cols or c == target_label]
            cols_to_use = [c for c in cols_to_use
                           if not c.startswith('label_') or c == target_label]

            label_data = processed_df[cols_to_use].copy()
            print(f"    Features: {len(cols_to_use) - 1}")

            # Configure miner for this label
            miner = AerialMiner(
                epochs=AERIAL_CONFIG['epochs'],
                max_antecedents=AERIAL_CONFIG['max_antecedents'],
                ant_similarity=AERIAL_CONFIG['ant_similarity'],
                cons_similarity=AERIAL_CONFIG['cons_similarity'],
                batch_size=AERIAL_CONFIG['batch_size'],
                layer_dims=AERIAL_CONFIG['layer_dims'],
                target_class=[target_label]
            )

            try:
                rules, stats = miner.mine_rules(label_data)
                print(f"    Mined: {len(rules)} rules")

                # Filter rules
                if rules:
                    rules = filter_rules(rules, criterion='confidence', threshold=MIN_CONFIDENCE)
                    print(f"    After confidence >= {MIN_CONFIDENCE}: {len(rules)} rules")

                    # rules = filter_rules(rules, criterion='zhangs_metric', threshold=MIN_ZHANGS)
                    # print(f"    After zhang's >= {MIN_ZHANGS}: {len(rules)} rules")

                all_results.append({
                    'n_bins': n_bins,
                    'label': target_label,
                    'rules': rules,
                    'stats': stats,
                    'num_rules': len(rules),
                    'num_features': len(cols_to_use) - 1
                })

            except Exception as e:
                print(f"    Error: {e}")
                all_results.append({
                    'n_bins': n_bins,
                    'label': target_label,
                    'rules': [],
                    'stats': {'error': str(e)},
                    'num_rules': 0
                })

    # Save results
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print("=" * 70)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Prepare sheets for Excel
    # Sheet 1: All rules (formatted for human readability)
    all_rules_data = []
    for result in all_results:
        for rule in result['rules']:
            rule_row = format_rule_for_excel(rule)
            all_rules_data.append(rule_row)

    # Sheet 2: Summary per label/bins
    summary_data = []
    for result in all_results:
        summary_data.append({
            'n_bins': result['n_bins'],
            'label': result['label'],
            'num_rules': result['num_rules'],
            'num_features': result.get('num_features', 0),
            'avg_confidence': sum(r.get('confidence', 0) for r in result['rules']) / len(result['rules']) if result[
                'rules'] else 0,
            'avg_support': sum(r.get('support', 0) for r in result['rules']) / len(result['rules']) if result[
                'rules'] else 0,
            'avg_zhangs_metric': sum(r.get('zhangs_metric', 0) for r in result['rules']) / len(result['rules']) if
            result[
                'rules'] else 0,
            'avg_interestingness': sum(r.get('interestingness', 0) for r in result['rules']) / len(result['rules']) if
            result[
                'rules'] else 0
        })

    # Sheet 3: Parameters
    params = {
        'data_path': DATA_PATH,
        'discretization': DISCRETIZATION_METHOD,
        'bin_sizes': str(BIN_SIZES),
        'target_labels': str(TARGET_LABELS),
        'excluded_features': str(EXCLUDED_FEATURES),
        'min_confidence': MIN_CONFIDENCE,
        'min_zhangs_metric': MIN_ZHANGS,
        **AERIAL_CONFIG,
        'timestamp': datetime.now().isoformat()
    }

    # Save
    filename = f"{timestamp}_deficiency_rules_experiment"
    save_experiment_results(
        output_path=output_path / filename,
        sheets={
            'All Rules': all_rules_data,
            'Summary': summary_data,
            'Parameters': params
        }
    )

    # Print final summary
    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults by bin size:")
    for n_bins in BIN_SIZES:
        bin_results = [r for r in all_results if r['n_bins'] == n_bins]
        total_rules = sum(r['num_rules'] for r in bin_results)
        print(f"  {n_bins} bins: {total_rules} total rules")
        for r in bin_results:
            print(f"    {r['label']}: {r['num_rules']} rules")

    print(f"\nTotal rules: {len(all_rules_data)}")
    print(f"Output: {output_path / filename}.xlsx")
    print("=" * 70)


if __name__ == '__main__':
    run_experiment()
