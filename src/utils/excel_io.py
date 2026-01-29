import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union


def save_rule_mining_results(
    rules: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_path: Union[str, Path],
    parameters: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
):
    """
    Save rule mining results to Excel with multiple sheets.

    Sheets:
        - Rules: All mined rules with metrics
        - Summary: Aggregate statistics
        - Parameters: Algorithm parameters used

    Args:
        rules: List of rule dictionaries
        stats: Statistics dictionary from mining
        output_path: Output file path (will add .xlsx if needed)
        parameters: Algorithm parameters used
        metadata: Additional metadata (dataset name, timestamp, etc.)
    """
    output_path = Path(output_path)
    if output_path.suffix != '.xlsx':
        output_path = output_path.with_suffix('.xlsx')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Rules
        if rules:
            rules_df = pd.DataFrame(rules)
            rules_df.to_excel(writer, sheet_name='Rules', index=False)

        # Sheet 2: Summary
        summary_data = {
            'Metric': list(stats.keys()),
            'Value': list(stats.values())
        }
        if metadata:
            summary_data['Metric'].extend(list(metadata.keys()))
            summary_data['Value'].extend(list(metadata.values()))
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 3: Parameters
        if parameters:
            params_df = pd.DataFrame({
                'Parameter': list(parameters.keys()),
                'Value': [str(v) for v in parameters.values()]
            })
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

    print(f"Results saved to: {output_path}")
    return output_path


def save_classification_results(
    fold_results: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    output_path: Union[str, Path],
    parameters: Dict[str, Any] = None,
    model_rules: List[Any] = None,
    metadata: Dict[str, Any] = None
):
    """
    Save classification results to Excel with multiple sheets.

    Sheets:
        - Fold Results: Per-fold accuracy and metrics
        - Summary: Average accuracy, std dev, etc.
        - Parameters: Classifier and experiment parameters
        - Model Rules: Learned rules (if applicable)

    Args:
        fold_results: List of per-fold result dictionaries
        summary_stats: Aggregate statistics (mean, std, etc.)
        output_path: Output file path
        parameters: Experiment parameters
        model_rules: Learned classification rules
        metadata: Additional metadata
    """
    output_path = Path(output_path)
    if output_path.suffix != '.xlsx':
        output_path = output_path.with_suffix('.xlsx')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Fold Results
        if fold_results:
            folds_df = pd.DataFrame(fold_results)
            folds_df.to_excel(writer, sheet_name='Fold Results', index=False)

        # Sheet 2: Summary
        summary_data = {'Metric': [], 'Value': []}
        for k, v in summary_stats.items():
            summary_data['Metric'].append(k)
            summary_data['Value'].append(v)
        if metadata:
            for k, v in metadata.items():
                summary_data['Metric'].append(k)
                summary_data['Value'].append(v)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 3: Parameters
        if parameters:
            params_df = pd.DataFrame({
                'Parameter': list(parameters.keys()),
                'Value': [str(v) for v in parameters.values()]
            })
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

        # Sheet 4: Model Rules
        if model_rules:
            rules_data = _format_model_rules(model_rules)
            if rules_data:
                rules_df = pd.DataFrame(rules_data)
                rules_df.to_excel(writer, sheet_name='Model Rules', index=False)

    print(f"Results saved to: {output_path}")
    return output_path


def save_combined_results(
    all_results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    parameters: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
):
    """
    Save combined results from multiple runs (e.g., per-label training).

    Sheets:
        - All Rules: Combined rules from all runs
        - Per-Label Summary: Stats per label/run
        - Overall Summary: Aggregate statistics
        - Parameters: Experiment parameters

    Args:
        all_results: List of result dicts, each with 'label', 'rules', 'stats' keys
        output_path: Output file path
        parameters: Experiment parameters
        metadata: Additional metadata
    """
    output_path = Path(output_path)
    if output_path.suffix != '.xlsx':
        output_path = output_path.with_suffix('.xlsx')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: All Rules
        all_rules = []
        for result in all_results:
            label = result.get('label', 'unknown')
            for rule in result.get('rules', []):
                rule_copy = rule.copy()
                rule_copy['label'] = label
                all_rules.append(rule_copy)

        if all_rules:
            rules_df = pd.DataFrame(all_rules)
            cols = ['label'] + [c for c in rules_df.columns if c != 'label']
            rules_df = rules_df[cols]
            rules_df.to_excel(writer, sheet_name='All Rules', index=False)

        # Sheet 2: Per-Label Summary
        per_label_data = []
        for result in all_results:
            row = {'label': result.get('label', 'unknown')}
            row['num_rules'] = len(result.get('rules', []))
            stats = result.get('stats', {})
            row.update(stats)
            per_label_data.append(row)

        if per_label_data:
            per_label_df = pd.DataFrame(per_label_data)
            per_label_df.to_excel(writer, sheet_name='Per-Label Summary', index=False)

        # Sheet 3: Overall Summary
        summary_data = {
            'Metric': [
                'total_labels',
                'total_rules',
                'avg_rules_per_label',
                'timestamp'
            ],
            'Value': [
                len(all_results),
                sum(len(r.get('rules', [])) for r in all_results),
                sum(len(r.get('rules', [])) for r in all_results) / len(all_results) if all_results else 0,
                datetime.now().isoformat()
            ]
        }
        if metadata:
            summary_data['Metric'].extend(list(metadata.keys()))
            summary_data['Value'].extend(list(metadata.values()))
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Overall Summary', index=False)

        # Sheet 4: Parameters
        if parameters:
            params_df = pd.DataFrame({
                'Parameter': list(parameters.keys()),
                'Value': [str(v) for v in parameters.values()]
            })
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

    print(f"Results saved to: {output_path}")
    return output_path


def save_experiment_results(
    output_path: Union[str, Path],
    sheets: Dict[str, Union[pd.DataFrame, List[Dict], Dict[str, Any]]]
):
    """
    Generic function to save experiment results with custom sheets.

    Args:
        output_path: Output file path
        sheets: Dictionary mapping sheet names to data.
                Data can be:
                - pd.DataFrame: Written directly
                - List[Dict]: Converted to DataFrame
                - Dict[str, Any]: Converted to key-value DataFrame
    """
    output_path = Path(output_path)
    if output_path.suffix != '.xlsx':
        output_path = output_path.with_suffix('.xlsx')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, data in sheets.items():
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame({
                    'Key': list(data.keys()),
                    'Value': [str(v) for v in data.values()]
                })
            else:
                continue

            # Truncate sheet name if too long (Excel limit is 31 chars)
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    print(f"Results saved to: {output_path}")
    return output_path


def format_rule_for_excel(rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a rule dictionary for Excel output with human-readable antecedent/consequent.

    Converts antecedent/consequent from list/dict formats to parseable strings:
    - Format: "feature1=value1 AND feature2=value2"
    - Parseable by splitting on " AND " then "="
    """
    formatted = rule.copy()

    for key in ['antecedents', 'consequent', 'lhs', 'rhs']:
        if key not in formatted:
            continue
        val = formatted[key]
        formatted[key] = _format_itemset(val)

    return formatted


def _format_itemset(val: Any) -> str:
    """Convert itemset to 'feature=value AND ...' string format."""
    if isinstance(val, str):
        # Handle aerial format: "feature__value" -> "feature=value"
        if '__' in val:
            parts = val.split('__')
            if len(parts) == 2:
                return f"{parts[0]}={parts[1]}"
        return val
    if isinstance(val, dict):
        # Handle {'feature': 'X', 'value': 'Y'} -> "X=Y"
        if 'feature' in val and 'value' in val:
            return f"{val['feature']}={val['value']}"
        return ' AND '.join(f"{k}={v}" for k, v in val.items())
    if isinstance(val, (list, tuple)):
        parts = []
        for item in val:
            if isinstance(item, dict):
                if 'feature' in item and 'value' in item:
                    parts.append(f"{item['feature']}={item['value']}")
                else:
                    parts.extend(f"{k}={v}" for k, v in item.items())
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                parts.append(f"{item[0]}={item[1]}")
            elif isinstance(item, str) and '__' in item:
                # Handle aerial format in lists
                item_parts = item.split('__')
                if len(item_parts) == 2:
                    parts.append(f"{item_parts[0]}={item_parts[1]}")
                else:
                    parts.append(item)
            else:
                parts.append(str(item))
        return ' AND '.join(parts)
    return str(val)


def save_rules_text(
    rules: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "MINED RULES",
    group_by: str = None,
    metadata: Dict[str, Any] = None
) -> Path:
    """
    Save rules in human-readable text format.

    Args:
        rules: List of rule dictionaries with keys like 'antecedent', 'consequent',
               'confidence', 'support', 'zhangs_metric', etc.
        output_path: Output file path (will add .txt if needed)
        title: Title for the output file header
        group_by: Optional key to group rules by (e.g., 'label', 'n_bins')
        metadata: Optional metadata to include in header
    """
    output_path = Path(output_path)
    if output_path.suffix != '.txt':
        output_path = output_path.with_suffix('.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def format_metric(value, decimals=4):
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}"
        return str(value) if value is not None else "N/A"

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if metadata:
            for key, val in metadata.items():
                f.write(f"{key}: {val}\n")
        f.write("=" * 80 + "\n\n")

        if not rules:
            f.write("No rules found.\n")
            print(f"Rules saved to: {output_path}")
            return output_path

        if group_by and rules and group_by in rules[0]:
            groups = {}
            for rule in rules:
                key = rule.get(group_by, 'unknown')
                groups.setdefault(key, []).append(rule)

            rule_num = 1
            for group_key, group_rules in groups.items():
                f.write("-" * 80 + "\n")
                f.write(f"{group_by.upper()}: {group_key}\n")
                f.write(f"Rules in group: {len(group_rules)}\n")
                f.write("-" * 80 + "\n\n")

                for rule in group_rules:
                    rule_num = _write_rule(f, rule, rule_num, format_metric)
                f.write("\n")
        else:
            for i, rule in enumerate(rules, 1):
                _write_rule(f, rule, i, format_metric)

        f.write("=" * 80 + "\n")
        f.write(f"Total rules: {len(rules)}\n")
        f.write("=" * 80 + "\n")

    print(f"Rules saved to: {output_path}")
    return output_path


def _write_rule(f, rule: Dict[str, Any], rule_num: int, format_metric) -> int:
    antecedent = rule.get('antecedent', rule.get('lhs', rule.get('conditions', 'N/A')))
    consequent = rule.get('consequent', rule.get('rhs', rule.get('prediction', 'N/A')))

    f.write(f"Rule #{rule_num}:\n")
    f.write(f"  IF {antecedent}\n")
    f.write(f"  THEN {consequent}\n\n")
    f.write(f"  Metrics:\n")

    metrics = [
        ('confidence', 'Confidence'),
        ('support', 'Support'),
        ('zhangs_metric', "Zhang's Metric"),
        ('interestingness', 'Interestingness'),
        ('lift', 'Lift'),
        ('conviction', 'Conviction'),
        ('leverage', 'Leverage'),
    ]

    for key, label in metrics:
        if key in rule:
            f.write(f"    {label:18s} {format_metric(rule[key])}\n")

    f.write("\n")
    return rule_num + 1


def _format_model_rules(model_rules: List[Any]) -> List[Dict]:
    """Format model rules for Excel output."""
    formatted = []
    for i, rule in enumerate(model_rules):
        if isinstance(rule, dict):
            row = {'rule_index': i}
            row.update(rule)
            formatted.append(row)
        elif isinstance(rule, (list, tuple)):
            # CORELS format: [(cond1, val1), (cond2, val2), ..., prediction]
            if rule and rule[0] and isinstance(rule[0], tuple) and rule[0][0] == 'default':
                formatted.append({
                    'rule_index': i,
                    'type': 'default',
                    'conditions': '',
                    'prediction': rule[0][1]
                })
            else:
                conditions = rule[:-1] if rule else []
                prediction = rule[-1] if rule else None
                cond_str = ' AND '.join(f"{k}={v}" for k, v in conditions) if conditions else ''
                formatted.append({
                    'rule_index': i,
                    'type': 'rule',
                    'conditions': cond_str,
                    'prediction': prediction
                })
    return formatted