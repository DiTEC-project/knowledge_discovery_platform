"""
CORELS classifier wrapper.

CORELS builds optimal rule list classifiers from frequent itemsets.
"""
from typing import Dict, List, Any
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from src.classification.base import ItemsetBasedClassifier
from src.classification.corels.corels import (
    create_corels_input_files,
    create_corels_freq_items_input,
    run_corels,
    test_corels_model
)


class CorelsClassifier(ItemsetBasedClassifier):
    """
    CORELS (Certifiably Optimal RulE ListS) classifier.

    Builds an optimal, interpretable rule list from frequent itemsets.

    Input: Frequent itemsets from:
           - AerialMiner.mine_itemsets()
           - MLxtendMiner.mine_itemsets()

    NOT compatible with:
           - NiaARMMiner (generates numerical rules, not itemsets)
    """

    def __init__(
            self,
            n_jobs: int = -1,
            verbose: bool = True,
            **kwargs
    ):
        """
        Initialize CORELS classifier.

        Args:
            n_jobs: Number of parallel jobs for preprocessing (-1 uses all cores)
            verbose: Whether to show progress bars
        """
        super().__init__(**kwargs)
        self.execution_time = None
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
            self,
            itemsets: List[Dict[str, Any]],
            X: pd.DataFrame,
            y: pd.Series,
            dataset_name: str = "blood_count_data"
    ) -> Any:
        """
        Train CORELS classifier from frequent itemsets.

        Args:
            itemsets: List of frequent itemsets
                     Each itemset: {'items': {feature: value, ...}, 'support': float}
            X: Training features (discretized)
            y: Training labels (binary)
            dataset_name: Dataset identifier for file management

        Returns:
            Trained CORELS model (rule list)
        """
        # Prepare CORELS input format (parallelized with pre-cached arrays)
        # Pre-convert DataFrame columns to numpy arrays for 2-3x speedup
        X_arrays = {
            col: X[col].astype(str).values if X[col].dtype == bool else X[col].values
            for col in X.columns
        }

        def process_itemset(itemset_dict):
            items = itemset_dict['items']
            return create_corels_freq_items_input(items, X_arrays)

        if self.verbose:
            print(f"Processing {len(itemsets)} itemsets...")
            itemsets_iter = tqdm(itemsets, desc="Converting itemsets to CORELS format", unit="itemset")
        else:
            itemsets_iter = itemsets

        corels_rules = Parallel(n_jobs=self.n_jobs)(
            delayed(process_itemset)(itemset_dict) for itemset_dict in itemsets_iter
        )

        # Prepare labels dictionary
        # CORELS expects: {class_label: [binary occurrences]}
        target_name = y.name if y.name else "target"
        labels_dict = {}

        # Get unique class values
        unique_classes = y.unique()
        for class_val in unique_classes:
            labels_dict[f"{target_name}_{class_val}"] = (y == class_val).astype(int).tolist()

        # Create CORELS input files
        create_corels_input_files(corels_rules, labels_dict, dataset_name)

        # Run CORELS optimization
        model, exec_time = run_corels(dataset_name)
        print("CORELS model: ", model)

        if model is None:
            raise RuntimeError("CORELS failed to produce a model. Check input data and CORELS binary.")

        self.model = model
        self.execution_time = exec_time
        return self.model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class labels using CORELS rule list (parallelized).

        Args:
            X: Test features (discretized)

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        default_rule = next(
            (rule[-1][1] for rule in self.model if rule[0][0] == 'default'),
            None
        )

        def predict_row(row_data):
            idx, feature_row = row_data
            predicted_label = None

            # Check rules in order
            for condition in self.model:
                if condition[0][0] == 'default':
                    continue

                conditions = condition[:-1]
                expected_label = condition[-1]

                # Check if all conditions are satisfied
                if all(feature_row.get(key) == value for key, value in conditions):
                    predicted_label = expected_label
                    break

            # Use default if no rule matches
            if predicted_label is None and default_rule is not None:
                predicted_label = default_rule

            return idx, predicted_label

        # Parallelize prediction over rows
        if self.verbose and len(X) > 1000:  # Only show progress for large datasets
            print(f"Predicting {len(X)} samples...")
            rows_iter = tqdm(X.iterrows(), desc="Predicting", unit="sample", total=len(X))
        else:
            rows_iter = X.iterrows()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_row)(row_data) for row_data in rows_iter
        )

        # Reconstruct predictions in correct order
        predictions = [pred for _, pred in sorted(results, key=lambda x: x[0])]

        return pd.Series(predictions, index=X.index)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy on test data (parallelized).

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy (0-100)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        accuracy = test_corels_model(self.model, X, y.to_frame(), n_jobs=self.n_jobs)
        return accuracy

    def get_model_rules(self) -> List[Dict[str, Any]]:
        """
        Get the CORELS rule list in structured format.

        Returns:
            List of rule dictionaries
        """
        if self.model is None:
            return []

        rules = []
        for rule in self.model:
            if rule[0][0] == 'default':
                rules.append({
                    'type': 'default',
                    'prediction': rule[0][1]
                })
            else:
                conditions = rule[:-1]
                prediction = rule[-1]
                rules.append({
                    'type': 'rule',
                    'conditions': [{'feature': k, 'value': v} for k, v in conditions],
                    'prediction': prediction
                })
        return rules

    def print_model(self):
        """Print the CORELS rule list in human-readable format."""
        if self.model is None:
            print("No model trained.")
            return

        print("CORELS Rule List:")
        print("-" * 60)
        for i, rule in enumerate(self.model):
            if rule[0][0] == 'default':
                print(f"  Default: Predict {rule[0][1]}")
            else:
                conditions = rule[:-1]
                prediction = rule[-1]
                cond_str = " AND ".join([f"{k}={v}" for k, v in conditions])
                print(f"  Rule {i + 1}: IF {cond_str} THEN {prediction}")
        print("-" * 60)
