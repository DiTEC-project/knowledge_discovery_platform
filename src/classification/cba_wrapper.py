"""
CBA (Classification Based on Associations) classifier wrapper.

CBA builds classifiers from association rules with class labels on the right-hand side.
"""
from typing import Dict, List, Any
import pandas as pd
from src.classification.base import RuleBasedClassifier
from src.classification.cba.cba_main import CBA
from src.classification.cba.data_structures.transaction_db import TransactionDB
from src.classification.cba.algorithms.rule_generation import createCARs
from src.classification.cba.algorithms.m1algorithm import M1Algorithm
from src.classification.cba.algorithms.m2algorithm import M2Algorithm


class CBAClassifier(RuleBasedClassifier):
    """
    CBA (Classification Based on Associations) classifier.

    Builds a classifier from association rules where the consequent
    is a class label.

    Input: Association rules from:
           - AerialMiner.mine_rules()
           - MLxtendMiner.mine_rules()
           - NiaARMMiner.mine_rules()

    Compatible with all rule miners that generate rules with class labels on RHS.
    """

    def __init__(
            self,
            support: float = 0.10,
            confidence: float = 0.5,
            maxlen: int = 10,
            algorithm: str = "m1",
            **kwargs
    ):
        """
        Initialize CBA classifier.

        Args:
            support: Minimum support threshold
            confidence: Minimum confidence threshold
            maxlen: Maximum length of rules
            algorithm: CBA algorithm variant ('m1' or 'm2')
        """
        super().__init__(**kwargs)
        self.support = support
        self.confidence = confidence
        self.maxlen = maxlen
        self.algorithm = algorithm
        self.cba_model = None
        self.target_class_name = None

    def _convert_rules_to_cba_format(
            self,
            rules: List[Dict[str, Any]],
            target_class: str
    ) -> List[tuple]:
        """
        Convert standard rules to CBA format.

        CBA expects: [(consequent, antecedent, support, confidence), ...]
        where:
            - consequent: str like "class:=:value"
            - antecedent: list/set like ["feature:=:value", ...]

        Args:
            rules: Standard format rules
            target_class: Name of target class column

        Returns:
            List of tuples in CBA format
        """
        cba_rules = []

        for rule in rules:
            # Parse antecedent
            ant_str = rule['antecedent']
            # Assuming format: "feature__value, feature2__value2, ..."
            # or "feature__value AND feature2__value2"

            # Split by common delimiters
            if ' AND ' in ant_str:
                ant_items = ant_str.split(' AND ')
            elif ', ' in ant_str:
                ant_items = ant_str.split(', ')
            else:
                ant_items = [ant_str]

            # Convert to CBA format: feature:=:value
            ant_formatted = []
            for item in ant_items:
                item = item.strip()
                if '__' in item:
                    feature, value = item.split('__', 1)
                    ant_formatted.append(f"{feature}:=:{value}")
                elif '=' in item:
                    # Already in feature=value format
                    feature, value = item.split('=', 1)
                    ant_formatted.append(f"{feature}:=:{value}")

            # Parse consequent
            cons_str = rule['consequent']
            # Check if consequent is for target class
            if target_class in cons_str:
                # Extract class value
                if '__' in cons_str:
                    _, class_value = cons_str.split('__', 1)
                elif '=' in cons_str:
                    _, class_value = cons_str.split('=', 1)
                else:
                    class_value = cons_str

                cons_formatted = f"{target_class}:=:{class_value}"

                # Add to CBA rules
                cba_rules.append((
                    cons_formatted,
                    set(ant_formatted),
                    rule['support'],
                    rule['confidence']
                ))

        return cba_rules

    def fit(
            self,
            rules: List[Dict[str, Any]],
            X: pd.DataFrame,
            y: pd.Series
    ) -> Any:
        """
        Train CBA classifier from association rules.

        Args:
            rules: List of association rules
                  Each rule: {'antecedent': str, 'consequent': str,
                             'support': float, 'confidence': float, ...}
            X: Training features (discretized)
            y: Training labels

        Returns:
            Trained CBA model
        """
        self.target_class_name = y.name if y.name else "target"

        # Convert rules to CBA format
        cba_rules = self._convert_rules_to_cba_format(rules, self.target_class_name)

        if len(cba_rules) == 0:
            raise ValueError("No rules found for target class. Ensure rules have class labels on RHS.")

        # Prepare data in TransactionDB format
        data_with_target = X.copy()
        data_with_target[self.target_class_name] = y

        # Convert to TransactionDB
        txns = TransactionDB.from_DataFrame(data_with_target)

        # Create and train CBA model
        self.cba_model = CBA(
            support=self.support,
            confidence=self.confidence,
            maxlen=self.maxlen,
            algorithm=self.algorithm
        )

        # Fit using pre-mined rules

        # Create CARs from rules
        cars = createCARs(cba_rules)

        # Build classifier
        algorithm_class = M1Algorithm if self.algorithm == "m1" else M2Algorithm
        self.cba_model.clf = algorithm_class(cars, txns).build()
        self.cba_model.target_class = self.target_class_name

        self.model = self.cba_model
        return self.model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class labels using CBA classifier.

        Args:
            X: Test features (discretized)

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert to TransactionDB
        txns = TransactionDB.from_DataFrame(X)

        # Predict
        predictions = self.cba_model.predict(txns)

        return pd.Series(predictions, index=X.index)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy on test data.

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare test data
        data_with_target = X.copy()
        data_with_target[self.target_class_name] = y
        txns = TransactionDB.from_DataFrame(data_with_target)

        # Test
        accuracy = self.cba_model.rule_model_accuracy(txns)

        return accuracy

    def __repr__(self):
        return (f"CBAClassifier(support={self.support}, confidence={self.confidence}, "
                f"maxlen={self.maxlen}, algorithm={self.algorithm})")
