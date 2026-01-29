"""
Base interfaces for rule-based classifiers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import pandas as pd


class ItemsetBasedClassifier(ABC):
    """
    Base class for classifiers that work with frequent itemsets.

    Example: CORELS

    Input: Frequent itemsets (from FrequentItemsetMiner or HybridMiner.mine_itemsets())
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.model = None

    @abstractmethod
    def fit(
            self,
            itemsets: List[Dict[str, Any]],
            X: pd.DataFrame,
            y: pd.Series
    ) -> Any:
        """
        Train the classifier using frequent itemsets.

        Args:
            itemsets: List of frequent itemsets from mining
                     Each itemset is a dict with 'items' and 'support'
            X: Training features (discretized)
            y: Training labels

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class labels for samples.

        Args:
            X: Features (discretized)

        Returns:
            Predicted labels
        """
        pass

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy.

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy (0-1 or 0-100 depending on implementation)
        """
        predictions = self.predict(X)
        return (predictions == y).mean()


class RuleBasedClassifier(ABC):
    """
    Base class for classifiers that work with association rules.

    Example: CBA

    Input: Association rules with class labels on RHS
           (from AssociationRuleMiner or HybridMiner.mine_rules())
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.model = None

    @abstractmethod
    def fit(
            self,
            rules: List[Dict[str, Any]],
            X: pd.DataFrame,
            y: pd.Series
    ) -> Any:
        """
        Train the classifier using association rules.

        Args:
            rules: List of association rules from mining
                  Each rule is a dict with 'antecedent', 'consequent',
                  'support', 'confidence', etc.
            X: Training features (discretized)
            y: Training labels

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class labels for samples.

        Args:
            X: Features (discretized)

        Returns:
            Predicted labels
        """
        pass

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy.

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy (0-1 or 0-100 depending on implementation)
        """
        predictions = self.predict(X)
        return (predictions == y).mean()
