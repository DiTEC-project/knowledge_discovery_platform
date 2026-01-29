"""
Base interfaces for rule mining algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Union
import pandas as pd


class FrequentItemsetMiner(ABC):
    """
    Base class for frequent itemset mining algorithms.

    These algorithms discover frequent co-occurring feature combinations
    without necessarily forming rules (no antecedent -> consequent structure).

    Compatible with: CORELS, CBA
    """

    def __init__(self, min_support: float = 0.01, **kwargs):
        self.min_support = min_support
        self.config = kwargs

    @abstractmethod
    def mine_itemsets(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Mine frequent itemsets from data.

        Args:
            data: DataFrame with discretized features

        Returns:
            Tuple of (itemsets, stats) where:
                itemsets: List of dicts with keys 'items' (frozenset/list) and 'support' (float)
                stats: Dict with mining statistics (execution_time, num_itemsets, etc.)
        """
        pass


class AssociationRuleMiner(ABC):
    """
    Base class for association rule mining algorithms.

    These algorithms discover rules in the form: antecedent -> consequent
    with quality metrics (support, confidence, etc.).

    Compatible with: CBA (if consequent is a class label)
    """

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5, **kwargs):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.config = kwargs

    @abstractmethod
    def mine_rules(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Mine association rules from data.

        Args:
            data: DataFrame with discretized features and labels

        Returns:
            Tuple of (rules, stats) where:
                rules: List of dicts with keys:
                    - 'antecedent': str or list (left-hand side)
                    - 'consequent': str or list (right-hand side)
                    - 'support': float
                    - 'confidence': float
                    - 'lift' (optional): float
                    - Other quality metrics
                stats: Dict with mining statistics
        """
        pass


class HybridMiner(FrequentItemsetMiner, AssociationRuleMiner):
    """
    Base class for algorithms that can produce both frequent itemsets and association rules.

    Examples: Aerial+, MLxtend FP-Growth
    Compatible with: CORELS (itemsets), CBA (rules)

    Important: max_items parameter applies to both modes:
    - For itemsets: Maximum total items in an itemset
    - For rules: Maximum items in the antecedent (LHS)
    """

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_items: int = None,
        **kwargs
    ):
        """
        Initialize hybrid miner with both support and confidence thresholds.

        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            max_items: Maximum number of items (itemset size OR antecedent length)
                      - For itemsets: max total items in itemset
                      - For rules: max items in antecedent (LHS)
            **kwargs: Additional configuration
        """
        # Call AssociationRuleMiner.__init__ which has both parameters
        AssociationRuleMiner.__init__(self, min_support, min_confidence, **kwargs)
        self.max_items = max_items

    @abstractmethod
    def mine_itemsets(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Mine frequent itemsets."""
        pass

    @abstractmethod
    def mine_rules(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Mine association rules."""
        pass
