"""Aerial+ rule mining using PyAerial library."""
import time
from typing import Dict, List, Tuple, Any
import pandas as pd
from aerial import model, rule_extraction

from src.rule_mining.base import HybridMiner


class AerialMiner(HybridMiner):
    """Aerial+ autoencoder-based rule miner."""

    def __init__(
            self,
            epochs: int = 2,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            max_length: int = 3,
            max_antecedents: int = 2,
            ant_similarity: float = 0.5,
            cons_similarity: float = 0.8,
            layer_dims: list = None,
            target_class=None,
            features_of_interest: list = None,
            quality_metrics: list = None,
            **kwargs
    ):
        super().__init__(min_support=0.0, min_confidence=0.0, **kwargs)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_antecedents = max_antecedents
        self.ant_similarity = ant_similarity
        self.cons_similarity = cons_similarity
        self.layer_dims = layer_dims
        self.target_class = target_class
        self.features_of_interest = features_of_interest
        self.quality_metrics = quality_metrics or ['support', 'confidence', 'zhangs_metric', 'interestingness']
        self.trained_model = None

    def _train_model(self, data: pd.DataFrame) -> Any:
        if self.trained_model is None:
            self.trained_model = model.train(
                data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                layer_dims=self.layer_dims
            )
        return self.trained_model

    def mine_itemsets(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        trained_ae = self._train_model(data)

        result = rule_extraction.generate_frequent_itemsets(
            trained_ae,
            features_of_interest=self.features_of_interest,
            similarity=self.ant_similarity,
            max_length=self.max_length
        )

        itemsets_with_support = result.get('itemsets', [])

        itemsets = []
        for itemset_dict in itemsets_with_support:
            items = {}
            for item_pair in itemset_dict['itemset']:
                items[item_pair['feature']] = item_pair['value']

            itemsets.append({
                'items': items,
                'support': float(itemset_dict['support'])
            })

        execution_time = time.time() - start_time
        avg_support = sum(i['support'] for i in itemsets) / len(itemsets) if itemsets else 0.0

        stats = {
            'num_itemsets': len(itemsets),
            'execution_time': execution_time,
            'average_support': float(avg_support),
            'algorithm': 'Aerial+',
            'mode': 'itemsets'
        }

        return itemsets, stats

    def mine_rules(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Mine association rules using Aerial+.

        Args:
            data: DataFrame with discretized features and labels

        Returns:
            Tuple of (rules, stats)
        """
        trained_ae = self._train_model(data)

        result = rule_extraction.generate_rules(
            trained_ae,
            features_of_interest=self.features_of_interest,
            ant_similarity=self.ant_similarity,
            cons_similarity=self.cons_similarity,
            max_antecedents=self.max_antecedents,
            target_classes=self.target_class,
            quality_metrics=self.quality_metrics
        )

        # Handle case where no rules are found
        if result is None:
            return [], {"num_rules": 0, "message": "No rules found with current parameters"}

        rules = result.get("rules", [])
        stats = result.get("statistics", {"num_rules": len(rules)})

        return rules, stats

    def __repr__(self):
        return (f"AerialMiner(epochs={self.epochs}, max_length={self.max_length}, "
                f"max_antecedents={self.max_antecedents}, ant_similarity={self.ant_similarity}, "
                f"cons_similarity={self.cons_similarity})")
