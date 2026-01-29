"""
MLxtend-based rule mining using various algorithms.

Supports multiple algorithms: FP-Growth, Apriori, Eclat (via FPMax)
Supports both frequent itemset mining and association rule mining.
"""
from typing import Dict, List, Tuple, Any
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori, fpmax, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.rule_mining.base import HybridMiner


class MLxtendMiner(HybridMiner):
    """
    MLxtend rule miner with multiple algorithm support.

    Supports algorithms:
    - 'fpgrowth': FP-Growth (default, fast)
    - 'apriori': Apriori (classic algorithm)
    - 'fpmax': FPMax (finds maximal itemsets)

    Can generate:
    - Frequent itemsets (for CORELS)
    - Association rules (for CBA)
    """

    def __init__(
        self,
        algorithm: str = 'fpgrowth',
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_items: int = None,
        metric: str = 'confidence',
        **kwargs
    ):
        """
        Initialize MLxtend miner.

        Args:
            algorithm: Mining algorithm ('fpgrowth', 'apriori', 'fpmax')
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            max_items: Maximum number of items
                      - For itemsets: max total items in itemset
                      - For rules: max items in antecedent (LHS)
            metric: Metric for rule generation ('confidence', 'lift', etc.)
        """
        super().__init__(min_support, min_confidence, max_items, **kwargs)
        self.algorithm = algorithm.lower()
        self.metric = metric

        # Validate algorithm
        valid_algorithms = ['fpgrowth', 'apriori', 'fpmax']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}, got '{self.algorithm}'")

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert categorical data to one-hot encoded format for mlxtend.

        Args:
            data: DataFrame with categorical values

        Returns:
            One-hot encoded DataFrame
        """
        # Convert DataFrame to transaction format: feature=value
        transactions = []
        for _, row in data.iterrows():
            transaction = []
            for col in data.columns:
                value = row[col]
                # Skip NaN values
                if pd.notna(value):
                    transaction.append(f"{col}__{value}")
            transactions.append(transaction)

        # One-hot encode
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        return df_encoded

    def mine_itemsets(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Mine frequent itemsets using FP-Growth.

        Args:
            data: DataFrame with discretized features

        Returns:
            Tuple of (itemsets, stats)
        """
        import time

        start_time = time.time()

        # Prepare data
        df_encoded = self._prepare_data(data)

        # Mine frequent itemsets using selected algorithm
        if self.algorithm == 'fpgrowth':
            frequent_itemsets_df = fpgrowth(
                df_encoded,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_items
            )
        elif self.algorithm == 'apriori':
            frequent_itemsets_df = apriori(
                df_encoded,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_items
            )
        elif self.algorithm == 'fpmax':
            frequent_itemsets_df = fpmax(
                df_encoded,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_items
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Convert to standard format
        itemsets = []
        for _, row in frequent_itemsets_df.iterrows():
            # Parse itemsets
            items = {}
            for item_str in row['itemsets']:
                if '__' in item_str:
                    feature, value = item_str.split('__', 1)
                    items[feature] = value

            itemsets.append({
                'items': items,
                'support': float(row['support'])
            })

        execution_time = time.time() - start_time

        stats = {
            'num_itemsets': len(itemsets),
            'execution_time': execution_time,
            'average_support': frequent_itemsets_df['support'].mean() if len(itemsets) > 0 else 0.0,
            'algorithm': f'MLxtend_{self.algorithm}',
            'mode': 'itemsets'
        }

        return itemsets, stats

    def mine_rules(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Mine association rules using FP-Growth.

        Args:
            data: DataFrame with discretized features and labels

        Returns:
            Tuple of (rules, stats)
        """
        import time

        start_time = time.time()

        # Prepare data
        df_encoded = self._prepare_data(data)

        # Mine frequent itemsets first using selected algorithm
        if self.algorithm == 'fpgrowth':
            frequent_itemsets_df = fpgrowth(
                df_encoded,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_items
            )
        elif self.algorithm == 'apriori':
            frequent_itemsets_df = apriori(
                df_encoded,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_items
            )
        elif self.algorithm == 'fpmax':
            frequent_itemsets_df = fpmax(
                df_encoded,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_items
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if len(frequent_itemsets_df) == 0:
            return [], {
                'num_rules': 0,
                'execution_time': time.time() - start_time,
                'algorithm': f'MLxtend_{self.algorithm}',
                'mode': 'rules'
            }

        # Generate association rules
        rules_df = association_rules(
            frequent_itemsets_df,
            metric=self.metric,
            min_threshold=self.min_confidence
        )

        # Filter to single-consequent rules only (matching Aerial format)
        rules_df = rules_df[rules_df['consequents'].apply(len) == 1]

        # Calculate consequent support for each rule
        itemset_supports = dict(zip(
            frequent_itemsets_df['itemsets'].apply(frozenset),
            frequent_itemsets_df['support']
        ))

        # Convert to standard format (matching Aerial output structure)
        rules = []
        for _, row in rules_df.iterrows():
            # Parse feature__value items into {'feature': x, 'value': y} dicts
            antecedents = []
            for item in row['antecedents']:
                if '__' in item:
                    feature, value = item.split('__', 1)
                    antecedents.append({'feature': feature, 'value': value})

            # Single consequent (already filtered)
            cons_item = list(row['consequents'])[0]
            if '__' in cons_item:
                feature, value = cons_item.split('__', 1)
                consequent = {'feature': feature, 'value': value}
            else:
                consequent = {'feature': cons_item, 'value': None}

            support = float(row['support'])
            confidence = float(row['confidence'])

            # Get consequent support for metric calculations
            cons_support = itemset_supports.get(row['consequents'], 0.0)

            # Calculate Zhang's metric
            if cons_support > 0 and cons_support < 1:
                if confidence >= cons_support:
                    zhangs_metric = (confidence - cons_support) / (1 - cons_support)
                else:
                    zhangs_metric = (confidence - cons_support) / cons_support
            else:
                zhangs_metric = 0.0

            # Calculate interestingness (support * confidence)
            interestingness = support * confidence

            rules.append({
                'antecedents': antecedents,
                'consequent': consequent,
                'support': support,
                'confidence': confidence,
                'zhangs_metric': round(zhangs_metric, 4),
                'interestingness': round(interestingness, 4),
                'lift': float(row['lift']) if 'lift' in row else None,
                'leverage': float(row['leverage']) if 'leverage' in row else None,
                'conviction': float(row['conviction']) if 'conviction' in row else None
            })

        execution_time = time.time() - start_time

        stats = {
            'num_rules': len(rules),
            'execution_time': execution_time,
            'average_support': sum(r['support'] for r in rules) / len(rules) if rules else 0.0,
            'average_confidence': sum(r['confidence'] for r in rules) / len(rules) if rules else 0.0,
            'average_zhangs_metric': sum(r['zhangs_metric'] for r in rules) / len(rules) if rules else 0.0,
            'average_interestingness': sum(r['interestingness'] for r in rules) / len(rules) if rules else 0.0,
            'algorithm': f'MLxtend_{self.algorithm}',
            'mode': 'rules'
        }

        return rules, stats

    def __repr__(self):
        return (f"MLxtendMiner(algorithm='{self.algorithm}', min_support={self.min_support}, "
                f"min_confidence={self.min_confidence}, max_items={self.max_items})")
