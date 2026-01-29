"""
NiaARM-based rule mining for numerical data.

NiaARM uses nature-inspired optimization algorithms and works with
numerical features. It generates association rules only (not itemsets).

Note: NiaARM generates rules for numerical data, so it cannot be used
with CORELS (which requires categorical itemsets).
"""
from typing import Dict, List, Tuple, Any
import pandas as pd
from niapy.algorithms.basic import (
    HarrisHawksOptimization,
    GreyWolfOptimizer,
    SineCosineAlgorithm,
    MothFlameOptimizer
)
from niaarm import Dataset, get_rules

from src.rule_mining.base import AssociationRuleMiner


class NiaARMMiner(AssociationRuleMiner):
    """
    NiaARM nature-inspired rule miner for numerical data.

    Uses optimization algorithms (HHO, GWO, SCA, MFO, etc.) to discover
    association rules from numerical features.

    Compatible with: CBA only (generates rules with numerical ranges)
    NOT compatible with: CORELS (doesn't generate categorical itemsets)
    """

    def __init__(
            self,
            algorithm,
            max_evals: int = 50000,
            metrics: List[str] = None,
            **kwargs
    ):
        """
        Initialize NiaARM miner.

        Args:
            algorithm: NiaPy algorithm instance (HarrisHawksOptimization, GreyWolfOptimizer, etc.)
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            max_evals: Maximum number of evaluations
            metrics: List of metrics to optimize (default: ['support', 'confidence'])
        """
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.max_evals = max_evals
        self.metrics = metrics if metrics is not None else ['support', 'confidence']

    def mine_rules(self, data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Mine association rules using nature-inspired optimization.

        Args:
            data: DataFrame with numerical or categorical features

        Returns:
            Tuple of (rules, stats)
        """
        import time

        # Create NiaARM dataset
        dataset = Dataset(data)

        start_time = time.time()

        # Mine rules
        rules_obj, run_time = get_rules(
            dataset,
            algorithm=self.algorithm,
            metrics=self.metrics,
            max_evals=self.max_evals,
            logging=False
        )

        execution_time = time.time() - start_time

        if len(rules_obj) == 0:
            return [], {
                'num_rules': 0,
                'execution_time': execution_time,
                'algorithm': f'NiaARM_{self.algorithm.__class__.__name__}',
                'mode': 'rules'
            }

        # Convert to standard format
        rules = []
        for rule in rules_obj:
            rules.append({
                'antecedent': rule.antecedent,
                'consequent': rule.consequent,
                'support': float(rule.support),
                'confidence': float(rule.confidence),
                'coverage': float(rule.coverage) if hasattr(rule, 'coverage') else None,
                'zhangs_metric': float(rule.zhang) if hasattr(rule, 'zhang') else None,
                'lift': float(rule.lift) if hasattr(rule, 'lift') else None,
                'interestingness': float(rule.interestingness) if hasattr(rule, 'interestingness') else None,
                'conviction': float(rule.conviction) if hasattr(rule, 'conviction') else None
            })

        return rules, execution_time

    def __repr__(self):
        return (f"NiaARMMiner(algorithm={self.algorithm.__class__.__name__}, "
                f"min_support={self.min_support}, min_confidence={self.min_confidence}, "
                f"max_evals={self.max_evals})")


class HHOMiner(NiaARMMiner):
    """
    Convenience class for Harris Hawks Optimization-based mining.

    Harris Hawks Optimization (HHO) is a novel nature-inspired algorithm
    proposed in 2019 that simulates the cooperative hunting behavior of
    Harris hawks. It provides excellent balance between exploration and
    exploitation phases.
    """

    def __init__(
            self,
            population: int = 40,
            levy: float = 0.01,
            max_evals: int = 50000,
            **kwargs
    ):
        algorithm = HarrisHawksOptimization(population_size=population, levy=levy)
        super().__init__(algorithm, max_evals, **kwargs)


class GWOMiner(NiaARMMiner):
    """
    Convenience class for Grey Wolf Optimizer-based mining.

    Grey Wolf Optimizer (GWO) mimics the leadership hierarchy and hunting
    mechanism of grey wolves. It's highly effective for various optimization
    problems with a good balance between exploration and exploitation.
    """

    def __init__(
            self,
            population: int = 50,
            max_evals: int = 50000,
            **kwargs
    ):
        algorithm = GreyWolfOptimizer(population_size=population)
        super().__init__(algorithm, max_evals, **kwargs)


class SCAMiner(NiaARMMiner):
    """
    Convenience class for Sine Cosine Algorithm-based mining.

    Sine Cosine Algorithm (SCA) uses mathematical sine and cosine functions
    to perform optimization. It provides a unique mathematical approach to
    balancing exploration and exploitation in the search space.
    """

    def __init__(
            self,
            population: int = 25,
            a: float = 3.0,
            r_min: float = 0.0,
            r_max: float = 2.0,
            max_evals: int = 50000,
            **kwargs
    ):
        algorithm = SineCosineAlgorithm(population_size=population, a=a, r_min=r_min, r_max=r_max)
        super().__init__(algorithm, max_evals, **kwargs)


class MFOMiner(NiaARMMiner):
    """
    Convenience class for Moth-Flame Optimizer-based mining.

    Moth-Flame Optimizer (MFO) simulates the navigation method of moths
    using transverse orientation. It maintains a good balance between
    exploration and exploitation through a logarithmic spiral movement.
    """

    def __init__(
            self,
            population: int = 50,
            max_evals: int = 50000,
            **kwargs
    ):
        algorithm = MothFlameOptimizer(population_size=population)
        super().__init__(algorithm, max_evals, **kwargs)
