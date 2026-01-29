def filter_rules(rules, criterion: str, threshold: float):
    """
    Filters rules based on a criterion >= threshold.
    Returns both filtered rules and a stats dictionary.

    Args:
        rules: List of rule dictionaries
        criterion: The rule metric to filter on (e.g., 'support', 'confidence', 'zhangs_metric')
        threshold: Minimum value for the criterion (inclusive)

    Returns:
        Tuple of (filtered_rules, stats) where:
            filtered_rules: List of rules meeting the criterion
            stats: Dictionary with average statistics for filtered rules
    """
    # Fast filtering with list comprehension
    filtered_rule_list = [rule for rule in rules if rule.get(criterion, float("-inf")) >= threshold]

    return filtered_rule_list


def filter_rules_by_pattern(
    rules,
    antecedent_contains: list = None,
    consequent_contains: list = None,
    antecedent_excludes: list = None,
    consequent_excludes: list = None,
    match_any: bool = False
):
    """
    Filter rules by antecedent/consequent patterns.

    Works with multiple rule formats:
    - String: "feature__value" or "feature=value"
    - List of dicts: [{'feature': 'X', 'value': 'Y'}]
    - Frozenset: frozenset({'feature__value', ...})

    Args:
        rules: List of rule dictionaries
        antecedent_contains: List of patterns that must appear in antecedent
        consequent_contains: List of patterns that must appear in consequent
        antecedent_excludes: List of patterns that must NOT appear in antecedent
        consequent_excludes: List of patterns that must NOT appear in consequent
        match_any: If True, match if ANY pattern matches. If False, ALL must match.

    Returns:
        List of filtered rules
    """
    def normalize_itemset(val):
        """Convert itemset to a set of normalized strings for matching."""
        if val is None:
            return set()
        if isinstance(val, str):
            return {val.lower()}
        if isinstance(val, frozenset):
            return {str(item).lower() for item in val}
        if isinstance(val, dict):
            if 'feature' in val and 'value' in val:
                return {f"{val['feature']}__{val['value']}".lower()}
            return {f"{k}__{v}".lower() for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            result = set()
            for item in val:
                if isinstance(item, dict):
                    if 'feature' in item and 'value' in item:
                        result.add(f"{item['feature']}__{item['value']}".lower())
                    else:
                        result.update(f"{k}__{v}".lower() for k, v in item.items())
                elif isinstance(item, str):
                    result.add(item.lower())
            return result
        return {str(val).lower()}

    def matches_patterns(itemset_str, patterns, match_any_pattern):
        """Check if itemset matches the given patterns."""
        if not patterns:
            return True
        itemset_normalized = normalize_itemset(itemset_str)
        patterns_lower = [p.lower() for p in patterns]

        if match_any_pattern:
            return any(
                any(p in item for item in itemset_normalized)
                for p in patterns_lower
            )
        else:
            return all(
                any(p in item for item in itemset_normalized)
                for p in patterns_lower
            )

    def excludes_patterns(itemset_str, patterns):
        """Check if itemset does NOT contain any of the patterns."""
        if not patterns:
            return True
        itemset_normalized = normalize_itemset(itemset_str)
        patterns_lower = [p.lower() for p in patterns]

        return not any(
            any(p in item for item in itemset_normalized)
            for p in patterns_lower
        )

    filtered = []
    for rule in rules:
        ant = rule.get('antecedent') or rule.get('antecedents')
        cons = rule.get('consequent') or rule.get('consequents')

        ant_match = matches_patterns(ant, antecedent_contains, match_any)
        cons_match = matches_patterns(cons, consequent_contains, match_any)
        ant_exclude = excludes_patterns(ant, antecedent_excludes)
        cons_exclude = excludes_patterns(cons, consequent_excludes)

        if ant_match and cons_match and ant_exclude and cons_exclude:
            filtered.append(rule)

    return filtered


def filter_rules_by_consequent(rules, targets: list, match_any: bool = True):
    """
    Filter rules to keep only those with consequent matching target patterns.

    Args:
        rules: List of rule dictionaries
        targets: List of target patterns (e.g., ['label_iron_deficiency'])
        match_any: If True, match if any target matches

    Returns:
        List of filtered rules
    """
    return filter_rules_by_pattern(rules, consequent_contains=targets, match_any=match_any)


def filter_rules_by_antecedent(rules, patterns: list, match_any: bool = True):
    """
    Filter rules to keep only those with antecedent matching patterns.

    Args:
        rules: List of rule dictionaries
        patterns: List of patterns to match in antecedent
        match_any: If True, match if any pattern matches

    Returns:
        List of filtered rules
    """
    return filter_rules_by_pattern(rules, antecedent_contains=patterns, match_any=match_any)


def filter_itemsets(itemsets, criterion: str = 'support', threshold: float = 0.0):
    """
    Filters frequent itemsets based on a criterion >= threshold.
    Returns both filtered itemsets and a stats dictionary.

    Args:
        itemsets: List of itemset dictionaries (each with 'items' and 'support' keys)
        criterion: The metric to filter on (default: 'support')
        threshold: Minimum value for the criterion (inclusive)

    Returns:
        Tuple of (filtered_itemsets, stats) where:
            filtered_itemsets: List of itemsets meeting the criterion
            stats: Dictionary with average statistics for filtered itemsets
    """
    # Fast filtering with list comprehension
    filtered_itemset_list = [itemset for itemset in itemsets if itemset.get(criterion, float("-inf")) >= threshold]

    count = len(filtered_itemset_list)
    if count == 0:
        stats = {
            "num_itemsets": 0,
            "average_support": 0.0,
        }
        return filtered_itemset_list, stats

    # Compute average support
    avg_support = sum(item.get("support", 0) for item in filtered_itemset_list) / count

    stats = {
        "num_itemsets": count,
        "average_support": round(avg_support, 3),
    }

    return filtered_itemset_list, stats
