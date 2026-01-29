import pandas as pd
import operator
import json

# Map operator strings to functions
OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq
}


def apply_label_rules(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Applies labelling rules to a pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    rules : list[dict]
        Rules in JSON-parsed Python form.

    Returns
    -------
    pandas.DataFrame
        Updated dataframe with new label columns added.
    """

    df = df.copy()

    for rule in rules:
        label_col = rule["label_column"]
        label_val = rule["label_value"]

        # If label column doesn't exist, create it
        if label_col not in df.columns:
            df[label_col] = False

        # Build combined mask
        mask = pd.Series([True] * len(df))

        for cond in rule["conditions"]:
            feature = cond["feature"]

            if cond["operator"] == "between":
                m = (df[feature] >= cond["low"]) & (df[feature] <= cond["high"])
            else:
                op = OPS[cond["operator"]]
                m = op(df[feature], cond["value"])

            mask &= m

        # Apply label
        df.loc[mask, label_col] = label_val

    return df


if __name__ == "__main__":
    DATA_PATH = '../../data/raw/bloodcounts.xlsx'
    LABELLING_RULE_PATH = '../../data/labelling_rules/cbc_morphology_label_rules.json'
    OUTPUT_PATH = '../../data/raw/bloodcounts_labelled.csv'

    data = pd.read_excel(DATA_PATH)
    with open(LABELLING_RULE_PATH) as f:
        labelling_rules = json.load(f)

    df = apply_label_rules(data, labelling_rules)
    df.to_csv(OUTPUT_PATH)