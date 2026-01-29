import subprocess
import time
import numpy as np
from joblib import Parallel, delayed


def create_corels_input_files(rules, labels, dataset_name):
    """
    CORELS expect frequent itemsets and their occurrences as an input file.
    This function creates those files.
    """
    corels_train_dataset_name = dataset_name.lower().replace(" ", "_")
    with open("classification/corels/data/" + corels_train_dataset_name + ".out", "w") as file:
        for row in rules:
            file.write(" ".join(map(str, row)) + "\n")

    with open("classification/corels/data/" + corels_train_dataset_name + ".label", "w") as file:
        for key in labels:
            file.write("{" + key + ":Yes} ")
            file.write(" ".join(map(str, labels[key])) + "\n")


def run_corels(dataset_name):
    """
    Run CORELS' C++ Code
    :param dataset_name:
    :return:
    """
    dataset_name = dataset_name.lower().replace(" ", "_")
    command = "./corels"
    parameters = ["-r", "0.001", "-c", "2", "-p", "1", "../data/" + dataset_name + ".out",
                  "../data/" + dataset_name + ".label"]

    try:
        start = time.time()
        result = subprocess.run([command] + parameters, capture_output=True, text=True, check=True,
                                cwd="./classification/corels/src")
        exec_time = time.time() - start
        optimal_rule_list_file = None
        time.sleep(1)
        # find where the optimal rule is stored by looking at CORELS' stdout
        for line in result.stdout.splitlines():
            if line.startswith("writing optimal rule list to: "):
                optimal_rule_list_file = "classification/corels/src/" + line[len("writing optimal rule list to: "):]
                break
        if optimal_rule_list_file:
            with open(optimal_rule_list_file, "r") as file:
                optimal_rule_list = file.read()
                corel_rule_list_model = parse_corels_rule_lists(optimal_rule_list)
            return corel_rule_list_model, exec_time

    except subprocess.CalledProcessError as e:
        # Handle errors
        print(f"Error occurred: {e}")
        print("Standard Error:")
        print(e.stderr)

    return None, None


def create_corels_freq_items_input(itemset, transactions):
    """
    Create CORELS input format for an itemset (highly optimized).

    Args:
        itemset: Dict of {feature: value} pairs
        transactions: DataFrame or dict of arrays

    Returns:
        List with itemset string followed by binary matches
    """
    corels_freq_item_string = "{" + ",".join(
        f"{key.replace(' ', '-')}:={value}" for key, value in itemset.items()) + "}"

    # Optimized vectorized matching
    if len(itemset) == 0:
        # Empty itemset matches everything
        # Handle both DataFrame and dict of arrays
        n_rows = len(transactions) if hasattr(transactions, '__len__') else len(next(iter(transactions.values())))
        conditions = np.ones(n_rows, dtype=np.int8)
    else:
        # Use reduce with bitwise AND for maximum efficiency
        conditions = None
        for key, value in itemset.items():
            # Support both DataFrame and pre-cached dict of arrays
            if isinstance(transactions, dict):
                col_data = transactions[key]
            else:
                col_data = transactions[key].values

            # Ensure proper comparison by converting to same type if needed
            # This handles string/categorical comparisons correctly
            col_match = (col_data == value).astype(bool)
            if conditions is None:
                conditions = col_match
            else:
                conditions = conditions & col_match

        # Convert to int8 directly (faster than bool->int)
        conditions = conditions.astype(np.int8)

    # Convert to list in one go
    corels_format_row = [corels_freq_item_string] + conditions.tolist()

    return corels_format_row


def parse_corels_rule_lists(rule_list_model_in_text):
    """
    Parse the output of CORELS (rule list models) into Python objects
    Example CORELS output: {spore-print-color_h:=0.0,gill-size_b:=1.0}~0;default~1
    :param rule_list_model_in_text:
    :return:
    """
    condition_blocks = rule_list_model_in_text.split(';')

    result = []

    for block in condition_blocks:
        # Check if the block contains conditions in curly braces
        if '{' in block and '}' in block:
            # Separate the condition and the "then" value using the '~' symbol
            condition_part, then_part = block.split('~')

            # Parse the conditions into key-value pairs by splitting at ':='
            condition_list = []
            for cond in condition_part.strip('{}').split(','):
                key, value = cond.split(':=')
                condition_list.append((key.strip(), value.strip()))

            # Append the parsed condition list and the "then" part to the result
            result.append(condition_list + [int(then_part.strip())])
        else:
            # Handle the default case
            _, default_value = block.split('~')
            result.append([("default", int(default_value.strip()))])

    return result


def test_corels_model(model, test_X, test_y, n_jobs=-1):
    """
    Test CORELS model accuracy (parallelized).

    Args:
        model: Trained CORELS model
        test_X: Test features
        test_y: Test labels
        n_jobs: Number of parallel jobs (-1 uses all cores)

    Returns:
        Accuracy percentage
    """
    test_X.reset_index(drop=True, inplace=True)
    test_y.reset_index(drop=True, inplace=True)

    default_rule = next((rule[-1][1] for rule in model if rule[0][0] == 'default'), None)

    def test_row(i, feature_row, actual_label):
        predicted_label = None

        # Iterate through the conditions in the model
        for condition in model:
            # Skip the default rule for now
            if condition[0][0] == 'default':
                continue

            # Extract conditions and the expected label
            conditions = condition[:-1]
            expected_label = condition[-1]

            # Check if all conditions are satisfied for the current row
            if all(feature_row[key] == value for key, value in conditions):
                predicted_label = expected_label
                break  # Stop checking further rules once a match is found

        # If no conditions match, use the default rule
        if predicted_label is None and default_rule is not None:
            predicted_label = default_rule

        # Compare predicted label with the actual label
        return predicted_label == actual_label

    # Parallelize row testing
    model_holds = Parallel(n_jobs=n_jobs)(
        delayed(test_row)(i, feature_row, test_y.iloc[i, 0])
        for i, feature_row in test_X.iterrows()
    )

    # Calculate the percentage of correct predictions
    accuracy = (sum(model_holds) / len(model_holds)) * 100
    return accuracy
