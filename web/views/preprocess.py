import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from views.utils import get_column_type, to_numeric_safe, save_version, reset_to_original, revert_to_version
from src.preprocessing.discretization import DataDiscretizer
from src.preprocessing.imputation import DataImputer
from src.preprocessing.outlier_handling import OutlierHandler


def format_cols_summary(cols, max_show=3):
    """Format column list for display: 'col1, col2, col3' or 'col1, col2 +N more'."""
    if len(cols) <= max_show:
        return ", ".join(cols)
    return f"{', '.join(cols[:max_show])} +{len(cols) - max_show} more"


DISCRETIZATION_METHODS = {
    "equal_frequency": {
        "name": "Equal Frequency",
        "description": "Each bin contains approximately the same number of samples. Good when data is skewed.",
        "example": "100 samples with 5 bins → each bin has ~20 samples",
        "supervised": False
    },
    "equal_width": {
        "name": "Equal Width",
        "description": "Divides the range of values into equal-sized intervals. Simple and fast.",
        "example": "Range 0-100 with 5 bins → [0-20], [20-40], [40-60], [60-80], [80-100]",
        "supervised": False
    },
    "kmeans": {
        "name": "K-Means Clustering",
        "description": "Groups similar values together using clustering. Good for finding natural groupings.",
        "example": "Values naturally cluster around certain points",
        "supervised": False
    },
    "quantile": {
        "name": "Quantile-Based",
        "description": "Bins based on data percentiles. Similar to equal frequency but uses quantile boundaries.",
        "example": "5 bins at 20th, 40th, 60th, 80th percentiles",
        "supervised": False
    },
    "zscore": {
        "name": "Z-Score Based",
        "description": "Bins based on standard deviations from the mean. Good for normally distributed data.",
        "example": "Low (<-1σ), Below Average (-1σ to 0), Above Average (0 to +1σ), High (>+1σ)",
        "supervised": False
    },
    "entropy": {
        "name": "Entropy-Based (Supervised)",
        "description": "Creates bins that best separate the target classes. Uses information gain to find optimal cut points.",
        "example": "Finds age=45 as cut point if it best separates healthy/unhealthy",
        "supervised": True
    },
    "decision_tree": {
        "name": "Decision Tree (Supervised)",
        "description": "Uses a decision tree to find optimal split points for the target. Similar to entropy but uses tree structure.",
        "example": "Tree finds that splitting at glucose=126 best predicts diabetes",
        "supervised": True
    },
    "chimerge": {
        "name": "ChiMerge (Supervised)",
        "description": "Merges adjacent intervals based on chi-square test. Finds statistically significant boundaries.",
        "example": "Merges bins until chi-square shows significant difference between adjacent bins",
        "supervised": True
    }
}

IMPUTATION_METHODS = {
    "numerical": {
        "mean": {
            "name": "Mean",
            "description": "Replace missing values with the column average. Simple but sensitive to outliers."
        },
        "median": {
            "name": "Median",
            "description": "Replace with the middle value. More robust to outliers than mean."
        },
        "knn": {
            "name": "KNN (K-Nearest Neighbors)",
            "description": "Predict missing values based on similar rows. More accurate but slower."
        },
        "mice": {
            "name": "MICE (Multiple Imputation)",
            "description": "Iteratively predicts each missing value using other columns. Most sophisticated."
        },
        "interpolate": {
            "name": "Linear Interpolation",
            "description": "Estimates missing values by interpolating between neighbors. Good for time series."
        }
    },
    "categorical": {
        "mode": {
            "name": "Most Frequent",
            "description": "Replace with the most common value in the column."
        },
        "constant": {
            "name": "Constant ('MISSING')",
            "description": "Replace with a constant value. Preserves the information that data was missing."
        },
        "knn_categorical": {
            "name": "KNN for Categories",
            "description": "Predict category based on similar rows. More accurate but slower."
        }
    }
}

OUTLIER_METHODS = {
    "iqr": {
        "name": "IQR (Interquartile Range)",
        "description": "Classic method. Outliers are values below Q1-1.5×IQR or above Q3+1.5×IQR.",
        "param": "iqr_multiplier",
        "param_name": "IQR Multiplier",
        "param_default": 1.5,
        "param_range": (1.0, 3.0)
    },
    "zscore": {
        "name": "Z-Score",
        "description": "Outliers are values more than N standard deviations from the mean.",
        "param": "zscore_threshold",
        "param_name": "Z-Score Threshold",
        "param_default": 3.0,
        "param_range": (2.0, 4.0)
    },
    "isolation_forest": {
        "name": "Isolation Forest",
        "description": "ML-based method. Detects anomalies by how easily they can be isolated.",
        "param": "contamination",
        "param_name": "Expected Outlier %",
        "param_default": 0.05,
        "param_range": (0.01, 0.2)
    },
    "lof": {
        "name": "Local Outlier Factor",
        "description": "Compares local density to neighbors. Good for clusters with different densities.",
        "param": "contamination",
        "param_name": "Expected Outlier %",
        "param_default": 0.05,
        "param_range": (0.01, 0.2)
    },
    "winsorize": {
        "name": "Winsorization",
        "description": "Caps extreme values at specified percentiles. Always caps, no detection step.",
        "param": "winsorize_limits",
        "param_name": "Percentile to cap",
        "param_default": 0.05,
        "param_range": (0.01, 0.1)
    },
    "clip": {
        "name": "Percentile Clipping",
        "description": "Clips values outside specified percentile range.",
        "param": "clip_quantiles",
        "param_name": "Lower/Upper percentile",
        "param_default": (0.01, 0.99),
        "param_range": None
    }
}

OUTLIER_ACTIONS = {
    "cap": {
        "name": "Cap/Clip to Bounds",
        "description": "Replace outliers with the boundary values. Keeps all rows."
    },
    "remove": {
        "name": "Remove Rows",
        "description": "Delete entire rows containing outliers. Reduces dataset size."
    },
    "nan": {
        "name": "Mark as Missing",
        "description": "Replace outliers with NaN. Can impute later if needed."
    },
    "flag": {
        "name": "Flag Only",
        "description": "Add a new column indicating outliers. Keeps original values."
    }
}


def render():
    st.header("⚙️ Preprocess Data")

    if st.session_state.current_data is None:
        st.warning("Please upload data first.")
        if st.button("Go to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return

    df = st.session_state.current_data

    # Show last preprocessing result if exists
    if "last_preprocess_result" in st.session_state and st.session_state.last_preprocess_result:
        result = st.session_state.last_preprocess_result
        st.success(f"✓ **{result['operation']}** completed successfully!")

        with st.expander("📋 View Result", expanded=True):
            st.markdown(f"**{result['message']}**")
            if "preview_cols" in result and result["preview_cols"]:
                st.markdown("**Preview of affected columns:**")
                st.dataframe(df[result["preview_cols"]].head(15), width="stretch")
            elif "preview_all" in result and result["preview_all"]:
                st.markdown("**Data preview:**")
                st.dataframe(df.head(15), width="stretch")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("✓ Dismiss", type="secondary"):
                st.session_state.last_preprocess_result = None
                st.rerun()
        with col2:
            if st.button("👁️ View Full Data"):
                st.session_state.last_preprocess_result = None
                st.session_state.current_page = "view_data"
                st.rerun()

        st.markdown("---")

    st.info("""
    **Why preprocess?** Clean and prepare your data before analysis.
    - **Discretization**: Convert numerical values to categories (required for rule mining)
    - **Handle Missing**: Fill or remove missing values
    - **Handle Outliers**: Detect and handle extreme values
    - **Create Labels**: Define new categorical columns based on conditions
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔢 Discretization",
        "🩹 Handle Missing",
        "📉 Handle Outliers",
        "🏷️ Create Labels",
        "📜 History"
    ])

    with tab1:
        render_discretization(df)
    with tab2:
        render_imputation(df)
    with tab3:
        render_outliers(df)
    with tab4:
        render_create_labels(df)
    with tab5:
        render_history()

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Explore"):
            st.session_state.current_page = "explore"
            st.rerun()
    with col2:
        if st.button("➡️ Continue to Analyze", type="primary"):
            st.session_state.current_page = "analyze"
            st.rerun()


def render_discretization(df):
    st.subheader("Discretization")

    with st.expander("ℹ️ What is discretization?"):
        st.markdown("""
        **Discretization** converts numerical values into categories (bins).

        **Why is it needed?** Association rule mining finds patterns like:
        *"IF age=high AND glucose=high THEN diabetes=yes"*

        This requires categorical values, not numbers like "age=45".

        **Example:** Age 25 → "Young", Age 45 → "Middle", Age 70 → "Senior"
        """)

    # Use column types from session state
    num_cols = [c for c in df.columns if get_column_type(c, df) == "numerical"]

    if not num_cols:
        st.success("✓ No numerical columns - data is already categorical!")
        return

    st.warning(f"Found {len(num_cols)} numerical columns that should be discretized for rule mining.")

    # Method selection
    method_options = list(DISCRETIZATION_METHODS.keys())
    method_labels = [f"{DISCRETIZATION_METHODS[m]['name']}" for m in method_options]

    col1, col2 = st.columns(2)
    with col1:
        method_idx = st.selectbox(
            "Method",
            range(len(method_options)),
            format_func=lambda i: method_labels[i]
        )
        method = method_options[method_idx]
        method_info = DISCRETIZATION_METHODS[method]

    with col2:
        n_bins = st.slider("Number of bins/categories", 2, 10, 5)

    # Show method description
    st.info(f"**{method_info['name']}:** {method_info['description']}")
    st.caption(f"Example: {method_info['example']}")

    # Target column for supervised methods
    target_col = None
    if method_info["supervised"]:
        st.warning("This is a **supervised method** - it uses the target column to optimize bin boundaries.")
        label_cols = [c for c in df.columns if c.startswith("label_")]
        target_options = label_cols if label_cols else df.columns.tolist()
        target_col = st.selectbox("Target column (required)", target_options)

    # Column selection with auto-detect option
    st.markdown("**Columns to discretize:**")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    # Track if buttons were clicked (before widget instantiation)
    select_all_clicked = btn_col1.button("Select All Numerical", key="disc_select_all", width="stretch")
    clear_all_clicked = btn_col2.button("Clear All", key="disc_clear_all", width="stretch")

    # Initialize multiselect key if not exists
    if "disc_cols_multiselect" not in st.session_state:
        st.session_state.disc_cols_multiselect = num_cols.copy()

    # Handle button clicks by updating the widget's key BEFORE it's instantiated
    if select_all_clicked:
        st.session_state.disc_cols_multiselect = num_cols.copy()
    if clear_all_clicked:
        st.session_state.disc_cols_multiselect = []

    # Remove any columns that are no longer numerical (already discretized)
    st.session_state.disc_cols_multiselect = [c for c in st.session_state.disc_cols_multiselect if c in num_cols]

    selected_cols = st.multiselect(
        "Columns to discretize",
        num_cols,
        key="disc_cols_multiselect",
        help="Select numerical columns to convert to categories",
        label_visibility="collapsed"
    )

    if selected_cols:
        with st.expander("Preview original values"):
            st.dataframe(df[selected_cols].head(10), width="stretch")

    if st.button("▶️ Apply Discretization", type="primary", disabled=not selected_cols):
        try:
            with st.spinner("Discretizing..."):
                # Create a copy
                df_to_discretize = df.copy()

                # Convert selected columns to numeric first (handles comma-separated numbers)
                for col in selected_cols:
                    df_to_discretize[col] = to_numeric_safe(df_to_discretize[col])

                discretizer = DataDiscretizer(
                    method=method,
                    n_bins=n_bins,
                    target_col=target_col
                )

                # Discretize only selected columns
                if method_info["supervised"]:
                    # For supervised, we need target col included
                    cols_for_disc = selected_cols + [target_col] if target_col not in selected_cols else selected_cols
                    disc_df = discretizer.discretize(df_to_discretize[cols_for_disc])
                    # Update only the discretized feature columns
                    for col in selected_cols:
                        df_to_discretize[col] = disc_df[col]
                else:
                    disc_df = discretizer.discretize(df_to_discretize[selected_cols])
                    for col in selected_cols:
                        df_to_discretize[col] = disc_df[col]

                st.session_state.current_data = df_to_discretize

                # Update column types - discretized columns are now categorical
                column_types = st.session_state.get("column_types", {})
                for col in selected_cols:
                    column_types[col] = "categorical"
                st.session_state.column_types = column_types

                st.session_state.processing_history.append({
                    "step": "Discretization",
                    "method": method_info["name"],
                    "columns": format_cols_summary(selected_cols),
                    "n_bins": n_bins
                })

                # Save to version history with descriptive details
                cols_detail = format_cols_summary(selected_cols)
                save_version(
                    f"Discretize: {cols_detail}",
                    df_to_discretize,
                    column_types,
                    details=f"{method_info['name']}, {n_bins} bins"
                )

                # Store result for display
                st.session_state.last_preprocess_result = {
                    "operation": "Discretization",
                    "message": f"Discretized {len(selected_cols)} columns using {method_info['name']} method with {n_bins} bins.",
                    "preview_cols": selected_cols
                }

            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def render_imputation(df):
    st.subheader("Handle Missing Values")

    missing_count = df.isna().sum().sum()

    if missing_count == 0:
        st.success("✓ No missing values in the dataset!")
        return

    st.warning(f"Found **{missing_count}** missing values across the dataset.")

    with st.expander("ℹ️ What is imputation?"):
        st.markdown("""
        **Imputation** fills in missing values so the data can be analyzed.

        Missing values can cause problems for many algorithms.
        The right strategy depends on your data and why values are missing.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**For numerical columns:**")
        num_methods = list(IMPUTATION_METHODS["numerical"].keys())
        num_strategy = st.selectbox(
            "Strategy",
            num_methods,
            index=1,  # median by default
            format_func=lambda x: IMPUTATION_METHODS["numerical"][x]["name"],
            key="num_impute"
        )
        st.caption(IMPUTATION_METHODS["numerical"][num_strategy]["description"])

    with col2:
        st.markdown("**For categorical columns:**")
        cat_methods = list(IMPUTATION_METHODS["categorical"].keys())
        cat_strategy = st.selectbox(
            "Strategy",
            cat_methods,
            format_func=lambda x: IMPUTATION_METHODS["categorical"][x]["name"],
            key="cat_impute"
        )
        st.caption(IMPUTATION_METHODS["categorical"][cat_strategy]["description"])

    label_cols = [c for c in df.columns if c.startswith("label_")]
    exclude = st.multiselect(
        "Exclude columns from imputation",
        df.columns.tolist(),
        default=label_cols,
        help="These columns will keep their missing values unchanged"
    )

    if st.button("▶️ Apply Imputation", type="primary"):
        try:
            with st.spinner("Imputing missing values..."):
                # Convert columns marked as numerical to actual numeric dtype
                df_for_impute = df.copy()
                num_cols = [c for c in df.columns if get_column_type(c, df) == "numerical" and c not in exclude]
                for col in num_cols:
                    df_for_impute[col] = to_numeric_safe(df_for_impute[col])

                imputer = DataImputer(
                    numerical_strategy=num_strategy,
                    categorical_strategy=cat_strategy
                )
                new_df = imputer.fit_transform(df_for_impute, exclude_cols=exclude)

                new_missing = new_df.isna().sum().sum()

                st.session_state.current_data = new_df
                filled = missing_count - new_missing
                st.session_state.processing_history.append({
                    "step": "Imputation",
                    "numerical": IMPUTATION_METHODS["numerical"][num_strategy]["name"],
                    "categorical": IMPUTATION_METHODS["categorical"][cat_strategy]["name"],
                    "filled": filled
                })

                save_version(
                    f"Impute: {filled} values filled",
                    new_df,
                    details=f"{num_strategy}/{cat_strategy}"
                )

                # Store result for display
                filled = missing_count - new_missing
                st.session_state.last_preprocess_result = {
                    "operation": "Imputation",
                    "message": f"Filled {filled} missing values. Remaining: {new_missing}. Methods: {num_strategy} (numerical), {cat_strategy} (categorical).",
                    "preview_all": True
                }

            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def render_outliers(df):
    st.subheader("Handle Outliers")

    with st.expander("ℹ️ What are outliers?"):
        st.markdown("""
        **Outliers** are extreme values that are unusually far from other values.

        They can:
        - Skew statistical measures (mean, std)
        - Affect model training
        - Be errors OR genuine extreme cases

        **Tip:** Explore your data first to understand if outliers are errors or real.
        """)

    # Use column types from session state
    num_cols = [c for c in df.columns if get_column_type(c, df) == "numerical"]

    if not num_cols:
        st.info("No numerical columns for outlier detection.")
        return

    col1, col2 = st.columns(2)

    with col1:
        method_options = list(OUTLIER_METHODS.keys())
        method = st.selectbox(
            "Detection method",
            method_options,
            format_func=lambda x: OUTLIER_METHODS[x]["name"]
        )
        method_info = OUTLIER_METHODS[method]
        st.caption(method_info["description"])

    with col2:
        action_options = list(OUTLIER_ACTIONS.keys())
        action = st.selectbox(
            "Action",
            action_options,
            format_func=lambda x: OUTLIER_ACTIONS[x]["name"]
        )
        st.caption(OUTLIER_ACTIONS[action]["description"])

    # Method-specific parameter
    param_value = None
    if method_info.get("param_range"):
        param_range = method_info["param_range"]
        param_value = st.slider(
            method_info["param_name"],
            param_range[0],
            param_range[1],
            method_info["param_default"],
            step=0.1 if param_range[1] <= 5 else 0.01
        )

    selected_cols = st.multiselect(
        "Columns to check for outliers",
        num_cols,
        default=num_cols
    )

    # Show outlier preview
    if selected_cols:
        with st.expander("Preview detected outliers"):
            outlier_preview = []
            for col in selected_cols[:5]:  # Limit preview
                col_data = to_numeric_safe(df[col]).dropna()
                if len(col_data) == 0:
                    continue

                if method == "iqr":
                    q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                    iqr = q3 - q1
                    mult = param_value if param_value else 1.5
                    outliers = ((col_data < q1 - mult * iqr) | (col_data > q3 + mult * iqr)).sum()
                elif method == "zscore":
                    thresh = param_value if param_value else 3.0
                    z = np.abs((col_data - col_data.mean()) / col_data.std())
                    outliers = (z > thresh).sum()
                else:
                    outliers = int(len(col_data) * 0.05)  # Estimate

                if outliers > 0:
                    outlier_preview.append({
                        "Column": col,
                        "Outliers": int(outliers),
                        "Percentage": f"{outliers / len(col_data) * 100:.1f}%"
                    })

            if outlier_preview:
                st.dataframe(pd.DataFrame(outlier_preview), width="stretch", hide_index=True)
            else:
                st.success("No outliers detected with current settings")

    if st.button("▶️ Apply Outlier Handling", type="primary", disabled=not selected_cols):
        try:
            with st.spinner("Handling outliers..."):
                # Build kwargs based on method
                kwargs = {}
                if method == "iqr" and param_value:
                    kwargs["iqr_multiplier"] = param_value
                elif method == "zscore" and param_value:
                    kwargs["zscore_threshold"] = param_value
                elif method in ["isolation_forest", "lof"] and param_value:
                    kwargs["contamination"] = param_value
                elif method == "winsorize" and param_value:
                    kwargs["winsorize_limits"] = (param_value, param_value)
                elif method == "clip":
                    kwargs["clip_quantiles"] = (0.01, 0.99)

                # Convert selected columns to numeric first
                df_for_outliers = df.copy()
                for col in selected_cols:
                    df_for_outliers[col] = to_numeric_safe(df_for_outliers[col])

                handler = OutlierHandler(method=method, action=action, **kwargs)
                new_df = handler.fit_transform(df_for_outliers,
                                               exclude_cols=[c for c in df.columns if c not in selected_cols])

                rows_diff = len(df) - len(new_df)

                st.session_state.current_data = new_df
                cols_detail = format_cols_summary(selected_cols)
                st.session_state.processing_history.append({
                    "step": "Outlier Handling",
                    "method": method_info["name"],
                    "action": OUTLIER_ACTIONS[action]["name"],
                    "columns": cols_detail,
                    "rows_affected": rows_diff if action == "remove" else "N/A"
                })

                save_version(
                    f"Outliers: {cols_detail}",
                    new_df,
                    details=f"{method_info['name']}, {action}"
                )

                # Store result for display
                if rows_diff > 0:
                    msg = f"Removed {rows_diff} rows with outliers using {method_info['name']} method."
                else:
                    msg = f"Outliers handled using {method_info['name']} method with {OUTLIER_ACTIONS[action]['name']} action."
                st.session_state.last_preprocess_result = {
                    "operation": "Outlier Handling",
                    "message": msg,
                    "preview_cols": selected_cols
                }

            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def render_create_labels(df):
    st.subheader("Create Custom Labels")

    with st.expander("ℹ️ What is this?"):
        st.markdown("""
        Create new categorical columns (labels) based on conditions on existing features.

        **Example:** Create a "High Risk" label where:
        - IF glucose > 126 AND blood_pressure > 140 THEN "High Risk"
        - OTHERWISE "Normal"

        This is useful for:
        - Defining outcome variables for rule mining
        - Combining multiple features into a single category
        - Creating clinical classifications based on thresholds
        """)

    # Initialize session state for conditions
    if "label_conditions" not in st.session_state:
        st.session_state.label_conditions = []

    st.markdown("### Define Conditions")

    # Add new condition
    st.markdown("**Add a condition:**")
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

    # Use column types from session state
    num_cols = [c for c in df.columns if get_column_type(c, df) == "numerical"]
    all_cols = df.columns.tolist()

    with col1:
        cond_feature = st.selectbox("Feature", all_cols, key="cond_feature")
    with col2:
        cond_operator = st.selectbox("Operator", ["<", "<=", ">", ">=", "==", "!="], key="cond_operator")
    with col3:
        # Show appropriate input based on column type
        if cond_feature in num_cols:
            num_col_data = to_numeric_safe(df[cond_feature]).dropna()
            if len(num_col_data) > 0:
                col_min = float(num_col_data.min())
                col_max = float(num_col_data.max())
            else:
                col_min, col_max = 0.0, 100.0
            cond_value = st.number_input("Threshold", value=(col_min + col_max) / 2, key="cond_value")
        else:
            unique_vals = df[cond_feature].dropna().unique().tolist()[:50]
            cond_value = st.selectbox("Value", unique_vals, key="cond_value_cat")
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", key="add_condition"):
            st.session_state.label_conditions.append({
                "feature": cond_feature,
                "operator": cond_operator,
                "value": cond_value if cond_feature in num_cols else cond_value
            })
            st.rerun()

    # Show current conditions
    if st.session_state.label_conditions:
        st.markdown("**Current conditions (combined with AND):**")
        for i, cond in enumerate(st.session_state.label_conditions):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.code(f"{cond['feature']} {cond['operator']} {cond['value']}")
            with col2:
                if st.button("🗑️", key=f"del_cond_{i}"):
                    st.session_state.label_conditions.pop(i)
                    st.rerun()

        if st.button("🗑️ Clear All Conditions"):
            st.session_state.label_conditions = []
            st.rerun()

    st.markdown("---")
    st.markdown("### Define Label")

    col1, col2, col3 = st.columns(3)
    with col1:
        new_col_name = st.text_input("New column name", value="label_custom", help="Name for the new label column")
    with col2:
        value_if_true = st.text_input("Value if conditions are TRUE", value="Yes",
                                      help="Label value when all conditions are met")
    with col3:
        value_if_false = st.text_input("Value if conditions are FALSE", value="No",
                                       help="Label value when conditions are not met")

    # Preview
    if st.session_state.label_conditions and new_col_name:
        st.markdown("### Preview")

        # Build condition mask
        mask = pd.Series([True] * len(df), index=df.index)
        condition_str_parts = []

        for cond in st.session_state.label_conditions:
            feat = cond["feature"]
            op = cond["operator"]
            val = cond["value"]
            condition_str_parts.append(f"{feat} {op} {val}")

            # Convert to numeric for comparison if it's a numerical column
            if feat in num_cols:
                feat_data = to_numeric_safe(df[feat])
            else:
                feat_data = df[feat]

            if op == "<":
                mask = mask & (feat_data < val)
            elif op == "<=":
                mask = mask & (feat_data <= val)
            elif op == ">":
                mask = mask & (feat_data > val)
            elif op == ">=":
                mask = mask & (feat_data >= val)
            elif op == "==":
                mask = mask & (feat_data == val)
            elif op == "!=":
                mask = mask & (feat_data != val)

        condition_str = " AND ".join(condition_str_parts)
        st.markdown(f"**Rule:** IF {condition_str} THEN `{value_if_true}` ELSE `{value_if_false}`")

        true_count = mask.sum()
        false_count = len(df) - true_count
        st.markdown(
            f"**Distribution:** {value_if_true}: {true_count} rows ({true_count / len(df) * 100:.1f}%) | {value_if_false}: {false_count} rows ({false_count / len(df) * 100:.1f}%)")

        # Preview table
        preview_df = df.head(10).copy()
        preview_mask = mask.head(10)
        preview_df[new_col_name] = preview_mask.map({True: value_if_true, False: value_if_false})
        st.dataframe(preview_df[[col for col in preview_df.columns if
                                 col == new_col_name or col in [c["feature"] for c in
                                                                st.session_state.label_conditions]]], width="stretch")

    # Apply button
    can_apply = bool(st.session_state.label_conditions) and bool(new_col_name) and new_col_name not in df.columns
    if new_col_name in df.columns:
        st.warning(f"Column '{new_col_name}' already exists. Choose a different name.")

    if st.button("▶️ Create Label Column", type="primary", disabled=not can_apply):
        try:
            # Build condition mask
            mask = pd.Series([True] * len(df), index=df.index)
            for cond in st.session_state.label_conditions:
                feat = cond["feature"]
                op = cond["operator"]
                val = cond["value"]

                # Convert to numeric for comparison if it's a numerical column
                if feat in num_cols:
                    feat_data = to_numeric_safe(df[feat])
                else:
                    feat_data = df[feat]

                if op == "<":
                    mask = mask & (feat_data < val)
                elif op == "<=":
                    mask = mask & (feat_data <= val)
                elif op == ">":
                    mask = mask & (feat_data > val)
                elif op == ">=":
                    mask = mask & (feat_data >= val)
                elif op == "==":
                    mask = mask & (feat_data == val)
                elif op == "!=":
                    mask = mask & (feat_data != val)

            new_df = df.copy()
            new_df[new_col_name] = mask.map({True: value_if_true, False: value_if_false})

            # Update column types
            column_types = st.session_state.get("column_types", {})
            column_types[new_col_name] = "categorical"
            st.session_state.column_types = column_types

            st.session_state.current_data = new_df
            st.session_state.processing_history.append({
                "step": "Create Label",
                "column": new_col_name,
                "conditions": len(st.session_state.label_conditions),
                "true_value": value_if_true,
                "false_value": value_if_false
            })

            save_version(
                f"Label: {new_col_name}",
                new_df,
                column_types,
                details=f"{len(st.session_state.label_conditions)} conditions"
            )

            # Clear conditions
            st.session_state.label_conditions = []

            # Store result for display
            true_count = mask.sum()
            st.session_state.last_preprocess_result = {
                "operation": "Create Label",
                "message": f"Created new column '{new_col_name}' with {true_count} '{value_if_true}' and {len(df) - true_count} '{value_if_false}' values.",
                "preview_cols": [new_col_name]
            }

            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def render_history():
    st.subheader("Processing History")

    history = st.session_state.processing_history
    data_versions = st.session_state.get("data_versions", [])

    if not history:
        st.info("No preprocessing steps applied yet.")
        return

    for i, step in enumerate(history):
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(f"### {i + 1}")
            with cols[1]:
                st.markdown(f"**{step['step']}**")
                details = {k: v for k, v in step.items() if k != "step"}
                for k, v in details.items():
                    st.caption(f"{k}: {v}")
        st.divider()

    st.subheader("Version Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reset to Original Data", type="secondary"):
            reset_to_original()
            st.rerun()

    with col2:
        if len(data_versions) > 1:
            # Use index-based selection to handle any duplicate names
            version_names = [v["name"] for v in data_versions[:-1]]
            selected_idx = st.selectbox(
                "Revert to version",
                range(len(version_names)),
                format_func=lambda i: version_names[i],
                key="preprocess_revert_select"
            )
            if st.button("⏪ Revert to Selected Version"):
                revert_to_version(selected_idx)
                st.success(f"Reverted to: {version_names[selected_idx]}")
                st.rerun()
        else:
            st.caption("No previous versions to revert to")
