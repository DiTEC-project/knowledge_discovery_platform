import streamlit as st
import pandas as pd
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from views.utils import get_column_type
from src.rule_mining.aerial_miner import AerialMiner
from src.rule_mining.mlxtend_miner import MLxtendMiner


def get_column_values(df, col):
    """Get unique values for a column"""
    return sorted([str(v) for v in df[col].dropna().unique()])


def render_column_value_selector(df, label, key_prefix, help_text="", allow_all_values=True):
    """
    Render a column and optional value selector.
    Returns a dict of {column: [values]} or {column: None} if all values selected.
    """
    all_cols = df.columns.tolist()
    multiselect_key = f"{key_prefix}_multiselect"

    # Initialize session state for the multiselect widget directly
    if multiselect_key not in st.session_state:
        # Default to label_ columns for targets
        if "target" in key_prefix:
            st.session_state[multiselect_key] = [c for c in all_cols if c.startswith("label_")]
        else:
            st.session_state[multiselect_key] = []

    if f"{key_prefix}_value_filters" not in st.session_state:
        st.session_state[f"{key_prefix}_value_filters"] = {}

    st.markdown(f"**{label}**")
    if help_text:
        st.caption(help_text)

    # Select All / Clear All buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        if st.button("Select All", key=f"{key_prefix}_select_all", width="stretch"):
            # Set the multiselect widget's key directly
            st.session_state[multiselect_key] = all_cols.copy()
            st.rerun()
    with btn_col2:
        if st.button("Clear All", key=f"{key_prefix}_clear_all", width="stretch"):
            # Set the multiselect widget's key directly
            st.session_state[multiselect_key] = []
            st.session_state[f"{key_prefix}_value_filters"] = {}
            st.rerun()

    # Column multiselect - uses the key as source of truth
    selected_cols = st.multiselect(
        "Select columns",
        all_cols,
        key=multiselect_key,
        label_visibility="collapsed"
    )

    # Clean up value filters for removed columns
    current_filters = st.session_state.get(f"{key_prefix}_value_filters", {})
    st.session_state[f"{key_prefix}_value_filters"] = {
        k: v for k, v in current_filters.items()
        if k in selected_cols
    }

    # Value selection for each selected column
    result = {}
    if selected_cols:
        with st.expander(f"🎯 Specify exact values (optional)", expanded=False):
            st.caption("Leave 'All Values' to include all values for a column, or select specific values.")

            for col in selected_cols:
                col_values = get_column_values(df, col)

                # Initialize value filter for this column if needed
                if col not in st.session_state[f"{key_prefix}_value_filters"]:
                    st.session_state[f"{key_prefix}_value_filters"][col] = None  # None means all values

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"`{col}`")
                with col2:
                    current_filter = st.session_state[f"{key_prefix}_value_filters"].get(col)

                    if allow_all_values:
                        use_all = st.checkbox(
                            "All Values",
                            value=(current_filter is None),
                            key=f"{key_prefix}_{col}_all_values"
                        )

                        if use_all:
                            st.session_state[f"{key_prefix}_value_filters"][col] = None
                            result[col] = None
                        else:
                            # Show value selector
                            selected_values = st.multiselect(
                                "Select values",
                                col_values,
                                default=current_filter if current_filter else [],
                                key=f"{key_prefix}_{col}_values",
                                label_visibility="collapsed"
                            )
                            st.session_state[f"{key_prefix}_value_filters"][
                                col] = selected_values if selected_values else None
                            result[col] = selected_values if selected_values else None
                    else:
                        selected_values = st.multiselect(
                            "Select values",
                            col_values,
                            default=current_filter if current_filter else [],
                            key=f"{key_prefix}_{col}_values",
                            label_visibility="collapsed"
                        )
                        st.session_state[f"{key_prefix}_value_filters"][
                            col] = selected_values if selected_values else None
                        result[col] = selected_values if selected_values else None

    return selected_cols, result


AERIAL_PRESETS = {
    "quick_overview": {
        "name": "Quick Overview",
        "description": "Find the most obvious, strong patterns quickly. Returns fewer but highly reliable rules. Best for getting a first impression.",
        "params": {"ant_similarity": 0.5, "cons_similarity": 0.9, "max_antecedents": 2, "epochs": 2, "batch_size": 32,
                   "layer_dims": None}
    },
    "balanced": {
        "name": "Balanced Analysis",
        "description": "Good balance between coverage and reliability. Recommended for most analyses.",
        "params": {"ant_similarity": 0.3, "cons_similarity": 0.8, "max_antecedents": 2, "epochs": 2, "batch_size": 16,
                   "layer_dims": None}
    },
    "deep_search": {
        "name": "Deep Search",
        "description": "Find subtle patterns that might be missed. Returns more rules including weaker associations.",
        "params": {"ant_similarity": 0.1, "cons_similarity": 0.7, "max_antecedents": 3, "epochs": 2, "batch_size": 16,
                   "layer_dims": None}
    },
    "comprehensive": {
        "name": "Comprehensive",
        "description": "Maximum coverage. Finds many patterns including complex ones. Takes longer to run.",
        "params": {"ant_similarity": 0.01, "cons_similarity": 0.7, "max_antecedents": 4, "epochs": 2, "batch_size": 2,
                   "layer_dims": None}
    }
}

FPGROWTH_PRESETS = {
    "strict": {
        "name": "Strict (High Confidence)",
        "description": "Only highly reliable patterns with strong support. Fewer rules but very trustworthy.",
        "params": {"min_support": 0.2, "min_confidence": 0.8, "max_items": 3}
    },
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of coverage and reliability. Recommended starting point.",
        "params": {"min_support": 0.1, "min_confidence": 0.7, "max_items": 4}
    },
    "exploratory": {
        "name": "Exploratory",
        "description": "Find more patterns including less common ones. More rules to explore.",
        "params": {"min_support": 0.01, "min_confidence": 0.5, "max_items": 5}
    }
}


def render():
    st.header("🔍 Run Analysis")

    if st.session_state.current_data is None:
        st.warning("Please upload data first.")
        if st.button("Go to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return

    df = st.session_state.current_data

    # Check for numerical columns using session state column types
    num_cols = [c for c in df.columns if get_column_type(c, df) == "numerical"]
    if len(num_cols) > 0:
        st.warning(f"""
        ⚠️ **{len(num_cols)} numerical columns detected:** {', '.join(num_cols[:5])}{'...' if len(num_cols) > 5 else ''}

        Association rule mining works best with categorical data.
        Consider going back to **Preprocess** to discretize these columns first.
        """)
        if st.button("Go to Preprocess"):
            st.session_state.current_page = "preprocess"
            st.rerun()

    st.markdown("---")

    # Method selection
    method = st.radio(
        "Choose analysis method",
        ["Aerial+ (Neural Network-based)", "FP-Growth (Classic Algorithm)"],
        horizontal=True
    )

    st.markdown("---")

    if "Aerial+" in method:
        render_aerial_config(df)
    else:
        render_fpgrowth_config(df)

    # Show persistent success message from last analysis
    if st.session_state.get("last_analysis_result", {}).get("success"):
        result = st.session_state.last_analysis_result
        st.success(f"✓ Analysis complete! Found **{result['num_rules']}** rules using {result['method']}.")
        if st.button("➡️ View Results", type="primary", width="stretch", key="view_results_btn"):
            st.session_state.last_analysis_result = None
            st.session_state.current_page = "results"
            st.rerun()

    # Navigation
    st.markdown("---")
    if st.button("⬅️ Back to Preprocess"):
        st.session_state.last_analysis_result = None
        st.session_state.current_page = "preprocess"
        st.rerun()


def _select_aerial_preset(key):
    preset = AERIAL_PRESETS[key]
    st.session_state.aerial_preset = key
    st.session_state.aerial_params = preset["params"].copy()
    st.session_state.aerial_ant_similarity = preset["params"]["ant_similarity"]
    st.session_state.aerial_cons_similarity = preset["params"]["cons_similarity"]
    st.session_state.aerial_max_antecedents = preset["params"]["max_antecedents"]
    st.session_state.aerial_epochs = preset["params"]["epochs"]
    st.session_state.aerial_batch_size = preset["params"].get("batch_size") or 32


def render_aerial_config(df):
    st.subheader("Aerial+ Configuration")

    st.markdown("**Select a preset:**")

    # Initialize preset params in session state if not exists
    if "aerial_params" not in st.session_state:
        default_preset = AERIAL_PRESETS["balanced"]
        st.session_state.aerial_params = default_preset["params"].copy()
        st.session_state.aerial_preset = "balanced"

    cols = st.columns(4)
    for i, (key, preset) in enumerate(AERIAL_PRESETS.items()):
        with cols[i]:
            selected = st.session_state.get("aerial_preset") == key
            st.button(
                preset["name"],
                key=f"aerial_{key}",
                width="stretch",
                type="primary" if selected else "secondary",
                on_click=_select_aerial_preset,
                args=(key,)
            )

    selected_preset = st.session_state.get("aerial_preset", "balanced")
    preset = AERIAL_PRESETS[selected_preset]

    st.info(f"**{preset['name']}:** {preset['description']}")

    # Use session state params (which get updated by presets)
    params = st.session_state.get("aerial_params", preset["params"].copy())

    st.markdown("---")

    # Target and feature selection with value filters
    st.markdown("### Column Selection")

    col1, col2 = st.columns(2)

    with col1:
        target_cols, target_values = render_column_value_selector(
            df,
            "Target columns (outcomes to find patterns for)",
            "aerial_target",
            help_text="Select the outcome columns you want to predict/explain"
        )

    with col2:
        # Filter out target columns from feature options
        feature_df = df[[c for c in df.columns if c not in target_cols]] if target_cols else df
        feature_cols, feature_values = render_column_value_selector(
            feature_df,
            "Features of interest (optional)",
            "aerial_features",
            help_text="Leave empty to use all features, or select specific ones to focus on. Only selected features will be used for training."
        )

    # Show summary of selections
    with st.expander("📋 Selection Summary", expanded=False):
        if target_cols:
            st.markdown("**Target Columns:**")
            for col in target_cols:
                vals = target_values.get(col)
                if vals:
                    st.markdown(f"- `{col}`: {', '.join(vals)}")
                else:
                    st.markdown(f"- `{col}`: All values")
        else:
            st.caption("No target columns selected")

        if feature_cols:
            st.markdown("**Features of Interest:**")
            for col in feature_cols:
                vals = feature_values.get(col)
                if vals:
                    st.markdown(f"- `{col}`: {', '.join(vals)}")
                else:
                    st.markdown(f"- `{col}`: All values")
            st.info(f"Note: Only these {len(feature_cols)} feature(s) will be used for model training.")
        else:
            st.caption("No specific features selected (will use all non-target columns)")

    # Advanced settings - use explicit keys so presets can update them
    with st.expander("⚙️ Advanced Settings (optional)"):
        st.caption("Adjust these if you understand the parameters. The preset values work well for most cases.")

        c1, c2 = st.columns(2)
        with c1:
            params["ant_similarity"] = st.slider(
                "Pattern Frequency",
                0.0, 1.0, params["ant_similarity"],
                key="aerial_ant_similarity",
                help="Lower = more rules, patterns occur in the data less often. Higher = less rules, more frequently occurring patterns."
            )
            params["cons_similarity"] = st.slider(
                "Pattern Strictness",
                0.0, 1.0, params["cons_similarity"],
                key="aerial_cons_similarity",
                help="Higher = stricter, fewer rules. Lower = more permissive but lower association strength."
            )
            params["max_antecedents"] = st.slider(
                "Max Conditions per Rule",
                1, 10, params["max_antecedents"],
                key="aerial_max_antecedents",
                help="Maximum conditions in the IF part (e.g., 2 = 'IF A and B THEN C')"
            )

        with c2:
            params["epochs"] = st.slider(
                "Training Length",
                1, 10, params["epochs"],
                key="aerial_epochs",
                help="More = more thorough but slower, higher number may need to overfitting"
            )
            params["batch_size"] = st.select_slider(
                "Training Batches",
                options=[2, 4, 8, 16, 32, 64, 128],
                value=params["batch_size"],
                key="aerial_batch_size",
                help="Should be on the higher end for datasets of enough rows (>1000), should be smaller for small datasets."
            )

    st.markdown("---")

    if st.button("🚀 Run Aerial+ Analysis", type="primary", width="stretch"):
        run_aerial(df, params, target_cols, target_values, feature_cols if feature_cols else None, feature_values)


def _select_fp_preset(key):
    preset = FPGROWTH_PRESETS[key]
    st.session_state.fp_preset = key
    st.session_state.fp_params = preset["params"].copy()
    st.session_state.fp_min_support = preset["params"]["min_support"]
    st.session_state.fp_min_confidence = preset["params"]["min_confidence"]
    st.session_state.fp_max_items = preset["params"]["max_items"]


def render_fpgrowth_config(df):
    st.subheader("FP-Growth Configuration")

    st.markdown("**Select a preset:**")

    # Initialize preset params in session state if not exists
    if "fp_params" not in st.session_state:
        default_preset = FPGROWTH_PRESETS["balanced"]
        st.session_state.fp_params = default_preset["params"].copy()
        st.session_state.fp_preset = "balanced"

    cols = st.columns(3)
    for i, (key, preset) in enumerate(FPGROWTH_PRESETS.items()):
        with cols[i]:
            selected = st.session_state.get("fp_preset") == key
            st.button(
                preset["name"],
                key=f"fp_{key}",
                width="stretch",
                type="primary" if selected else "secondary",
                on_click=_select_fp_preset,
                args=(key,)
            )

    selected_preset = st.session_state.get("fp_preset", "balanced")
    preset = FPGROWTH_PRESETS[selected_preset]

    st.info(f"**{preset['name']}:** {preset['description']}")

    # Use session state params
    params = st.session_state.get("fp_params", preset["params"].copy())

    st.markdown("---")

    # Target selection with Select All functionality
    st.markdown("### Column Selection")

    col1, col2 = st.columns(2)

    with col1:
        target_cols, target_values = render_column_value_selector(
            df,
            "Filter rules by target columns (optional)",
            "fpgrowth_target",
            help_text="Only show rules where the outcome is one of these columns"
        )

    with col2:
        # Feature selection for FP-Growth (to filter which columns to include)
        feature_df = df[[c for c in df.columns if c not in target_cols]] if target_cols else df
        feature_cols, feature_values = render_column_value_selector(
            feature_df,
            "Features to include (optional)",
            "fpgrowth_features",
            help_text="Leave empty to use all features, or select specific ones to focus on"
        )

    # Show summary of selections
    with st.expander("📋 Selection Summary", expanded=False):
        if target_cols:
            st.markdown("**Target Columns (filter outcomes):**")
            for col in target_cols:
                vals = target_values.get(col)
                if vals:
                    st.markdown(f"- `{col}`: {', '.join(vals)}")
                else:
                    st.markdown(f"- `{col}`: All values")
        else:
            st.caption("No target filtering (all outcomes will be shown)")

        if feature_cols:
            st.markdown("**Features to Include:**")
            st.caption(f"{len(feature_cols)} feature(s) selected")
        else:
            st.caption("No specific features selected (will use all columns)")

    # Advanced settings - use explicit keys so presets can update them
    with st.expander("⚙️ Advanced Settings (optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            params["min_support"] = st.slider(
                "Minimum Support",
                0.001, 0.5, params["min_support"], 0.01,
                format="%.3f",
                key="fp_min_support",
                help="How common a pattern must be (fraction of data). 0.02 = 2%"
            )
        with c2:
            params["min_confidence"] = st.slider(
                "Minimum Confidence",
                0.1, 1.0, params["min_confidence"], 0.05,
                key="fp_min_confidence",
                help="How reliable a rule must be. 0.8 = 80% accuracy"
            )
        with c3:
            params["max_items"] = st.slider(
                "Max Items per Pattern",
                2, 10, params["max_items"],
                key="fp_max_items",
                help="Maximum complexity of patterns"
            )

    st.markdown("---")

    if st.button("🚀 Run FP-Growth Analysis", type="primary", width="stretch"):
        run_fpgrowth(df, params, target_cols, target_values, feature_cols if feature_cols else None, feature_values)


def run_aerial(df, params, target_cols, target_values, features, feature_values):
    progress = st.progress(0, "Initializing...")
    status_placeholder = st.empty()

    try:
        status_placeholder.info("⏳ Preparing data...")
        progress.progress(10, "Preparing data...")

        # Validate and clean data for Aerial
        df_clean = df.copy()

        # If specific features are selected, only use those + target columns
        if features:
            cols_to_keep = list(set(features + (target_cols if target_cols else [])))
            cols_to_keep = [c for c in cols_to_keep if c in df_clean.columns]
            if cols_to_keep:
                df_clean = df_clean[cols_to_keep]
                status_placeholder.info(f"⏳ Using {len(cols_to_keep)} selected columns for training...")

        # Remove columns with only one unique value (causes division by zero in Aerial)
        constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
        if constant_cols:
            st.warning(
                f"Removing {len(constant_cols)} constant columns (only 1 unique value): {', '.join(constant_cols[:5])}{'...' if len(constant_cols) > 5 else ''}")
            df_clean = df_clean.drop(columns=constant_cols)
            # Update target_cols and features if they contained removed columns
            if target_cols:
                target_cols = [c for c in target_cols if c not in constant_cols]
            if features:
                features = [f for f in features if f not in constant_cols]

        # Remove columns with all NA values
        all_na_cols = [col for col in df_clean.columns if df_clean[col].isna().all()]
        if all_na_cols:
            st.warning(f"Removing {len(all_na_cols)} columns with all missing values: {', '.join(all_na_cols)}")
            df_clean = df_clean.drop(columns=all_na_cols)
            if target_cols:
                target_cols = [c for c in target_cols if c not in all_na_cols]
            if features:
                features = [f for f in features if f not in all_na_cols]

        if len(df_clean) == 0:
            status_placeholder.empty()
            st.error("No data remaining after cleaning.")
            return

        if len(df_clean.columns) < 2:
            status_placeholder.empty()
            st.error("Need at least 2 columns for rule mining.")
            return

        # Build target_class parameter for Aerial+
        # PyAerial format: ["feature1", "feature2", {"feature3": "value1"}, ...]
        # Either a feature name as str, or specific value in dict form {feature: value}
        aerial_target_class = None
        if target_cols:
            aerial_target_class = []
            for col in target_cols:
                vals = target_values.get(col)
                if vals:
                    # Add specific {column: value} dicts for each value
                    for val in vals:
                        aerial_target_class.append({col: val})
                else:
                    # Add just column name string for all values
                    aerial_target_class.append(col)

        # Build features_of_interest parameter for Aerial+
        # Same format as target_classes
        aerial_features = None
        if features:
            aerial_features = []
            for col in features:
                vals = feature_values.get(col)
                if vals:
                    # Add specific {column: value} dicts for each value
                    for val in vals:
                        aerial_features.append({col: val})
                else:
                    # Add just column name string for all values
                    aerial_features.append(col)

        miner = AerialMiner(
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            max_antecedents=params["max_antecedents"],
            ant_similarity=params["ant_similarity"],
            cons_similarity=params["cons_similarity"],
            layer_dims=params["layer_dims"],
            target_class=aerial_target_class,
            features_of_interest=aerial_features,
            quality_metrics=["support", "confidence", "zhangs_metric", "interestingness"]
        )

        status_placeholder.info("⏳ Training neural network... (this may take a moment)")
        progress.progress(30, "Training neural network... (this may take a moment)")

        rules, stats = miner.mine_rules(df_clean)

        # Handle case where mining returns no results
        if not rules:
            progress.progress(100, "Done!")
            status_placeholder.empty()
            st.warning(
                "No rules found. Try adjusting parameters: lower Pattern Frequency, lower Pattern Strictness, or more Training Length.")
            return

        status_placeholder.info("⏳ Processing results...")
        progress.progress(90, "Processing results...")

        st.session_state.mining_results = {
            "rules": rules,
            "stats": stats,
            "method": "Aerial+",
            "params": params
        }

        progress.progress(100, "Done!")
        status_placeholder.empty()

        st.session_state.last_analysis_result = {
            "success": True,
            "method": "Aerial+",
            "num_rules": len(rules)
        }

    except Exception as e:
        status_placeholder.empty()
        st.error(f"Error running analysis: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def run_fpgrowth(df, params, target_cols, target_values, features, feature_values):
    progress = st.progress(0, "Initializing...")
    status_placeholder = st.empty()

    try:
        status_placeholder.info("⏳ Preparing data...")
        progress.progress(10, "Preparing data...")

        # Validate and clean data
        df_clean = df.copy()

        # If specific features are selected, only use those + target columns
        if features:
            cols_to_keep = list(set(features + (target_cols if target_cols else [])))
            cols_to_keep = [c for c in cols_to_keep if c in df_clean.columns]
            if cols_to_keep:
                df_clean = df_clean[cols_to_keep]
                status_placeholder.info(f"⏳ Using {len(cols_to_keep)} selected columns...")

        # Remove columns with only one unique value
        constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
        if constant_cols:
            st.warning(
                f"Removing {len(constant_cols)} constant columns: {', '.join(constant_cols[:5])}{'...' if len(constant_cols) > 5 else ''}")
            df_clean = df_clean.drop(columns=constant_cols)
            if target_cols:
                target_cols = [c for c in target_cols if c not in constant_cols]
            if features:
                features = [f for f in features if f not in constant_cols]

        if len(df_clean) == 0:
            status_placeholder.empty()
            st.error("No data remaining after cleaning.")
            return

        miner = MLxtendMiner(
            algorithm="fpgrowth",
            min_support=params["min_support"],
            min_confidence=params["min_confidence"],
            max_items=params["max_items"]
        )

        status_placeholder.info("⏳ Mining frequent patterns...")
        progress.progress(30, "Mining frequent patterns...")

        rules, stats = miner.mine_rules(df_clean)

        status_placeholder.info("⏳ Filtering rules...")
        progress.progress(70, "Filtering rules...")

        # Filter by target columns
        if target_cols:
            filtered_rules = []
            for r in rules:
                cons_feature = r.get("consequent", {}).get("feature")
                cons_value = r.get("consequent", {}).get("value")

                if cons_feature in target_cols:
                    # Check if specific values are required for this target
                    target_vals = target_values.get(cons_feature)
                    if target_vals is None or str(cons_value) in target_vals:
                        filtered_rules.append(r)
            rules = filtered_rules

        status_placeholder.info("⏳ Processing results...")
        progress.progress(90, "Processing results...")

        st.session_state.mining_results = {
            "rules": rules,
            "stats": stats,
            "method": "FP-Growth",
            "params": params
        }

        progress.progress(100, "Done!")
        status_placeholder.empty()

        st.session_state.last_analysis_result = {
            "success": True,
            "method": "FP-Growth",
            "num_rules": len(rules)
        }

    except Exception as e:
        status_placeholder.empty()
        st.error(f"Error running analysis: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
