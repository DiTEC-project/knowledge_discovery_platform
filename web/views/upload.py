import streamlit as st
import pandas as pd
import numpy as np
import csv
import io
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from views.utils import reset_to_original, revert_to_version


def detect_delimiter(file_content):
    """Auto-detect CSV delimiter using csv.Sniffer"""
    try:
        sample = file_content[:8192]
        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
        return dialect.delimiter
    except:
        return ','


def infer_column_type(series, total_rows):
    """
    Infer if a column should be treated as categorical or numerical.
    Rules:
    - If values contain commas (like 1,000), treat as numerical
    - If unique values < 10
    """
    n_unique = series.nunique()
    is_numeric_dtype = pd.api.types.is_numeric_dtype(series)

    if not is_numeric_dtype:
        # Check if string values look like comma-separated numbers (e.g., "1,000")
        sample = series.dropna().head(100).astype(str)
        comma_number_pattern = sample.str.match(r'^-?[\d,]+\.?\d*$')
        if comma_number_pattern.sum() > len(sample) * 0.8:  # 80% match
            return "numerical"
        return "categorical"

    # For numeric types: < 10 unique OR < 1% of total rows → categorical
    if n_unique < 10:
        return "categorical"

    return "numerical"


def render():
    st.header("📁 Upload Data")

    st.markdown("""
    Upload your tabular data file to begin the analysis.
    Supported formats: **CSV**, **Excel** (.xlsx, .xls)
    """)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Upload a tabular dataset with column headers in the first row."
    )

    if uploaded_file is not None:
        is_csv = uploaded_file.name.lower().endswith(".csv")

        if is_csv:
            file_content = uploaded_file.read().decode('utf-8', errors='replace')
            uploaded_file.seek(0)

            detected_delimiter = detect_delimiter(file_content)
            delimiter_names = {',': 'Comma (,)', ';': 'Semicolon (;)', '\t': 'Tab', '|': 'Pipe (|)'}

            st.success(f"Detected delimiter: **{delimiter_names.get(detected_delimiter, detected_delimiter)}**")

            with st.expander("CSV Options (change if detection is wrong)"):
                col1, col2 = st.columns(2)
                with col1:
                    delim_options = [',', ';', '\t', '|']
                    default_idx = delim_options.index(detected_delimiter) if detected_delimiter in delim_options else 0
                    delimiter = st.selectbox(
                        "Delimiter",
                        delim_options,
                        index=default_idx,
                        format_func=lambda x: delimiter_names.get(x, x)
                    )
                with col2:
                    on_bad_lines = st.selectbox(
                        "Rows with wrong column count",
                        ["skip", "warn", "error"],
                        index=0,
                        help="'Skip' will ignore problematic rows"
                    )
        else:
            delimiter = ','
            on_bad_lines = 'skip'

        if st.button("📥 Load File", type="primary"):
            status_placeholder = st.empty()
            try:
                with status_placeholder:
                    st.info("⏳ Loading file...")

                uploaded_file.seek(0)
                if is_csv:
                    df = pd.read_csv(
                        uploaded_file,
                        delimiter=delimiter,
                        on_bad_lines=on_bad_lines,
                        engine="python"
                    )
                else:
                    df = pd.read_excel(uploaded_file)

                with status_placeholder:
                    st.info("⏳ Analyzing column types...")

                # Infer column types
                column_types = {}
                for col in df.columns:
                    column_types[col] = infer_column_type(df[col], len(df))

                st.session_state.data = df
                st.session_state.current_data = df.copy()
                st.session_state.filename = uploaded_file.name
                st.session_state.processing_history = []
                st.session_state.column_types = column_types
                st.session_state.data_versions = [{"name": "v0: Original", "data": df.copy(), "column_types": column_types.copy()}]
                st.session_state.current_version_idx = 0
                st.session_state.sidebar_version_select = 0
                st.session_state.mining_results = None

                status_placeholder.empty()
                st.success(f"✓ Loaded **{uploaded_file.name}**: {len(df)} rows × {len(df.columns)} columns")
                st.rerun()

            except Exception as e:
                status_placeholder.empty()
                st.error(f"Error loading file: {str(e)}")

    # Show data if loaded
    if st.session_state.current_data is not None:
        df = st.session_state.current_data

        st.markdown("---")
        st.subheader("Data Preview")

        # Show version selector if there are multiple versions
        data_versions = st.session_state.get("data_versions", [])
        if len(data_versions) > 1:
            version_names = [v["name"] for v in data_versions]
            version_idx = st.selectbox(
                "View data version",
                range(len(version_names)),
                index=len(version_names) - 1,
                format_func=lambda i: version_names[i],
                help="Select a previous version to view"
            )
            display_df = data_versions[version_idx]["data"]
        else:
            display_df = df

        n_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=1)
        st.dataframe(display_df.head(n_rows), width="stretch")

        # Column information with type override
        st.subheader("Column Information")

        with st.expander("ℹ️ About column type detection"):
            st.markdown("""
            **Automatic detection rule:**
            - A numeric column is treated as **Categorical** if it has < 10 unique values OR has comma seperated (float) values
            - Otherwise, numeric columns are treated as **Numerical**
            - Non-numeric columns are always **Categorical**

            You can manually change the type below if the automatic detection is wrong.
            """)

        column_types = st.session_state.get("column_types", {})
        if not column_types:
            column_types = {col: infer_column_type(df[col], len(df)) for col in df.columns}
            st.session_state.column_types = column_types

        col_info = []
        for col in df.columns:
            col_type = column_types.get(col, "categorical")
            col_info.append({
                "Column": col,
                "Type": col_type.capitalize(),
                "Missing": f"{df[col].isna().sum()} ({df[col].isna().mean() * 100:.1f}%)",
                "Unique Values": df[col].nunique(),
            })

        st.dataframe(pd.DataFrame(col_info), width="stretch", hide_index=True)

        # Allow type override
        with st.expander("🔧 Change column types"):
            st.caption("Select columns to change their detected type")

            cols_to_change = st.multiselect("Select columns", df.columns.tolist())

            if cols_to_change:
                new_type = st.radio("Change selected columns to:", ["Numerical", "Categorical"], horizontal=True)

                if st.button("Apply Type Change"):
                    for col in cols_to_change:
                        column_types[col] = new_type.lower()
                    st.session_state.column_types = column_types
                    st.success(f"Changed {len(cols_to_change)} column(s) to {new_type}")
                    st.rerun()

        # Quick summary
        st.subheader("Quick Summary")

        num_cols = [c for c, t in column_types.items() if t == "numerical"]
        cat_cols = [c for c, t in column_types.items() if t == "categorical"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Numerical Columns", len(num_cols))
        with col4:
            missing = df.isna().sum().sum()
            st.metric("Missing Values", missing)

        # Navigation guidance
        st.markdown("---")
        st.subheader("Next Steps")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Recommended workflow:**
            1. **Explore** your data to understand distributions
            2. **Preprocess** (handle missing values, outliers, discretize)
            3. **Analyze** to discover association rules
            4. **View Results** and export
            """)
        with col2:
            if st.button("➡️ Continue to Explore Data", type="primary", width="stretch"):
                st.session_state.current_page = "explore"
                st.rerun()

            if st.button("⏭️ Skip to Preprocess", width="stretch"):
                st.session_state.current_page = "preprocess"
                st.rerun()

        # Reset option
        if len(data_versions) > 1:
            st.markdown("---")
            st.subheader("Data Version Management")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset to Original Data", type="secondary"):
                    reset_to_original()
                    st.rerun()

            with col2:
                if len(data_versions) > 1:
                    version_names = [v["name"] for v in data_versions[:-1]]
                    selected_idx = st.selectbox(
                        "Revert to version",
                        range(len(version_names)),
                        format_func=lambda i: version_names[i],
                        key="revert_select"
                    )
                    if st.button("⏪ Revert"):
                        revert_to_version(selected_idx)
                        st.success(f"Reverted to: {version_names[selected_idx]}")
                        st.rerun()
