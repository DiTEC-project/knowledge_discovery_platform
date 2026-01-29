import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from views.utils import get_column_type, to_numeric_safe


def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical-categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0 or n == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))

def render():
    st.header("📊 Explore Data")

    if st.session_state.current_data is None:
        st.warning("Please upload data first.")
        if st.button("Go to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return

    df = st.session_state.current_data

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distributions",
        "🔗 Correlations",
        "❓ Missing Data",
        "📈 Feature vs Target",
        "📋 Summary Stats"
    ])

    with tab1:
        render_distributions(df)
    with tab2:
        render_correlations(df)
    with tab3:
        render_missing(df)
    with tab4:
        render_feature_target(df)
    with tab5:
        render_summary(df)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
    with col2:
        if st.button("➡️ Continue to Preprocess", type="primary"):
            st.session_state.current_page = "preprocess"
            st.rerun()

def render_distributions(df):
    st.subheader("Column Distributions")

    col = st.selectbox("Select column to visualize", df.columns.tolist(), key="dist_col")

    if col:
        col_data = df[col].dropna()
        col_type = get_column_type(col, df)

        if col_type == "numerical":
            # Convert to numeric safely
            col_data_num = to_numeric_safe(col_data).dropna()

            col1, col2 = st.columns([3, 1])
            with col2:
                bins = st.slider("Number of bins", 5, 50, 20)

            # Use converted numeric data for histogram
            fig = px.histogram(x=col_data_num, nbins=bins, title=f"Distribution of {col}")
            fig.update_layout(showlegend=False, xaxis_title=col, yaxis_title="Count")
            st.plotly_chart(fig, width="stretch")

            if len(col_data_num) > 0:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Mean", f"{col_data_num.mean():.2f}")
                with c2:
                    st.metric("Median", f"{col_data_num.median():.2f}")
                with c3:
                    st.metric("Std Dev", f"{col_data_num.std():.2f}")
                with c4:
                    st.metric("Range", f"{col_data_num.min():.2f} - {col_data_num.max():.2f}")
            else:
                st.warning("Could not convert column values to numbers.")
        else:
            value_counts = col_data.value_counts().head(30)
            fig = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f"Distribution of {col}",
                labels={"x": col, "y": "Count"}
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Showing top 30 of {len(col_data.unique())} unique values")

def render_correlations(df):
    st.subheader("Correlation Analysis")

    # Get column types
    num_cols = [c for c in df.columns if get_column_type(c, df) == "numerical"]
    cat_cols = [c for c in df.columns if get_column_type(c, df) == "categorical"]

    corr_type = st.radio(
        "Correlation type",
        ["Numerical (Pearson/Spearman)", "Categorical (Cramér's V)"],
        horizontal=True
    )

    if "Numerical" in corr_type:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numerical columns for correlation analysis.")
            return

        with st.expander("ℹ️ What is correlation?"):
            st.markdown("""
            **Correlation** measures the linear relationship between two numerical variables.

            - **+1**: Perfect positive correlation (both increase together)
            - **0**: No linear relationship
            - **-1**: Perfect negative correlation (one increases, other decreases)

            **Pearson**: Assumes linear relationship, sensitive to outliers
            **Spearman**: Based on ranks, works for monotonic relationships
            """)

        if len(num_cols) > 15:
            selected_cols = st.multiselect(
                "Select columns (many columns = hard to read)",
                num_cols,
                default=num_cols[:10]
            )
        else:
            selected_cols = num_cols

        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)

        # Show status while calculating
        status_placeholder = st.empty()
        chart_placeholder = st.empty()

        with status_placeholder:
            st.info("⏳ Calculating correlations...")

        # Convert to numeric, handling comma-separated numbers and other formats
        num_df = df[selected_cols].copy() if selected_cols else df[num_cols].copy()
        for col in num_df.columns:
            if not pd.api.types.is_numeric_dtype(num_df[col]):
                num_df[col] = num_df[col].astype(str).str.replace(',', '').str.strip()
                num_df[col] = pd.to_numeric(num_df[col], errors='coerce')

        # Drop columns that couldn't be converted
        num_df = num_df.select_dtypes(include=[np.number])
        if len(num_df.columns) < 2:
            status_placeholder.empty()
            st.warning("Not enough valid numerical columns for correlation analysis.")
            return

        corr = num_df.corr(method=method)

        # Clear status and show chart
        status_placeholder.empty()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"{method.capitalize()} Correlation (Numerical Columns Only)"
        )
        fig.update_layout(height=500)
        with chart_placeholder:
            st.plotly_chart(fig, width="stretch")

    else:  # Categorical correlation
        if len(cat_cols) < 2:
            st.warning("Need at least 2 categorical columns for this analysis.")
            return

        with st.expander("ℹ️ What is Cramér's V?"):
            st.markdown("""
            **Cramér's V** measures the association between two categorical variables.

            - **0**: No association
            - **1**: Perfect association

            Values above **0.3** indicate moderate association, above **0.5** is strong.
            """)

        selected_cols = st.multiselect(
            "Select columns",
            cat_cols,
            default=cat_cols[:min(8, len(cat_cols))]
        )

        if len(selected_cols) >= 2:
            # Status and chart placeholders
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()

            with status_placeholder:
                st.info(f"⏳ Calculating Cramér's V for {len(selected_cols)} columns...")

            # Calculate with progress
            cramers_matrix = pd.DataFrame(index=selected_cols, columns=selected_cols, dtype=float)
            total_pairs = len(selected_cols) * len(selected_cols)
            current = 0

            for col1 in selected_cols:
                for col2 in selected_cols:
                    if col1 == col2:
                        cramers_matrix.loc[col1, col2] = 1.0
                    else:
                        cramers_matrix.loc[col1, col2] = cramers_v(
                            df[col1].astype(str).fillna("NA"),
                            df[col2].astype(str).fillna("NA")
                        )
                    current += 1

                # Update progress
                progress = current / total_pairs
                with progress_placeholder:
                    st.progress(progress, text=f"Processing {current}/{total_pairs} pairs...")

            # Clear status
            status_placeholder.empty()
            progress_placeholder.empty()

            fig = px.imshow(
                cramers_matrix.astype(float),
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Blues",
                zmin=0, zmax=1,
                title="Cramér's V (Categorical Association)"
            )
            fig.update_layout(height=500)
            with chart_placeholder:
                st.plotly_chart(fig, width="stretch")

def render_missing(df):
    st.subheader("Missing Data Analysis")

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values
    }).sort_values("Missing Count", ascending=False)

    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        st.success("✓ No missing values in the dataset!")
        return

    st.warning(f"Found missing values in {len(missing_df)} columns")

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing %",
        title="Missing Values by Column",
        text="Missing %"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig, width="stretch")

    st.dataframe(missing_df, width="stretch", hide_index=True)

def render_feature_target(df):
    st.subheader("Feature vs Target Relationship")

    with st.expander("ℹ️ How to use this"):
        st.markdown("""
        This helps you see how a feature relates to an outcome (target).

        - **Numerical feature + Categorical target**: Shows distribution of the feature for each target class
        - **Categorical feature + Categorical target**: Shows a heatmap of co-occurrences

        **Note:** Target must be categorical (e.g., class labels). Numerical targets are not supported here.
        """)

    # Get categorical columns only for target
    cat_cols = [c for c in df.columns if get_column_type(c, df) == "categorical"]
    all_cols = df.columns.tolist()

    # Default to label columns if they exist
    label_cols = [c for c in cat_cols if c.startswith("label_")]

    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox("Feature (input)", all_cols, key="ft_feature")
    with col2:
        if not cat_cols:
            st.warning("No categorical columns available for target selection.")
            return

        default_target = label_cols[0] if label_cols else cat_cols[0]
        default_idx = cat_cols.index(default_target) if default_target in cat_cols else 0
        target = st.selectbox(
            "Target (outcome) - categorical only",
            cat_cols,
            index=default_idx,
            key="ft_target",
            help="Only categorical columns can be selected as target"
        )

    if feature and target and feature != target:
        feature_type = get_column_type(feature, df)
        target_is_binary = df[target].nunique() <= 2

        if feature_type == "numerical":
            # Convert feature to numeric safely
            plot_df = df[[feature, target]].copy()
            plot_df[feature] = to_numeric_safe(plot_df[feature])
            plot_df = plot_df.dropna(subset=[feature])

            if len(plot_df) == 0:
                st.warning("Could not convert feature values to numbers.")
                return

            if target_is_binary:
                fig = px.violin(
                    plot_df,
                    x=target,
                    y=feature,
                    color=target,
                    box=True,
                    points="outliers",
                    title=f"{feature} distribution by {target}"
                )
                st.plotly_chart(fig, width="stretch")

                summary = plot_df.groupby(target)[feature].agg(['mean', 'median', 'std', 'count'])
                summary.columns = ['Mean', 'Median', 'Std Dev', 'Count']
                st.markdown("**Summary by group:**")
                st.dataframe(summary.round(2), width="stretch")
            else:
                fig = px.box(
                    plot_df,
                    x=target,
                    y=feature,
                    color=target,
                    title=f"{feature} by {target}"
                )
                st.plotly_chart(fig, width="stretch")
        else:
            # Both categorical - use heatmap
            ct = pd.crosstab(df[feature], df[target], normalize='index') * 100
            fig = px.imshow(
                ct,
                text_auto=".1f",
                title=f"{feature} vs {target} (row percentages)",
                labels=dict(color="% of row"),
                aspect="auto"
            )
            st.plotly_chart(fig, width="stretch")

            st.caption("Each row shows what percentage of that category falls into each target class")

def render_summary(df):
    st.subheader("Summary Statistics")

    with st.expander("ℹ️ Understanding these statistics"):
        st.markdown("""
        **For Numerical Columns:**
        - **Count**: Number of non-missing values
        - **Mean**: Average value
        - **Std**: Standard deviation (spread of values)
        - **Min**: Smallest value
        - **25% (Q1)**: First quartile - 25% of values are below this
        - **50% (Median)**: Middle value - 50% of values are below this
        - **75% (Q3)**: Third quartile - 75% of values are below this
        - **Max**: Largest value

        **For Categorical Columns:**
        - **Unique Values**: Number of different values in the column
        - **Most Frequent Value**: The value that appears most often
        - **Most Frequent Count**: How many times the most frequent value appears
        - **Missing**: Number of empty/null values
        """)

    # Use column types from session state
    num_cols = [c for c in df.columns if get_column_type(c, df) == "numerical"]
    cat_cols = [c for c in df.columns if get_column_type(c, df) == "categorical"]

    if num_cols:
        st.markdown("**Numerical Columns**")
        # Convert to numeric safely
        num_df = df[num_cols].copy()
        for col in num_df.columns:
            num_df[col] = to_numeric_safe(num_df[col])
        num_df = num_df.select_dtypes(include=[np.number])

        if len(num_df.columns) > 0:
            desc = num_df.describe().T
            desc.columns = ['Count', 'Mean', 'Std', 'Min', '25% (Q1)', '50% (Median)', '75% (Q3)', 'Max']
            st.dataframe(desc.round(2), width="stretch")
        else:
            st.warning("Could not convert numerical columns to numbers.")

    if cat_cols:
        st.markdown("**Categorical Columns**")
        cat_summary = []
        for col in cat_cols:
            vc = df[col].value_counts()
            cat_summary.append({
                "Column": col,
                "Unique Values": df[col].nunique(),
                "Most Frequent Value": str(vc.index[0]) if len(vc) > 0 else "-",
                "Most Frequent Count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "Missing": int(df[col].isna().sum())
            })
        st.dataframe(pd.DataFrame(cat_summary), width="stretch", hide_index=True)
