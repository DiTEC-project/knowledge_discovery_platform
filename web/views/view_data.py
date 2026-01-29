import streamlit as st
import pandas as pd
import io
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from views.utils import get_column_type, render_version_selector, reset_to_original, revert_to_version


def render():
    st.header("👁️ View Data")

    if st.session_state.current_data is None:
        st.warning("Please upload data first.")
        if st.button("Go to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
        return

    df = st.session_state.current_data
    data_versions = st.session_state.get("data_versions", [])
    history = st.session_state.processing_history

    # Status bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Preprocessing Steps", len(history))
    with col4:
        st.metric("Data Versions", len(data_versions) if data_versions else 1)

    st.markdown("---")

    # Version selector - use index-based selection to handle duplicate names
    if len(data_versions) > 1:
        version_idx, version_data = render_version_selector(
            key="view_data_version",
            label="Select data version to view"
        )
        display_df = version_data["data"] if version_data else df
    else:
        display_df = df

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📊 Column Info", "🔄 Version History", "📥 Export"])

    with tab1:
        render_data_table(display_df)

    with tab2:
        render_column_info(display_df)

    with tab3:
        render_version_history(data_versions, history)

    with tab4:
        render_export(display_df)

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_page = "upload"
            st.rerun()
    with col2:
        if st.button("📊 Explore Data"):
            st.session_state.current_page = "explore"
            st.rerun()
    with col3:
        if st.button("➡️ Preprocess", type="primary"):
            st.session_state.current_page = "preprocess"
            st.rerun()


def render_data_table(df):
    st.subheader("Data Preview")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search in data", placeholder="Type to filter rows...", key="data_search")
    with col2:
        n_rows = st.selectbox("Rows to display", [10, 25, 50, 100, 500], index=1, key="view_rows")

    # Create placeholders for status and table
    status_placeholder = st.empty()
    count_placeholder = st.empty()
    table_placeholder = st.empty()

    # Filter by search
    if search:
        with status_placeholder:
            st.info("⏳ Searching...")
        mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
        filtered_df = df[mask]
        status_placeholder.empty()
        with count_placeholder:
            st.caption(f"Showing {len(filtered_df)} rows matching '{search}'")
    else:
        filtered_df = df

    with table_placeholder:
        st.dataframe(filtered_df.head(n_rows), width="stretch")

    # Show full stats
    st.caption(f"Total: {len(df)} rows × {len(df.columns)} columns")


def render_column_info(df):
    st.subheader("Column Information")

    col_summary = []
    for col in df.columns:
        col_type = get_column_type(col, df)
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        missing_pct = n_missing / len(df) * 100

        # Sample values
        sample = df[col].dropna().head(3).tolist()
        sample_str = ", ".join([str(v)[:15] for v in sample])

        col_summary.append({
            "Column": col,
            "Type": col_type.capitalize(),
            "Dtype": dtype,
            "Unique": n_unique,
            "Missing": f"{n_missing} ({missing_pct:.1f}%)",
            "Sample": sample_str
        })

    st.dataframe(pd.DataFrame(col_summary), width="stretch", hide_index=True)


def render_version_history(data_versions, history):
    st.subheader("Version History")

    if not history:
        st.info("No preprocessing steps applied yet. This is the original data.")
        return

    # Timeline of changes
    for i, step in enumerate(history):
        with st.container():
            cols = st.columns([1, 5])
            with cols[0]:
                st.markdown(f"### {i + 1}")
            with cols[1]:
                st.markdown(f"**{step['step']}**")
                details = {k: v for k, v in step.items() if k != "step"}
                for k, v in details.items():
                    st.caption(f"{k}: {v}")
        st.divider()

    # Compare versions
    if len(data_versions) > 1:
        st.subheader("Compare Versions")
        col1, col2 = st.columns(2)
        with col1:
            compare_from = st.selectbox("Compare FROM", [v["name"] for v in data_versions], index=0, key="cmp_from")
        with col2:
            compare_to = st.selectbox("Compare TO", [v["name"] for v in data_versions], index=len(data_versions) - 1, key="cmp_to")

        if compare_from != compare_to:
            from_idx = [v["name"] for v in data_versions].index(compare_from)
            to_idx = [v["name"] for v in data_versions].index(compare_to)
            from_df = data_versions[from_idx]["data"]
            to_df = data_versions[to_idx]["data"]

            changes = []
            if len(from_df) != len(to_df):
                changes.append(f"- Rows: {len(from_df)} → {len(to_df)} ({len(to_df) - len(from_df):+d})")
            if len(from_df.columns) != len(to_df.columns):
                changes.append(f"- Columns: {len(from_df.columns)} → {len(to_df.columns)} ({len(to_df.columns) - len(from_df.columns):+d})")

            new_cols = set(to_df.columns) - set(from_df.columns)
            if new_cols:
                changes.append(f"- New columns: {', '.join(new_cols)}")

            removed_cols = set(from_df.columns) - set(to_df.columns)
            if removed_cols:
                changes.append(f"- Removed columns: {', '.join(removed_cols)}")

            if changes:
                st.markdown("**Changes:**")
                for change in changes:
                    st.markdown(change)
            else:
                st.caption("No structural changes (data values may differ)")

    # Revert options
    st.subheader("Revert Data")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reset to Original", type="secondary"):
            reset_to_original()
            st.success("Reset to original data!")
            st.rerun()

    with col2:
        if len(data_versions) > 1:
            version_names = [v["name"] for v in data_versions[:-1]]
            selected_idx = st.selectbox(
                "Revert to version",
                range(len(version_names)),
                format_func=lambda i: version_names[i],
                key="revert_version"
            )
            if st.button("⏪ Revert"):
                revert_to_version(selected_idx)
                st.success(f"Reverted to: {version_names[selected_idx]}")
                st.rerun()


def render_export(df):
    st.subheader("Export Data")

    st.markdown("Download the current data version:")

    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            csv,
            "processed_data.csv",
            "text/csv",
            width="stretch"
        )

    with col2:
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        st.download_button(
            "📥 Download as Excel",
            buffer.getvalue(),
            "processed_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

    st.markdown("---")
    st.markdown("**Preview (first 20 rows):**")
    st.dataframe(df.head(20), width="stretch")
