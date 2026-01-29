import streamlit as st

st.set_page_config(
    page_title="Knowledge Discovery Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "data" not in st.session_state:
    st.session_state.data = None
if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "processing_history" not in st.session_state:
    st.session_state.processing_history = []
if "mining_results" not in st.session_state:
    st.session_state.mining_results = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"


def navigate_to(page):
    st.session_state.current_page = page
    if "last_preprocess_result" in st.session_state:
        st.session_state.last_preprocess_result = None
    st.rerun()


# Sidebar
with st.sidebar:
    st.markdown("## 🔬 Knowledge Discovery")

    # Navigation with status indicators
    pages = [
        ("upload", "Upload Data", "📁"),
        ("view_data", "View Data", "👁️"),
        ("explore", "Explore Data", "📊"),
        ("preprocess", "Preprocess", "⚙️"),
        ("analyze", "Analyze", "🔍"),
        ("results", "Results", "📋"),
    ]

    data_loaded = st.session_state.data is not None

    for page_id, label, icon in pages:
        is_current = st.session_state.current_page == page_id
        is_disabled = not data_loaded and page_id != "upload"

        if st.button(
                f"{icon} {label}",
                key=f"nav_{page_id}",
                width="stretch",
                type="primary" if is_current else "secondary",
                disabled=is_disabled
        ):
            navigate_to(page_id)

    # Status summary and version selector
    if st.session_state.data is not None:
        st.markdown("---")
        # File info with truncation and hover for full name
        full_filename = st.session_state.get('filename', 'Unknown')
        if len(full_filename) > 30:
            display_filename = full_filename[:27] + "..."
            st.markdown(f'<span title="{full_filename}" style="font-size: 0.875rem; color: rgba(49, 51, 63, 0.6);">📄 <b>{display_filename}</b></span>', unsafe_allow_html=True)
        else:
            st.caption(f"📄 **{full_filename}**")
        df = st.session_state.current_data
        st.caption(f"{len(df)} rows × {len(df.columns)} cols")

        # Data version selector (only if multiple versions)
        data_versions = st.session_state.get("data_versions", [])
        if len(data_versions) > 1:
            version_names = [v["name"] for v in data_versions]

            if "current_version_idx" not in st.session_state:
                st.session_state.current_version_idx = len(version_names) - 1
            if st.session_state.current_version_idx >= len(version_names):
                st.session_state.current_version_idx = len(version_names) - 1

            short_names = [n[:22] + "..." if len(n) > 25 else n for n in version_names]

            # Sync widget to current_version_idx (handles programmatic version changes)
            widget_val = st.session_state.get("sidebar_version_select")
            expected_idx = st.session_state.current_version_idx
            if widget_val is not None and widget_val != expected_idx:
                # Widget is out of sync - update it to match current_version_idx
                st.session_state.sidebar_version_select = expected_idx

            def on_sidebar_version_change():
                """Called only when user manually changes the version selector."""
                idx = st.session_state.sidebar_version_select
                if idx < len(st.session_state.data_versions):
                    st.session_state.current_version_idx = idx
                    st.session_state.current_data = st.session_state.data_versions[idx]["data"].copy()
                    if "column_types" in st.session_state.data_versions[idx]:
                        st.session_state.column_types = st.session_state.data_versions[idx]["column_types"].copy()

            st.selectbox(
                "Version",
                range(len(version_names)),
                index=st.session_state.current_version_idx,
                format_func=lambda i: short_names[i],
                key="sidebar_version_select",
                on_change=on_sidebar_version_change
            )

        # Compact status line
        column_types = st.session_state.get("column_types", {})
        num_cols = sum(1 for t in column_types.values() if t == "numerical")
        cat_cols = sum(1 for t in column_types.values() if t == "categorical")
        steps = len(st.session_state.processing_history)
        rules = len(st.session_state.mining_results.get('rules', [])) if st.session_state.mining_results else 0

        status_parts = []
        if num_cols > 0:
            status_parts.append(f"{num_cols}N/{cat_cols}C")
        if steps > 0:
            status_parts.append(f"{steps} steps")
        if rules > 0:
            status_parts.append(f"{rules} rules")
        if status_parts:
            st.caption(" | ".join(status_parts))
    else:
        st.markdown("---")
        st.info("Upload data to begin")

# Main content - lazy import views for faster startup
page = st.session_state.current_page
if page == "upload":
    from views import upload
    upload.render()
elif page == "view_data":
    from views import view_data
    view_data.render()
elif page == "explore":
    from views import explore
    explore.render()
elif page == "preprocess":
    from views import preprocess
    preprocess.render()
elif page == "analyze":
    from views import analyze
    analyze.render()
elif page == "results":
    from views import results
    results.render()
