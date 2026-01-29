import streamlit as st
import pandas as pd
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def get_column_type(col, df):
    """Get column type from session state or infer it."""
    column_types = st.session_state.get("column_types", {})
    if col in column_types:
        return column_types[col]
    if pd.api.types.is_numeric_dtype(df[col]):
        n_unique = df[col].nunique()
        if n_unique < 10 or n_unique < (len(df) * 0.01):
            return "categorical"
        return "numerical"
    return "categorical"


def to_numeric_safe(series):
    """Convert series to numeric, handling comma-separated numbers."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = series.astype(str).str.replace(',', '').str.strip()
    return pd.to_numeric(converted, errors='coerce')


def get_numerical_cols(df):
    """Get list of numerical columns from DataFrame."""
    return [c for c in df.columns if get_column_type(c, df) == "numerical"]


def get_categorical_cols(df):
    """Get list of categorical columns from DataFrame."""
    return [c for c in df.columns if get_column_type(c, df) == "categorical"]


def format_version_name(step_name, details=None):
    """Create a version name with step number for uniqueness."""
    n_versions = len(st.session_state.get("data_versions", []))
    if details:
        return f"v{n_versions}: {step_name} ({details})"
    return f"v{n_versions}: {step_name}"


def save_version(name, df, column_types=None, details=None):
    """Save a new version to version history with unique naming."""
    if column_types is None:
        column_types = st.session_state.get("column_types", {}).copy()
    if "data_versions" not in st.session_state:
        st.session_state.data_versions = [{
            "name": "v0: Original",
            "data": st.session_state.data.copy(),
            "column_types": st.session_state.get("column_types", {}).copy()
        }]

    version_name = format_version_name(name, details)
    st.session_state.data_versions.append({
        "name": version_name,
        "data": df.copy(),
        "column_types": column_types.copy()
    })
    st.session_state.current_version_idx = len(st.session_state.data_versions) - 1


def reset_to_original():
    """Reset data to original version."""
    data_versions = st.session_state.get("data_versions", [])
    st.session_state.current_data = st.session_state.data.copy()
    st.session_state.processing_history = []
    original_version = data_versions[0] if data_versions else {}
    if "column_types" in original_version:
        st.session_state.column_types = original_version["column_types"].copy()
    st.session_state.data_versions = [{
        "name": "v0: Original",
        "data": st.session_state.data.copy(),
        "column_types": st.session_state.get("column_types", {}).copy()
    }]
    st.session_state.current_version_idx = 0


def revert_to_version(idx):
    """Revert to a specific version by index."""
    data_versions = st.session_state.get("data_versions", [])
    if idx < len(data_versions):
        st.session_state.current_data = data_versions[idx]["data"].copy()
        if "column_types" in data_versions[idx]:
            st.session_state.column_types = data_versions[idx]["column_types"].copy()
        st.session_state.data_versions = data_versions[:idx + 1]
        st.session_state.processing_history = st.session_state.processing_history[:idx]
        st.session_state.current_version_idx = idx


def render_version_selector(key, label="Select version", show_info=True):
    """Render a version selector that works correctly with duplicate names.
    Returns (selected_index, selected_version_data) or (None, None) if no versions."""
    data_versions = st.session_state.get("data_versions", [])
    if not data_versions:
        return None, None

    version_names = [v["name"] for v in data_versions]
    current_idx = st.session_state.get("current_version_idx", len(data_versions) - 1)

    selected_idx = st.selectbox(
        label,
        range(len(version_names)),
        index=min(current_idx, len(version_names) - 1),
        format_func=lambda i: version_names[i],
        key=key
    )

    if show_info and selected_idx != len(data_versions) - 1:
        st.info(f"Viewing: **{version_names[selected_idx]}** (not current)")

    return selected_idx, data_versions[selected_idx]