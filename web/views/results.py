import streamlit as st
import pandas as pd
import numpy as np
import io

METRIC_INFO = {
    "support": {
        "name": "Support",
        "description": "The fraction of data rows where this rule's pattern appears.",
        "interpretation": "A support of 0.10 means the pattern appears in 10% of all rows.",
        "good_value": "> 1% (0.01) is usually meaningful",
        "range": "0 to 1"
    },
    "confidence": {
        "name": "Confidence",
        "description": "When the IF conditions are true, how often is the THEN outcome also true?",
        "interpretation": "A confidence of 0.85 means the rule is correct 85% of the time when conditions are met.",
        "good_value": "> 70% (0.70) is usually considered reliable",
        "range": "0 to 1"
    },
    "zhangs_metric": {
        "name": "Association Strength (Zhang's Metric)",
        "description": "Measures whether the association is stronger than what we'd expect by random chance. Accounts for how common the outcome already is.",
        "interpretation": "Positive = positive association (conditions increase likelihood). Negative = negative association. Near 0 = no real association beyond chance.",
        "good_value": "> 0.3 or < -0.3 indicates meaningful association",
        "range": "-1 to 1"
    },
    "interestingness": {
        "name": "Interestingness",
        "description": "A combined measure of how common AND reliable the pattern is (support × confidence).",
        "interpretation": "Higher values indicate patterns that are both frequent and reliable—the most noteworthy findings.",
        "good_value": "> 0.05 is often considered interesting",
        "range": "0 to 1"
    },
    "lift": {
        "name": "Lift",
        "description": "How much more likely is the outcome when conditions are met, compared to random chance?",
        "interpretation": "Lift of 2.0 means the outcome is twice as likely when the conditions are met. Lift of 1.0 means no effect.",
        "good_value": "> 1.5 indicates meaningful increase in likelihood",
        "range": "0 to infinity (1 = no effect)"
    }
}

def render():
    st.header("📋 Analysis Results")

    if st.session_state.mining_results is None:
        st.warning("No results yet. Run an analysis first.")
        if st.button("Go to Analyze", type="primary"):
            st.session_state.current_page = "analyze"
            st.rerun()
        return

    results = st.session_state.mining_results
    rules = results.get("rules", [])
    method = results.get("method", "Unknown")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rules Found", len(rules))
    with col2:
        st.metric("Method Used", method)
    with col3:
        if rules:
            avg_conf = np.mean([r.get("confidence", 0) for r in rules])
            st.metric("Average Confidence", f"{avg_conf:.1%}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Rules Table", "📊 Summary Statistics", "💾 Export"])

    with tab1:
        render_rules_table(rules)
    with tab2:
        render_summary(rules)
    with tab3:
        render_export(rules)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Analyze"):
            st.session_state.current_page = "analyze"
            st.rerun()
    with col2:
        if st.button("🔄 Run New Analysis"):
            st.session_state.mining_results = None
            st.session_state.current_page = "analyze"
            st.rerun()

def format_rule(rule):
    """Format a rule for display"""
    ants = rule.get("antecedents", [])
    cons = rule.get("consequent", {})

    if isinstance(ants, list):
        ant_parts = []
        for a in ants:
            feat = a.get("feature", a.get("name", "?"))
            val = a.get("value", "?")
            ant_parts.append(f"{feat} = {val}")
        ant_str = " AND ".join(ant_parts)
    else:
        ant_str = str(ants)

    if isinstance(cons, dict):
        feat = cons.get("feature", cons.get("name", "?"))
        val = cons.get("value", "?")
        cons_str = f"{feat} = {val}"
    else:
        cons_str = str(cons)

    return ant_str, cons_str

def render_rules_table(rules):
    if not rules:
        st.info("No rules found with the current settings. Try adjusting the analysis parameters.")
        return

    # Metric explanations
    with st.expander("ℹ️ Understanding the metrics"):
        for key, info in METRIC_INFO.items():
            st.markdown(f"**{info['name']}**")
            st.markdown(f"- {info['description']}")
            st.markdown(f"- *Interpretation:* {info['interpretation']}")
            st.markdown(f"- *Good value:* {info['good_value']}")
            st.markdown("")

    st.subheader("Filter Rules")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_support = st.slider("Min Support", 0.0, 0.5, 0.0, 0.01, format="%.2f",
                               help="Only show rules appearing in at least this fraction of data")
    with col2:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, format="%.2f",
                                  help="Only show rules with at least this reliability")
    with col3:
        min_zhangs = st.slider("Min Assoc. Strength", -1.0, 1.0, -1.0, 0.1, format="%.1f",
                              help="Only show rules with at least this association strength")
    with col4:
        search = st.text_input("Search in rules", help="Filter rules containing this text")

    # Sorting
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["confidence", "support", "zhangs_metric", "interestingness"],
            format_func=lambda x: METRIC_INFO.get(x, {}).get("name", x)
        )
    with col2:
        sort_desc = st.checkbox("Descending", value=True)

    # Create placeholders for status and results
    status_placeholder = st.empty()
    count_placeholder = st.empty()
    table_placeholder = st.empty()

    # Show loading indicator while filtering
    with status_placeholder:
        st.info("⏳ Filtering and sorting rules...")

    # Apply filters
    filtered = rules.copy()
    if min_support > 0:
        filtered = [r for r in filtered if r.get("support", 0) >= min_support]
    if min_confidence > 0:
        filtered = [r for r in filtered if r.get("confidence", 0) >= min_confidence]
    if min_zhangs > -1:
        filtered = [r for r in filtered if (r.get("zhangs_metric") or -1) >= min_zhangs]
    if search:
        search_lower = search.lower()
        filtered = [r for r in filtered if search_lower in str(format_rule(r)).lower()]

    # Sort
    filtered = sorted(filtered, key=lambda x: x.get(sort_by, 0) or 0, reverse=sort_desc)

    # Clear loading status
    status_placeholder.empty()

    # Show count
    with count_placeholder:
        st.caption(f"Showing **{len(filtered)}** of {len(rules)} rules")

    # Build table - multiply by 100 for percentage display while keeping numeric for sorting
    rows = []
    for r in filtered:
        ant, cons = format_rule(r)
        support = r.get("support")
        confidence = r.get("confidence")
        interest = r.get("interestingness")
        rows.append({
            "IF (conditions)": ant,
            "THEN (outcome)": cons,
            "Support (%)": support * 100 if support is not None else None,
            "Confidence (%)": confidence * 100 if confidence is not None else None,
            "Assoc. Strength": r.get("zhangs_metric"),
            "Interest (%)": interest * 100 if interest is not None else None
        })

    if rows:
        df = pd.DataFrame(rows)

        # Keep numeric values as numbers for proper sorting, use column_config for formatting
        with table_placeholder:
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                height=500,
                column_config={
                    "IF (conditions)": st.column_config.TextColumn(width="large"),
                    "THEN (outcome)": st.column_config.TextColumn(width="medium"),
                    "Support (%)": st.column_config.NumberColumn(
                        format="%.1f",
                        help="Fraction of data where this pattern appears (%)"
                    ),
                    "Confidence (%)": st.column_config.NumberColumn(
                        format="%.1f",
                        help="How often the rule is correct when conditions are met (%)"
                    ),
                    "Assoc. Strength": st.column_config.NumberColumn(
                        format="%+.2f",
                        help="Association strength (Zhang's metric, -1 to +1)"
                    ),
                    "Interest (%)": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Combined measure of frequency and reliability (%)"
                    ),
                }
            )

def render_summary(rules):
    if not rules:
        st.info("No rules to summarize.")
        return

    st.subheader("Results Overview")

    # Statistics
    supports = [r.get("support", 0) for r in rules]
    confidences = [r.get("confidence", 0) for r in rules]
    zhangs = [r.get("zhangs_metric", 0) for r in rules if r.get("zhangs_metric") is not None]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Support Distribution**")
        st.caption(f"Min: {min(supports):.3f}")
        st.caption(f"Max: {max(supports):.3f}")
        st.caption(f"Average: {np.mean(supports):.3f}")
        st.caption(f"Median: {np.median(supports):.3f}")

    with col2:
        st.markdown("**Confidence Distribution**")
        st.caption(f"Min: {min(confidences):.3f}")
        st.caption(f"Max: {max(confidences):.3f}")
        st.caption(f"Average: {np.mean(confidences):.3f}")
        st.caption(f"Median: {np.median(confidences):.3f}")

    with col3:
        if zhangs:
            st.markdown("**Association Strength**")
            st.caption(f"Min: {min(zhangs):.3f}")
            st.caption(f"Max: {max(zhangs):.3f}")
            st.caption(f"Average: {np.mean(zhangs):.3f}")

    st.markdown("---")

    # Outcome distribution
    st.subheader("Rules by Outcome")

    consequents = {}
    for r in rules:
        cons = r.get("consequent", {})
        key = f"{cons.get('feature', '?')} = {cons.get('value', '?')}"
        consequents[key] = consequents.get(key, 0) + 1

    cons_df = pd.DataFrame([
        {"Outcome": k, "Number of Rules": v}
        for k, v in sorted(consequents.items(), key=lambda x: -x[1])
    ])

    if not cons_df.empty:
        st.dataframe(cons_df, width="stretch", hide_index=True)

    # Common conditions
    st.subheader("Most Common Conditions")

    antecedents = {}
    for r in rules:
        for a in r.get("antecedents", []):
            key = f"{a.get('feature', '?')} = {a.get('value', '?')}"
            antecedents[key] = antecedents.get(key, 0) + 1

    ant_df = pd.DataFrame([
        {"Condition": k, "Appears in # Rules": v}
        for k, v in sorted(antecedents.items(), key=lambda x: -x[1])[:15]
    ])

    if not ant_df.empty:
        st.dataframe(ant_df, width="stretch", hide_index=True)

def render_export(rules):
    st.subheader("Export Results")

    if not rules:
        st.info("No rules to export.")
        return

    # Prepare data
    rows = []
    for r in rules:
        ant, cons = format_rule(r)
        rows.append({
            "Antecedent (IF)": ant,
            "Consequent (THEN)": cons,
            "Support": r.get("support"),
            "Confidence": r.get("confidence"),
            "Association Strength": r.get("zhangs_metric"),
            "Interestingness": r.get("interestingness"),
            "Lift": r.get("lift")
        })

    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            csv,
            "association_rules.csv",
            "text/csv",
            width="stretch"
        )

    with col2:
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        st.download_button(
            "📥 Download as Excel",
            buffer.getvalue(),
            "association_rules.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

    st.markdown("---")
    st.subheader("Preview (first 20 rules)")
    st.dataframe(df.head(20), width="stretch", hide_index=True)
