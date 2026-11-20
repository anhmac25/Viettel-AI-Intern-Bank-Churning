"""Streamlit demo: a retention tool for bank staff.

Run locally:        streamlit run streamlit_app.py
Use the API instead of in-process scoring:   API_URL=http://localhost:8000 streamlit run streamlit_app.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from churnlib import config as C

st.set_page_config(page_title="Churn & Segmentation - Digital Bank", page_icon="🏦", layout="wide")

RISK_COLOR = {"High": "#E4572E", "Medium": "#F2A541", "Low": "#3BA55D"}
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.badge {display:inline-block; padding:.25rem .8rem; border-radius:999px; color:white; font-weight:700; font-size:1.05rem;}
.card {border:1px solid #e3e6ea; border-radius:12px; padding:1rem 1.2rem; background:#fafbfc;}
.small {color:#6b7280; font-size:.85rem;}
</style>""", unsafe_allow_html=True)

API_URL = os.environ.get("API_URL", "").rstrip("/")


# ------------------------------------------------------------------ backend (in-process or HTTP API)
@st.cache_resource(show_spinner="Loading model...")
def get_service():
    from churnlib.service import ChurnService
    return ChurnService()


def record_to_payload(rec: dict) -> dict:
    p = {k: rec[k] for k in C.PROFILE_COLS + ["login_count_30d", "days_since_last_login", "days_since_last_txn", "complaints_6m",
                                               "atm_share", "mobile_share", "branch_share", "preferred_channel"]}
    p["txn_counts"] = [int(rec[f"txn_count_m{i}"]) for i in range(1, 7)]
    p["txn_amounts"] = [float(rec[f"txn_amount_m{i}"]) for i in range(1, 7)]
    for k in ("credit_card", "active_member", "credit_score", "age", "tenure", "products_number"):
        p[k] = int(p[k])
    return p


def do_predict(rec: dict) -> dict:
    if API_URL:
        import requests
        r = requests.post(f"{API_URL}/predict", json=record_to_payload(rec), timeout=30)
        r.raise_for_status()
        return r.json()
    return get_service().score_record(rec)


def do_lookup(cid: int):
    if API_URL:
        import requests
        r = requests.get(f"{API_URL}/customer/{cid}", timeout=30)
        return None if r.status_code == 404 else r.json()
    return get_service().lookup(cid)


@st.cache_data
def customers():
    return pd.read_csv(C.MERGED_CLEAN)


@st.cache_data
def read_json(path):
    return json.loads(Path(path).read_text())


meta = read_json(C.MODELS / "churn_metadata.json")
personas = read_json(C.MODELS / "personas.json")
THR, T_HIGH = meta["threshold"], meta["band_high_from"]


# ------------------------------------------------------------------ reusable pieces
def gauge(p: float, band: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Churn probability (next quarter)", "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLOR[band], "thickness": 0.35},
               "steps": [{"range": [0, THR * 100], "color": "#dff3e5"}, {"range": [THR * 100, T_HIGH * 100], "color": "#fdeccc"},
                         {"range": [T_HIGH * 100, 100], "color": "#f8d3c9"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def drivers_chart(factors):
    f = factors[::-1]
    fig = go.Figure(go.Bar(
        x=[x["impact_pp"] for x in f], y=[x["factor"] for x in f], orientation="h",
        marker_color=["#E4572E" if x["impact_pp"] > 0 else "#3BA55D" for x in f],
        text=[f"{x['impact_pp']:+.1f} pp" for x in f], textposition="outside",
        customdata=[[x["customer_value"], x["typical_value"]] for x in f],
        hovertemplate="<b>%{y}</b><br>this customer: %{customdata[0]}<br>typical: %{customdata[1]}<extra></extra>"))
    fig.update_layout(height=290, margin=dict(l=10, r=40, t=30, b=30), xaxis_title="contribution to the churn probability (percentage points)",
                      title="What drives this customer's risk", title_font_size=15)
    return fig


def result_panel(res: dict, rec: dict, actual=None):
    band = res["risk_band"]
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.plotly_chart(gauge(res["churn_probability"], band), width="stretch")
        st.markdown(f"<span class='badge' style='background:{RISK_COLOR[band]}'>{band} risk</span> &nbsp; "
                    f"<span class='small'>Low &lt; {THR:.0%} &le; Medium &lt; {T_HIGH:.0%} &le; High</span>", unsafe_allow_html=True)
        if actual is not None:
            st.caption(f"Outcome recorded in the dataset: **{'churned' if actual == 1 else 'stayed'}**")
    with c2:
        st.markdown(f"<div class='card'><b>Segment:</b> {res['persona']}<br><span class='small'>{res['persona_description']}</span></div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"**Recommended action**  \n{res['recommended_action']}")
        m1, m2 = st.columns(2)
        m1.metric("Customer lifetime value (proxy)", f"€{res['clv_eur']:,.0f}")
        m2.metric("Expected value of contacting", f"€{res['expected_value_of_contact_eur']:,.0f}",
                  "worth contacting" if res["contact_recommended"] else "not worth the cost", delta_color="normal" if res["contact_recommended"] else "off")
    st.plotly_chart(drivers_chart(res["top_factors"]), width="stretch")
    with st.expander("How to read this"):
        st.markdown("Each bar is the average of two comparisons: how much lower this customer's risk would be if that group of inputs were replaced by the "
                    "typical customer's values, and how much higher a typical customer's risk would be if they had this customer's values "
                    "(a two-point approximation of Shapley values). Red pushes the risk up, green pushes it down. Hover a bar to compare this customer's "
                    "value with the typical one. The CLV and the expected value use business assumptions that should be replaced by the bank's real figures.")
    # monthly pattern
    cnt = [rec[f"txn_count_m{i}"] for i in range(1, 7)]
    ref = [get_ref(f"txn_count_m{i}") for i in range(1, 7)]
    fig = go.Figure()
    fig.add_scatter(x=list(range(1, 7)), y=cnt, mode="lines+markers", name="this customer", line=dict(color=RISK_COLOR[band], width=3))
    fig.add_scatter(x=list(range(1, 7)), y=ref, mode="lines", name="typical customer", line=dict(color="#9ca3af", dash="dash"))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=35, b=30), title="Monthly transactions (month 6 = most recent)", title_font_size=15, xaxis_title="month", yaxis_title="transactions")
    st.plotly_chart(fig, width="stretch")


@st.cache_data
def ref_profile():
    return read_json(C.MODELS / "reference_profile.json")


def get_ref(k):
    return ref_profile()[k]


# ------------------------------------------------------------------ header + sidebar
st.title("🏦 Customer Churn & Segmentation")
st.caption("Retention tool for bank staff - demo built on a public Kaggle dataset plus synthetic behavioural data (not real customers).")
with st.sidebar:
    st.header("Model")
    st.metric("Algorithm", meta["algorithm"].upper())
    st.metric("Test ROC-AUC", f"{meta['metrics_test_at_business_threshold']['roc_auc']:.3f}")
    st.metric("Recall at decision threshold", f"{meta['metrics_test_at_business_threshold']['recall']:.0%}")
    st.caption(f"Version {meta['model_version']} - trained {meta['trained_at'][:10]}")
    st.caption(f"Scoring mode: {'API (' + API_URL + ')' if API_URL else 'in-process'}")
    st.divider()
    st.caption("Business assumptions")
    st.json(meta["business"], expanded=False)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔎 Customer lookup", "🎛️ What-if simulator", "👥 Segments", "📊 Model", "🩺 Monitoring"])

# ------------------------------------------------------------------ tab 1: lookup
with tab1:
    df = customers()
    if "cid" not in st.session_state:
        st.session_state.cid = int(pd.read_csv(C.TEST_FILE).customer_id.iloc[0])
    b1, b2, b3, b4 = st.columns([2, 1, 1, 1])
    with b4:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random customer", width="stretch"):
            st.session_state.cid = int(df.sample(1).customer_id.iloc[0]); st.rerun()
    with b2:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("🔴 A churner", width="stretch"):
            st.session_state.cid = int(df[df.churn == 1].sample(1).customer_id.iloc[0]); st.rerun()
    with b3:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("🟢 A loyal one", width="stretch"):
            st.session_state.cid = int(df[df.churn == 0].sample(1).customer_id.iloc[0]); st.rerun()
    with b1:
        cid = st.number_input("Customer ID", value=st.session_state.cid, step=1, format="%d")
    res = do_lookup(int(cid))
    if res is None:
        st.warning("Customer not found. Try the buttons for a random customer.")
    else:
        row = df[df.customer_id == int(cid)].iloc[0].to_dict()
        st.subheader(f"Customer {int(cid)} - {row['country']}, {row['age']} years, {row['products_number']} product(s)")
        result_panel(res, row, actual=res.get("actual_churn_label"))
        with st.expander("Customer record"):
            st.dataframe(pd.DataFrame(row, index=["value"]).T.astype(str), width="stretch")

# ------------------------------------------------------------------ tab 2: what-if
with tab2:
    st.markdown("Change a customer's profile and behaviour and watch the risk and the recommended action update.")
    L, R = st.columns(2)
    with L:
        st.markdown("**Profile**")
        a1, a2 = st.columns(2)
        country = a1.selectbox("Country", ["France", "Germany", "Spain"], index=1)
        gender = a2.selectbox("Gender", ["Female", "Male"])
        age = a1.slider("Age", 18, 92, 48)
        tenure = a2.slider("Tenure (years)", 0, 10, 4)
        products = a1.slider("Products held", 1, 4, 2)
        active = a2.selectbox("Active member", ["Yes", "No"])
        balance = a1.number_input("Balance (EUR)", 0, 300000, 90000, step=5000)
        salary = a2.number_input("Salary (EUR/yr)", 0, 250000, 80000, step=5000)
        cscore = a1.slider("Credit score", 350, 850, 650)
        ccard = a2.selectbox("Credit card", ["Yes", "No"])
    with R:
        st.markdown("**Behaviour (last 6 months)**")
        b1_, b2_ = st.columns(2)
        base_tx = b1_.slider("Typical transactions / month", 0, 60, 14)
        change = b2_.slider("Change in the last 3 months (%)", -80, 40, -40, step=5)
        ticket = b1_.slider("Average transaction (EUR)", 10, 300, 70)
        logins = b2_.slider("App logins (30 days)", 0, 60, 4)
        since_tx = b1_.slider("Days since last transaction", 0, 120, 12)
        complaints = b2_.slider("Complaints (6 months)", 0, 6, 1)
        mobile = b1_.slider("Mobile-app share of transactions", 0.0, 1.0, 0.25, step=0.05)
        branch_frac = b2_.slider("Branch vs ATM (rest of the share)", 0.0, 1.0, 0.6, step=0.1)
    ramp = np.array([1, 1, 1, 1 + change / 100 / 3, 1 + 2 * change / 100 / 3, 1 + change / 100])
    counts = np.maximum(np.round(base_tx * ramp), 0).astype(int)
    rest = 1 - mobile
    atm, br = round(rest * (1 - branch_frac), 4), round(rest * branch_frac, 4)
    sh = {"ATM": atm, "Mobile app": mobile, "Branch": br}
    rec = {"credit_score": cscore, "country": country, "gender": gender, "age": age, "tenure": tenure, "balance": float(balance),
           "products_number": products, "credit_card": int(ccard == "Yes"), "active_member": int(active == "Yes"), "estimated_salary": float(salary),
           **{f"txn_count_m{i + 1}": int(counts[i]) for i in range(6)},
           **{f"txn_amount_m{i + 1}": float(round(counts[i] * ticket, 2)) for i in range(6)},
           "login_count_30d": float(logins), "days_since_last_login": float(0 if logins > 20 else min(60, 30 / (logins + 1))),
           "days_since_last_txn": since_tx, "complaints_6m": complaints, "atm_share": atm, "mobile_share": float(mobile),
           "branch_share": br, "preferred_channel": max(sh, key=sh.get)}
    st.divider()
    result_panel(do_predict(rec), rec)

# ------------------------------------------------------------------ tab 3: segments
with tab3:
    st.markdown("Customers are grouped by **behaviour** (K-Means on RFM and engagement features, not using the churn label) and each group is given a persona.")
    order = sorted(personas, key=lambda p: -personas[p]["churn_rate_all"])
    tab = pd.DataFrame({"Persona": order, "Share of customers": [personas[p]["share_of_customers"] for p in order],
                        "Churn rate": [personas[p]["churn_rate_all"] for p in order],
                        "What defines them": [personas[p]["description"] for p in order]})
    st.dataframe(tab, hide_index=True, width="stretch", column_config={
        "Share of customers": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Churn rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    c1, c2 = st.columns(2)
    c1.image(str(C.FIGS / "segment_profiles.png"), caption="Standardised segment profiles", width="stretch")
    c2.image(str(C.FIGS / "segment_churn_share.png"), caption="Churn rate and size per segment", width="stretch")
    st.subheader("Retention playbook")
    rows = [{"Persona": p, "Risk band": b, "Recommended action": personas[p]["actions"][b.lower()]} for p in order for b in ["High", "Medium", "Low"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab 4: model
with tab4:
    m = meta["metrics_test_at_business_threshold"]
    c = st.columns(5)
    c[0].metric("ROC-AUC (test)", f"{m['roc_auc']:.3f}"); c[1].metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c[2].metric("Recall", f"{m['recall']:.0%}"); c[3].metric("Precision", f"{m['precision']:.0%}"); c[4].metric("Threshold", f"{meta['threshold']:.2f}")
    st.caption("Recall and precision are measured at the business threshold chosen with the cost rule: contact a customer when a x (p x CLV - incentive) > contact cost.")
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(C.REPORTS / "model_comparison.csv"), hide_index=True, width="stretch")
    st.image(str(C.FIGS / "model_curves.png"), width="stretch")
    st.image(str(C.FIGS / "model_threshold.png"), width="stretch")
    st.info(f"Behavioural data lifts the test AUC from {meta['auc_profile_only']:.3f} (profile only) to {meta['auc_profile_plus_behaviour']:.3f}. "
            "Because the behavioural layer is synthetic, part of that lift reflects the generator's assumptions.")

# ------------------------------------------------------------------ tab 5: monitoring
with tab5:
    summ = read_json(C.MONITOR / "summary.json")
    st.markdown("Six **simulated** months of incoming customers, each checked against the training data. Triggers: PSI > 0.10 warning, > 0.25 alert; "
                "AUC drop > 0.05 or recall < 85% for two consecutive valid months -> retrain; > 5% missing data -> block.")
    sm = pd.DataFrame(summ["batches"])[["month", "scenario", "auc", "recall", "input_drift", "prediction_drift", "performance", "data_quality", "action"]]

    def colour(v):
        return {"ALERT": "background-color:#f8d3c9", "BLOCK": "background-color:#f8d3c9", "WARNING": "background-color:#fdeccc",
                "OK": "background-color:#dff3e5"}.get(v, "")
    st.dataframe(sm.style.map(colour, subset=["input_drift", "prediction_drift", "performance", "data_quality"]).format({"auc": "{:.3f}", "recall": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.image(str(C.FIGS / "monitor_performance.png"), width="stretch")
    st.image(str(C.FIGS / "monitor_psi_heatmap.png"), width="stretch")
    st.info("Months 4 and 6 show why input-drift monitoring is not enough: inputs look normal (PSI near 0) while performance collapses because customer behaviour changed.")
    pick = st.selectbox("Evidently report", ["Month 1 (stable)", "Month 3 (activity drop)", "Month 4 (concept drift)"])
    f = C.MONITOR / f"evidently_month_{pick.split()[1]}.html"
    if f.exists():
        if hasattr(st, "iframe"):                       # newer Streamlit
            st.iframe(f, height=900)
        else:                                           # older Streamlit versions
            components.html(f.read_text(encoding="utf-8"), height=900, scrolling=True)
