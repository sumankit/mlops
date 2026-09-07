"""
Streamlit dashboard for the Student Performance and Dropout-Risk Prediction
System (Part 1: descriptive analytics on top of the SQLite warehouse).

Run with:
    streamlit run dashboard/app.py
"""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_URI

st.set_page_config(page_title="Student Performance & Dropout-Risk Dashboard", layout="wide")

engine = create_engine(DB_URI)


@st.cache_data(ttl=60)
def load_data():
    fact = pd.read_sql("SELECT * FROM fact_assessment", engine)
    dim = pd.read_sql("SELECT * FROM dim_student", engine)
    subject_summary = pd.read_sql("SELECT * FROM agg_subject_summary", engine)
    dept_summary = pd.read_sql("SELECT * FROM agg_department_summary", engine)
    df = fact.merge(dim, on="student_id", how="left")
    return df, subject_summary, dept_summary


df, subject_summary, dept_summary = load_data()

# The UCI dataset's `school` code stands in for "department" here (GP / MS
# are the two participating secondary schools); mapped to their real names
# for readability. There's no native semester field either -- G1, G2, G3 are
# the three periodic assessment checkpoints across the academic term, so
# they're treated as "Semester 1 / 2 / Final" for the semester-wise view.
DEPARTMENT_LABELS = {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}
SEMESTER_LABELS = {"G1": "Semester 1", "G2": "Semester 2", "G3": "Semester 3 (Final)"}

df["department"] = df["school"].map(DEPARTMENT_LABELS).fillna(df["school"])

st.title("🎓 Student Performance & Dropout-Risk Dashboard")
st.caption("Source: UCI Student Performance Dataset (Cortez, 2008) — Mathematics & Portuguese subjects")

# --- Sidebar filters (dropdown: click to open, pick one or more) -------
st.sidebar.header("Filters")


def dropdown_filter(label, options):
    """A multi-select dropdown -- collapsed until clicked, then lets you
    tick any number of options (defaults to all selected)."""
    options = list(options)
    return st.sidebar.multiselect(label, options=options, default=options)

subjects = dropdown_filter("Subject", sorted(df["subject"].unique()))
departments = dropdown_filter("Department (School)", sorted(df["department"].unique()))
semesters = dropdown_filter("Semester", list(SEMESTER_LABELS.values()))
risk_filter = dropdown_filter("Risk level", sorted(df["risk_flag"].unique()))

filtered = df[
    df["subject"].isin(subjects) & df["department"].isin(departments) & df["risk_flag"].isin(risk_filter)
]

if filtered.empty:
    st.warning("⚠️ No records match the current filters. Select at least one option in each dropdown on the left.")
    st.stop()

# --- KPI row -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", len(filtered))
c2.metric("Avg Attendance %", f"{filtered['attendance_percentage'].mean():.1f}%")
c3.metric("Avg Final Grade %", f"{filtered['final_grade_percent'].mean():.1f}%")
c4.metric("High-Risk Students", int((filtered["risk_flag"] == "High Risk").sum()))

st.divider()

# --- View 1: Attendance vs Marks analysis ---------------------------
st.subheader("1️⃣ Attendance vs Marks Analysis")
fig1 = px.scatter(
    filtered, x="attendance_percentage", y="final_grade_percent",
    color="risk_flag", hover_data=["student_id", "subject"],
    labels={"attendance_percentage": "Attendance %", "final_grade_percent": "Final Grade %"},
    color_discrete_map={"High Risk": "#d62728", "Low Risk": "#2ca02c"},
)
st.plotly_chart(fig1, use_container_width=True)

col_a, col_b = st.columns(2)

# --- View 2: Subject-wise pass/fail distribution ---------------------
with col_a:
    st.subheader("2️⃣ Subject-wise Pass/Fail Distribution")
    pf = filtered.groupby(["subject", "pass_fail"]).size().reset_index(name="count")
    fig2 = px.bar(pf, x="subject", y="count", color="pass_fail", barmode="group",
                  color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"})
    st.plotly_chart(fig2, use_container_width=True)

# --- View 3: Department and semester performance ----------------------
with col_b:
    st.subheader("3️⃣ Department & Semester Performance")
    period_df = filtered.melt(
        id_vars=["department"], value_vars=["G1", "G2", "G3"],
        var_name="period", value_name="grade",
    )
    period_df["semester"] = period_df["period"].map(SEMESTER_LABELS)
    period_df = period_df[period_df["semester"].isin(semesters)]
    period_df["grade_percent"] = period_df["grade"] / 20 * 100

    dept_semester_summary = (
        period_df.groupby(["department", "semester"])["grade_percent"]
        .mean().reset_index()
    )
    fig3 = px.bar(
        dept_semester_summary, x="department", y="grade_percent", color="semester",
        barmode="group",
        category_orders={"semester": list(SEMESTER_LABELS.values())},
        labels={"grade_percent": "Avg Grade %", "department": "Department"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- View 4: High-risk student list -------------------------------
st.subheader("4️⃣ High-Risk Student List (Early Intervention Candidates)")
high_risk = (
    filtered[filtered["risk_flag"] == "High Risk"]
    .sort_values("risk_score", ascending=False)
    [["student_id", "subject", "department", "attendance_percentage", "final_grade_percent",
      "failures", "risk_score"]]
)
st.dataframe(high_risk, use_container_width=True, height=300)
st.caption(f"{len(high_risk)} students flagged for early academic intervention.")

# --- View 5: Student profile & intervention drill-down ----------------
st.subheader("5️⃣ Student Profile & Intervention Detail")
selected_id = st.selectbox("Select a student_id", options=sorted(filtered["student_id"].unique()))
profile = filtered[filtered["student_id"] == selected_id].iloc[0]

pc1, pc2, pc3 = st.columns(3)
pc1.metric("Attendance %", f"{profile['attendance_percentage']:.1f}%")
pc2.metric("Final Grade %", f"{profile['final_grade_percent']:.1f}%")
pc3.metric("Risk Level", profile["risk_flag"])

st.write("**Grade trajectory (Semester 1 → 2 → Final):**")
trend_df = pd.DataFrame({
    "Semester": list(SEMESTER_LABELS.values()),
    "Grade": [profile["G1"], profile["G2"], profile["G3"]],
})
fig5 = px.line(trend_df, x="Semester", y="Grade", markers=True, range_y=[0, 20])
st.plotly_chart(fig5, use_container_width=True)

st.write("**Demographics & engagement:**")
st.json({
    "department": profile["department"], "sex": profile["sex"], "age": int(profile["age"]),
    "address": profile["address"], "internet_access": profile["internet"],
    "study_time_scale_1to4": int(profile["studytime"]),
    "past_class_failures": int(profile["failures"]),
    "assignment_completion_rate_proxy": f"{profile['assignment_completion_rate']:.1f}%",
})

st.divider()
st.caption("Part 1 – Data Pipeline & Analytics Dashboard | Data Engineering and MLOps")
