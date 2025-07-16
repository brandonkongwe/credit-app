from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3

st.header("Saved Credit Assessments")

conn = sqlite3.connect('credit_assessments.db')
assessments_df = pd.read_sql_query("SELECT * FROM assessments ORDER BY assessment_date DESC", conn)
conn.close()

if len(assessments_df) > 0:
    st.subheader("Recent Assessments")

    for idx, row in assessments_df.head(10).iterrows():
        with st.expander(f"{row['name']} - {row['customer_id']} ({row['assessment_date']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Risk Score:** {row['risk_score']:.2%}")
                st.write(f"**Age:** {row['age']}")
            
            with col2:
                st.write(f"**Risk Level:** {row['risk_level']}")
                st.write(f"**Gender:** {row['gender']}")
            
            with col3:
                st.write(f"**Confidence:** {row['confidence']:.2%}")
                st.write(f"**Date:** {row['assessment_date']}")

    csv = assessments_df.to_csv(index=False)
    st.download_button(
        label="Download All Assessments",
        data=csv,
        file_name=f"credit_assessments_{datetime.now().strftime('%d%m%Y_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.subheader("Assessment Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Assessments", len(assessments_df))

    with col2:
        high_risk_count = len(assessments_df[assessments_df['risk_level'] == 'High Risk'])
        st.metric("High Risk Customers", high_risk_count)

    with col3:
        avg_risk_score = assessments_df['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk_score:.2%}")

    risk_counts = assessments_df['risk_level'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                title="Risk Level Distribution", color=risk_counts.index, color_discrete_map={
                    'Low Risk': '#66B2FF',
                    'Medium Risk': 'orange',
                    'High Risk': '#990000'
                })
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No assessments saved yet. Complete some risk assessments to see them here!")
