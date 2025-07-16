from home import predict_risk, get_risk_level, save_assessment
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px


st.header("Credit Risk Assessment")

assessment_type = st.radio("Assessment Type:", ["Manual Entry", "Batch Processing"])

if assessment_type == "Manual Entry":
    st.subheader("Enter Borrower Information")

    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.text_input("Customer ID", value="CST_NEW001")
        name = st.text_input("Name", value="John Doe")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["M", "F"])
        owns_car = st.selectbox("Owns Car", ["Y", "N"])
        owns_house = st.selectbox("Owns House", ["Y", "N"])
        no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        net_yearly_income = st.number_input("Net Yearly Income", min_value=10000, value=100000)


    with col2:
        no_of_days_employed = st.number_input("Days Employed", min_value=0, value=1000)
        occupation_type = st.selectbox("Occupation", ['Unknown', 'Laborers', 'Core staff', 'Accountants',
                                        'High skill tech staff', 'Sales staff', 'Managers', 'Drivers',
                                        'Medicine staff', 'Cleaning staff', 'HR staff', 'Security staff',
                                        'Cooking staff', 'Waiters/barmen staff', 'Low-skill Laborers',
                                        'Private service staff', 'Secretaries', 'Realty agents',
                                        'IT staff'])
        total_family_members = st.number_input("Total Family Members", min_value=1, value=2)
        migrant_worker = st.selectbox("Migrant Worker", [0, 1])
        yearly_debt_payments = st.number_input("Yearly Debt Payments", min_value=0, value=20000)
        credit_limit = st.number_input("Credit Limit", min_value=1000, value=30000)
        credit_limit_used = st.number_input("Credit Limit Used (%)", min_value=0.0, max_value=100.0, value=50.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=1000, value=700)
        prev_defaults = st.number_input("Previous Defaults", min_value=0, value=0)
        default_in_last_6months = st.selectbox("Default in Last 6 Months", [0, 1])

    if st.button("Assess Risk", type="primary"):
        input_data = {
            'customer_id': customer_id,
            'name': name,
            'age': age,
            'gender': gender,
            'owns_car': owns_car,
            'owns_house': owns_house,
            'no_of_children': no_of_children,
            'net_yearly_income': net_yearly_income,
            'no_of_days_employed': no_of_days_employed,
            'occupation_type': occupation_type,
            'total_family_members': total_family_members,
            'migrant_worker': migrant_worker,
            'yearly_debt_payments': yearly_debt_payments,
            'credit_limit': credit_limit,
            'credit_limit_used(%)': credit_limit_used,
            'credit_score': credit_score,
            'prev_defaults': prev_defaults,
            'default_in_last_6months': default_in_last_6months
        }

        risk_prob, risk_pred, confidence = predict_risk(st.session_state.model_components, input_data)
        risk_level = get_risk_level(risk_prob)

        st.subheader("Risk Assessment Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Risk Score", f"{risk_prob:.2%}")


        with col2:
            st.metric("Risk Level", f"{risk_level}")

        
        with col3:
            st.metric("Confidence", f"{confidence:.2%}")

        
        if risk_level == "High Risk":
            st.html(f'<h4>High Risk Customer</h4><p>This customer has a {risk_prob:.1%} probability of default. Consider additional verification or higher interest rates.</p>')
        elif risk_level == "Medium Risk":
            st.html(f'<h4>Medium Risk Customer</h4><p>This customer has a {risk_prob:.1%} probability of default. Standard approval process recommended.</p>')
        else:
            st.html(f'<h4>Low Risk Customer</h4><p>This customer has a {risk_prob:.1%} probability of default. Approved for standard terms.</p>')


        st.subheader("Feature Importance Analysis")

        if hasattr(st.session_state.model_components['model_rf'], 'feature_importances_'):
            feature_importance = st.session_state.model_components['model_rf'].feature_importances_
            feature_names = st.session_state.model_components['features']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        title="Top 10 Most Important Features")
            
            st.plotly_chart(fig, use_container_width=True)

        save_assessment(input_data, risk_prob, risk_level, confidence)

        st.subheader("Credit Assessment Report")
                
        report_data = {
            'Customer Information': {
                'Customer ID': customer_id,
                'Name': name,
                'Age': age,
                'Gender': gender,
                'Income': f"${net_yearly_income:,.2f}",
                'Credit Score': credit_score
            },
            'Risk Assessment': {
                'Risk Score': f"{risk_prob:.2%}",
                'Risk Level': risk_level,
                'Confidence': f"{confidence:.2%}",
                'Assessment Date': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'Recommendation': {
                'Decision': 'Approve' if risk_level == 'Low Risk' else 'Review Required' if risk_level == 'Medium Risk' else 'Decline',
                'Interest Rate Adjustment': '+0%' if risk_level == 'Low Risk' else '+2%' if risk_level == 'Medium Risk' else '+5%',
                'Credit Limit Recommendation': f"${credit_limit * (1 - risk_prob):,.2f}"
            }
        }

        report_text = "CREDIT RISK ASSESSMENT REPORT\n"
        report_text += "=" * 50 + "\n\n"

        for section, data in report_data.items():
            report_text += f"{section.upper()}\n"
            report_text += "-" * len(section) + "\n"
            for key, value in data.items():
                report_text += f"{key}: {value}\n"
            report_text += "\n"

        st.download_button(
            label="Download Credit Report",
            data=report_text,
            file_name=f"credit_report_{customer_id}_{datetime.now().strftime('%d%m%Y_%H%M%S')}.txt",
            mime="text/plain"
        )

elif assessment_type == "Batch Processing":
    st.subheader("Batch Risk Assessment")

    uploaded_batch_file = st.file_uploader(
        "Upload CSV file with borrower data for batch processing",
        type=['csv'],
        help="Upload a CSV file containing borrower information. The file should include all required columns."
    )

    if uploaded_batch_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch_file)
            
            st.success(f"Loaded {len(batch_df)} records for batch processing")
            
            st.subheader("Preview of Uploaded Data")
            st.dataframe(batch_df.head())
            
            required_columns = ['customer_id', 'name', 'age', 'gender', 'owns_car', 'owns_house', 
                                'no_of_children', 'net_yearly_income', 'no_of_days_employed', 
                                'occupation_type', 'total_family_members', 'migrant_worker', 
                                'yearly_debt_payments', 'credit_limit', 'credit_limit_used(%)', 
                                'credit_score', 'prev_defaults', 'default_in_last_6months']
            
            missing_columns = [col for col in required_columns if col not in batch_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your CSV file contains all required columns.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    process_limit = st.number_input(
                        "Number of records to process", 
                        min_value=1, 
                        max_value=len(batch_df), 
                        value=min(100, len(batch_df)),
                        help="Limit the number of records to process (for performance)"
                    )
                
                if st.button("Start Batch Processing", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    batch_results = []
                    
                    for idx, row in batch_df.head(process_limit).iterrows():
                        progress = (idx + 1) / process_limit
                        progress_bar.progress(progress)
                        status_text.text(f"Processing record {idx + 1} of {process_limit}: {row['name']}")
                        
                        try:
                            input_data = {
                                'customer_id': row['customer_id'],
                                'name': row['name'],
                                'age': row['age'],
                                'gender': row['gender'],
                                'owns_car': row['owns_car'],
                                'owns_house': row['owns_house'],
                                'no_of_children': row['no_of_children'],
                                'net_yearly_income': row['net_yearly_income'],
                                'no_of_days_employed': row['no_of_days_employed'],
                                'occupation_type': row['occupation_type'],
                                'total_family_members': row['total_family_members'],
                                'migrant_worker': row['migrant_worker'],
                                'yearly_debt_payments': row['yearly_debt_payments'],
                                'credit_limit': row['credit_limit'],
                                'credit_limit_used(%)': row['credit_limit_used(%)'],
                                'credit_score': row['credit_score'],
                                'prev_defaults': row['prev_defaults'],
                                'default_in_last_6months': row['default_in_last_6months']
                            }
                            
                            risk_prob, risk_pred, confidence = predict_risk(st.session_state.model_components, input_data)
                            risk_level = get_risk_level(risk_prob)
                            
                            result = {
                                'customer_id': row['customer_id'],
                                'name': row['name'],
                                'age': row['age'],
                                'gender': row['gender'],
                                'net_yearly_income': row['net_yearly_income'],
                                'credit_score': row['credit_score'],
                                'risk_score': risk_prob,
                                'risk_level': risk_level,
                                'confidence': confidence,
                                'decision': 'Approve' if risk_level == 'Low Risk' else 'Review Required' if risk_level == 'Medium Risk' else 'Decline',
                                'interest_rate_adjustment': '+0%' if risk_level == 'Low Risk' else '+2%' if risk_level == 'Medium Risk' else '+5%',
                                'recommended_credit_limit': row['credit_limit'] * (1 - risk_prob)
                            }
                            
                            batch_results.append(result)
                            
                            save_assessment(input_data, risk_prob, risk_level, confidence)
                            
                        except Exception as e:
                            st.error(f"Error processing record {idx + 1}: {str(e)}")
                            continue
                    
                    progress_bar.progress(1.0)
                    status_text.text("Batch processing completed!")
                    
                    st.success(f"Successfully processed {len(batch_results)} records")
                    
                    results_df = pd.DataFrame(batch_results)
                    
                    st.subheader("Batch Processing Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Processed", len(results_df))
                    
                    with col2:
                        high_risk_count = len(results_df[results_df['risk_level'] == 'High Risk'])
                        st.metric("High Risk", high_risk_count)
                    
                    with col3:
                        medium_risk_count = len(results_df[results_df['risk_level'] == 'Medium Risk'])
                        st.metric("Medium Risk", medium_risk_count)
                    
                    with col4:
                        low_risk_count = len(results_df[results_df['risk_level'] == 'Low Risk'])
                        st.metric("Low Risk", low_risk_count)
                    
                    st.subheader("Risk Distribution")
                    risk_counts = results_df['risk_level'].value_counts()
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                                title="Risk Level Distribution", color=risk_counts.index, color_discrete_map={
                                    'Low Risk': '#66B2FF',
                                    'Medium Risk': 'orange',
                                    'High Risk': '#990000'
                                })
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Detailed Results")
                    
                    risk_filter = st.selectbox(
                        "Filter by Risk Level",
                        ["All"] + list(results_df['risk_level'].unique())
                    )
                    
                    if risk_filter != "All":
                        filtered_results = results_df[results_df['risk_level'] == risk_filter]
                    else:
                        filtered_results = results_df
                    
                    st.dataframe(
                        filtered_results[['customer_id', 'name', 'age', 'credit_score', 
                                        'risk_score', 'risk_level', 'confidence', 'decision']],
                        use_container_width=True
                    )
                    
                    st.subheader("Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Detailed Results (CSV)",
                            data=csv_data,
                            file_name=f"batch_risk_assessment_{datetime.now().strftime('%d%m%Y_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        summary_report = f"""BATCH RISK ASSESSMENT SUMMARY REPORT
                        {'='*60}

                        Processing Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}
                        Total Records Processed: {len(results_df)}

                        RISK DISTRIBUTION:
                        {'-'*20}
                        High Risk: {high_risk_count} ({high_risk_count/len(results_df)*100:.1f}%)
                        Medium Risk: {medium_risk_count} ({medium_risk_count/len(results_df)*100:.1f}%)
                        Low Risk: {low_risk_count} ({low_risk_count/len(results_df)*100:.1f}%)

                        STATISTICS:
                        {'-'*20}
                        Average Risk Score: {results_df['risk_score'].mean():.2%}
                        Average Confidence: {results_df['confidence'].mean():.2%}
                        Average Credit Score: {results_df['credit_score'].mean():.0f}
                        Average Income: ${results_df['net_yearly_income'].mean():,.2f}

                        RECOMMENDATIONS:
                        {'-'*20}
                        Approve: {len(results_df[results_df['decision'] == 'Approve'])}
                        Review Required: {len(results_df[results_df['decision'] == 'Review Required'])}
                        Decline: {len(results_df[results_df['decision'] == 'Decline'])}

                        HIGH RISK CUSTOMERS:
                        {'-'*20}
                        """
                        
                        high_risk_customers = results_df[results_df['risk_level'] == 'High Risk']
                        for _, customer in high_risk_customers.iterrows():
                            summary_report += f"- {customer['name']} ({customer['customer_id']}): {customer['risk_score']:.1%} risk\n"
                        
                        st.download_button(
                            label="Download Summary Report",
                            data=summary_report,
                            file_name=f"batch_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    progress_bar.empty()
                    status_text.empty()
        
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains all required columns.")
    
    else:
        st.info("Upload a CSV file to start batch processing")
        
        st.subheader("Required CSV Format")
        st.write("Your CSV file should contain the following columns:")
        
        sample_data = {
            'customer_id': ['CST_001', 'CST_002'],
            'name': ['John Doe', 'Jane Smith'],
            'age': [35, 42],
            'gender': ['M', 'F'],
            'owns_car': ['Y', 'N'],
            'owns_house': ['Y', 'Y'],
            'no_of_children': [0, 2],
            'net_yearly_income': [100000, 120000],
            'no_of_days_employed': [1000, 2000],
            'occupation_type': ['Core staff', 'Accountants'],
            'total_family_members': [2, 4],
            'migrant_worker': [0, 1],
            'yearly_debt_payments': [20000, 25000],
            'credit_limit': [30000, 40000],
            'credit_limit_used(%)': [50.0, 60.0],
            'credit_score': [700, 650],
            'prev_defaults': [0, 1],
            'default_in_last_6months': [0, 1]
        }
        st.dataframe(pd.DataFrame(sample_data))
