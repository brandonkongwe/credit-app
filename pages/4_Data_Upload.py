from home import train_model
import pandas as pd
import streamlit as st


st.header("Data Upload & Management")

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

uploaded_file = st.file_uploader("Upload borrower data", type=['csv'], 
                                 help="Upload a CSV file containing borrower information. The file should include all required columns.")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df 
        st.success("Data uploaded successfully!")

        with st.spinner("Retraining model with new data..."):
            model_components, X_test, y_test = train_model(df)
            st.session_state.model_components = model_components
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

        st.success("Model retrained successfully!")
    except Exception as e:
        st.error(f"Error uploading data: {str(e)}")


st.subheader("Current Dataset")
st.write(f"Shape: {st.session_state.df.shape}")
st.dataframe(st.session_state.df.head())

st.subheader("Data Summary")
st.write(st.session_state.df.describe())