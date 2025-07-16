from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sqlite3


st.set_page_config(
    page_title="Credit Risk Scoring App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.session_state.df = pd.read_csv("data/train.csv")

def init_database():
    conn = sqlite3.connect('credit_assessments.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            risk_score REAL,
            risk_level TEXT,
            confidence REAL,
            assessment_date TEXT,
            features TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_database()


def preprocess_data(df):
    """Preprocess the data for model training"""
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    le_dict = {}
    categorical_cols = ['gender', 'owns_car', 'owns_house', 'occupation_type', 'migrant_worker']

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    return df, le_dict


def train_model(df):
    """Train the credit risk model"""
    features = df.drop(['customer_id', 'name', 'credit_card_default'], axis=1).columns
    df_processed, le_dict = preprocess_data(df)
    X = df_processed[features]
    y = df_processed['credit_card_default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)

    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

    gb_pred = gb_model.predict(X_test_scaled)
    gb_prob = gb_model.predict_proba(X_test_scaled)[:, 1]

    accuracy_rf = accuracy_score(y_test, rf_pred)
    precision_rf = precision_score(y_test, rf_pred)
    recall_rf = recall_score(y_test, rf_pred)
    auc_score_rf = roc_auc_score(y_test, rf_prob)

    accuracy_gb = accuracy_score(y_test, gb_pred)
    precision_gb = precision_score(y_test, gb_pred)
    recall_gb = recall_score(y_test, gb_pred)
    auc_score_gb = roc_auc_score(y_test, gb_prob)

    model_components = {
        'model_rf': rf_model,
        'model_gb': gb_model,
        'scaler': scaler,
        'features': features,
        'le_dict': le_dict,
        'accuracy_rf': accuracy_rf,
        'precision_rf': precision_rf,
        'recall_rf': recall_rf,
        'auc_score_rf': auc_score_rf,
        'accuracy_gb': accuracy_gb,
        'precision_gb': precision_gb,
        'recall_gb': recall_gb,
        'auc_score_gb': auc_score_gb
    }

    return model_components, X_test_scaled, y_test


def predict_risk(model_components, input_data):
    """Predict risk for a single borrower"""
    input_df = pd.DataFrame([input_data])

    for col, le in model_components['le_dict'].items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                # handling unseen categories
                input_df[col] = 0

    X_input = input_df[model_components['features']]
    X_input_scaled = model_components['scaler'].transform(X_input)

    risk_prob = model_components['model_rf'].predict_proba(X_input_scaled)[0, 1]
    risk_pred = model_components['model_rf'].predict(X_input_scaled)[0]
    confidence = max(risk_prob, 1 - risk_prob)

    return risk_prob, risk_pred, confidence


def get_risk_level(risk_prob):
    """Determine risk level based on probability"""
    if risk_prob < 0.3:
        return "Low Risk"
    elif risk_prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"
    


def save_assessment(customer_data, risk_score, risk_level, confidence):
    """Save assessment to database"""
    conn = sqlite3.connect('credit_assessments.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO assessments (customer_id, name, age, gender, risk_score, risk_level, confidence, assessment_date, features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        customer_data.get('customer_id', 'N/A'),
        customer_data.get('name', 'N/A'),
        customer_data.get('age', 0),
        customer_data.get('gender', 'N/A'),
        risk_score,
        risk_level,
        confidence,
        datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        str(customer_data)
    ))

    conn.commit()
    conn.close()


def main():
    if 'model_components' not in st.session_state:
        with st.spinner("Training model..."):
            model_components, X_test, y_test = train_model(st.session_state.df)
            st.session_state.model_components = model_components
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

    st.title("Credit Risk Scoring App")
    st.header("Credit Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", len(st.session_state.df))


    with col2:
        default_rate = st.session_state.df['credit_card_default'].mean()
        st.metric("Default Rate", f"{default_rate:.2%}")


    with col3:
        avg_credit_score = st.session_state.df['credit_score'].mean()
        st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")


    with col4:
        model_accuracy = st.session_state.model_components['accuracy_rf']
        st.metric("Model Accuracy", f"{model_accuracy:.2%}")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Rate by Age Group")
        age_groups = pd.cut(st.session_state.df['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
        default_by_age = st.session_state.df.groupby(age_groups)['credit_card_default'].mean()

        fig = px.bar(x=default_by_age.index, y=default_by_age.values,
                    labels={'x': 'Age Group', 'y': 'Default Rate'})
        st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.subheader("Credit Score Distribution")
        fig = px.histogram(st.session_state.df, x='credit_score', nbins=30,
                        color='credit_card_default', barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()