import streamlit as st
import pandas as pd
import plotly.express as px

st.header("Model Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.metric("Model Accuracy", f"{st.session_state.model_components['accuracy_rf']:.2%}")

with col2:
    st.metric("AUC Score", f"{st.session_state.model_components['auc_score_rf']:.3f}")


st.subheader("Feature Importance")
        
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

st.subheader("Model Performance Comparison")

performance_data = {
    'Metric': ['Accuracy', 'AUC Score', 'Precision', 'Recall'],
    'Random Forest': [st.session_state.model_components['accuracy_rf'],
                      st.session_state.model_components['precision_rf'], 
                      st.session_state.model_components['recall_rf'],
                      st.session_state.model_components['auc_score_rf']
                      ],
    'Gradient Boosting': [st.session_state.model_components['accuracy_gb'],
                          st.session_state.model_components['precision_gb'],
                          st.session_state.model_components['recall_gb'],
                          st.session_state.model_components['auc_score_gb']]
}

performance_df = pd.DataFrame(performance_data)
st.dataframe(performance_df)