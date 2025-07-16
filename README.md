# Credit Risk Scoring App

This is a Python-based web application designed to assess and evaluate the creditworthiness of borrowers. The app provides a comprehensive framework to assess credit risk, visualize model performance, and manage borrower data. 

**Data Source**: [American Express Credit dataset](https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021)

## Features

### 1. Credit Risk Assessment
- **Manual Entry**: Input borrower details manually to receive a detailed credit risk assessment.
- **Batch Processing**: Upload a CSV file containing borrower data for batch risk assessment. The app processes multiple records simultaneously and provides key metrics such as risk scores, confidence levels, and recommendations.

### 2. Model Performance Analysis
- Visualize key performance metrics of the credit risk model, such as:
  - **Accuracy**
  - **AUC Score**
  - **Feature Importance**
- Compare the performance of different algorithms (Random Forest and Gradient Boosting).

### 3. Data Upload and Preprocessing
- Upload new borrower datasets in CSV format.
- Automatically preprocess data for training and evaluation.
- Retrain the credit risk model with uploaded data.

### 4. Saved Assessments
- View and manage previously conducted risk assessments.
- Export assessment results in CSV format.

## Key Metrics
- **Risk Score**: Probability of default.
- **Risk Level**: Categorized into High Risk, Medium Risk, and Low Risk.
- **Confidence**: Confidence level of the model's prediction.
- **Recommendations**: Approval or denial of credit applications.

## Technologies Used
- **Streamlit**: Framework for building the web application.
- **Pandas**: Data manipulation and analysis.
- **Plotly**: Data visualization.
- **SQLite**: Database for storing credit assessments.
- **Scikit-learn**: Machine learning models for credit risk evaluation.

## Structure
```bash
credit-app/
├── data/
│   ├── train.csv
│   └── test.csv
├── pages/
│   ├── 1_Risk_Assessment.py
│   ├── 2_Model_Performance.py
│   ├── 3_Saved_Assessments.py
│   └── 4_Data_Upload.py
├── home.py
├── README.md
├── requirements.txt
└── credit_assessments.db
```

## Installation

### Prerequisites
- Python 3.8 or above.
- The following Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `plotly`
  - `scikit-learn`
  - `sqlite3`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/brandonkongwe/credit-app.git
   cd credit-app
   ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate  
    ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run home.py
   ```

4. Access the application in your web browser at `http://localhost:8501`.

## Usage

### Risk Assessment
1. Navigate to the **Risk Assessment** page.
2. Choose between **Manual Entry** or **Batch Processing**.
3. Enter the required borrower information or upload a CSV file.
4. View detailed results, including:
   - Risk Score
   - Risk Level
   - Confidence
   - Recommendations

### Model Performance
1. Navigate to the **Model Performance** page.
2. View metrics such as Accuracy, AUC Score, and Feature Importance.
3. Compare the performance of different machine learning models.

### Data Upload
1. Navigate to the **Data Upload** page.
2. Upload new borrower data in CSV format.
3. Retrain the model with the new dataset.

### Saved Assessments
1. Navigate to the **Saved Assessments** page.
2. View previously assessed borrowers.
3. Download the assessment results in CSV format.

## Data Requirements

The following columns must be included in the borrower data for accurate risk assessment:

| Variable Name | Data Type | Description | Example Values |
|--------------|-----------|-------------|----------------|
| **customer_id** | String | Unique identifier for each customer | "CST_115179", "CST_121920" |
| **name** | String | Customer's full name | "ita Bose", "Alper Jonathan" |
| **age** | Integer | Age of the customer in years | 46, 29, 37 |
| **gender** | String | Gender ("M" for Male, "F" for Female) | "M", "F" |
| **owns_car** | String | Car ownership ("Y" for Yes, "N" for No) | "Y", "N" |
| **owns_house** | String | House ownership ("Y" for Yes, "N" for No) | "Y", "N" |
| **no_of_children** | Float | Number of children | 0.0, 1.0, 2.0 |
| **net_yearly_income** | Float | Annual net income | 107934.04, 109862.62 |
| **no_of_days_employed** | Float | Number of days employed | 612.0, 2771.0 |
| **occupation_type** | String | Type of occupation | "Laborers", "Core staff", "Accountants", "Unknown" |
| **total_family_members** | Float | Total family members | 1.0, 2.0, 3.0 |
| **migrant_worker** | Float | Migrant worker status (1.0=Yes, 0.0=No) | 1.0, 0.0 |
| **yearly_debt_payments** | Float | Total yearly debt payments | 33070.28, 15329.53 |
| **credit_limit** | Float | Total available credit limit | 18690.93, 37745.19 |
| **credit_limit_used(%)** | Integer | Percentage of credit limit used | 73, 52, 43 |
| **credit_score** | Float | Credit score | 544.0, 857.0 |
| **prev_defaults** | Integer | Number of previous credit defaults | 0, 1, 2 |
| **default_in_last_6months** | Integer | Default in last 6 months (1=Yes, 0=No) | 0, 1 |
| **credit_card_default** | Integer | Credit card default status (1=Yes, 0=No) | 0, 1 |
