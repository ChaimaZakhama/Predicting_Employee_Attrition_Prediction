import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore

st.write("""
# Employee Attrition Prediction App

This app predicts Employee Attrition!

Data obtained from the [IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) from Kaggle.
""")

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.slider('Age', 18, 60, 30)
        daily_rate = st.sidebar.slider('DailyRate', 100, 1500, 800)
        distance_from_home = st.sidebar.slider('DistanceFromHome', 1, 30, 10)
        environment_satisfaction = st.sidebar.selectbox('EnvironmentSatisfaction', (1, 2, 3, 4))
        hourly_rate = st.sidebar.slider('HourlyRate', 30, 100, 60)
        job_role = st.sidebar.selectbox('JobRole', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
        job_satisfaction = st.sidebar.selectbox('JobSatisfaction', (1, 2, 3, 4))
        monthly_income = st.sidebar.slider('MonthlyIncome', 1000, 20000, 5000)
        monthly_rate = st.sidebar.slider('MonthlyRate', 2000, 30000, 15000)
        num_companies_worked = st.sidebar.slider('NumCompaniesWorked', 0, 10, 3)
        percent_salary_hike = st.sidebar.slider('PercentSalaryHike', 0, 25, 15)
        relationship_satisfaction = st.sidebar.selectbox('RelationshipSatisfaction', (1, 2, 3, 4))
        total_working_years = st.sidebar.slider('TotalWorkingYears', 0, 40, 10)
        years_at_company = st.sidebar.slider('YearsAtCompany', 0, 40, 5)
        years_with_curr_manager = st.sidebar.slider('YearsWithCurrManager', 0, 20, 5)

        data = {
            'Age': age,
            'DailyRate': daily_rate,
            'DistanceFromHome': distance_from_home,
            'EnvironmentSatisfaction': environment_satisfaction,
            'HourlyRate': hourly_rate,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies_worked,
            'PercentSalaryHike': percent_salary_hike,
            'RelationshipSatisfaction': relationship_satisfaction,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'YearsWithCurrManager': years_with_curr_manager
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
employee_data_raw = pd.read_csv('employee_data_oversampled.csv')
attrition = employee_data_raw.drop(columns=['Attrition'], axis=1)
df = pd.concat([input_df,attrition],axis=0)


encode = ['JobRole']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('attrition_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
employee_attrition = np.array(['No', 'Yes'])
st.write(f"Employee attrition prediction: {employee_attrition[prediction][0]}")

st.subheader('Prediction Probability')
prob_df = pd.DataFrame(prediction_proba, columns=['Probability of Staying', 'Probability of Leaving'])
st.write(prob_df)

# Add text indicating what each probability means
if prediction_proba[0][1] > 0.5:
    st.write("The model predicts that the employee is likely to leave.")
else:
    st.write("The model predicts that the employee is likely to stay.")
