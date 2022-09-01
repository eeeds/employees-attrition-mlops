import pickle

import streamlit as st

st.write(
    """
# Attrition Prediction App
This app predicts if a employee will leave the company or not.
"""
)

st.markdown(
    """
<style>
.big-font {
    font-size:50px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    model = '../models/pipeline.bin'

    with open(model, 'rb') as f_in:
        pipeline = pickle.load(f_in)

    return pipeline


pipeline = load_model()

col11, col12, col13 = st.columns(3)

business_travel = col11.selectbox('Business Travel', ('Travel_Frequently', 'Travel_Rarely', 'Non-Travel'))

department = col12.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))
education_field = col13.selectbox(
    'Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other')
)
col21, col22, col23 = st.columns(3)
gender = col21.selectbox('Gender', ('Female', 'Male'))

job_role = col22.selectbox(
    'Job Role',
    (
        'Sales Executive',
        'Research Scientist',
        'Laboratory Technician',
        'Manufacturing Director',
        'Healthcare',
        'Representative',
        'Manager',
        'Sales Representative',
        'Research Director',
        'Human Resorces',
    ),
)

marital_status = col23.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))

col31, col32, col33 = st.columns(3)


age = col31.slider(
    'Age',
    min_value=18,
    max_value=90,
    value=30,
)

daily_rate = col32.slider(
    'Daily Rate',
    min_value=0,
    max_value=1500,
    value=0,
)
distance_from_home = col33.slider(
    'Distance From Home',
    min_value=0,
    max_value=30,
    value=0,
)

col41, col42, col43 = st.columns(3)

education = col41.slider(
    'Education',
    min_value=1,
    max_value=5,
    value=1,
)
enviroment_satisfaction = col42.slider(
    'Enviroment Satisfaction',
    min_value=1,
    max_value=4,
    value=1,
)

hourly_rate = col43.slider(
    'Hourly Rate',
    min_value=0,
    max_value=100,
    value=0,
)
col51, col52, col53 = st.columns(3)

job_involvement = col51.slider(
    'Job Involvement',
    min_value=1,
    max_value=4,
    value=1,
)
job_level = col52.slider(
    'Job Level',
    min_value=1,
    max_value=5,
    value=1,
)
job_satisfaction = col53.slider(
    'Job Satisfaction',
    min_value=1,
    max_value=4,
    value=1,
)
col61, col62, col63 = st.columns(3)

monthly_income = col61.slider(
    'Monthly Income',
    min_value=0,
    max_value=100000,
    value=0,
)
monthly_rate = col62.slider(
    'Monthly Rate',
    min_value=0,
    max_value=30000,
    value=0,
)
num_comp_worked_prev = col63.slider(
    'Number of Companies Worked Previously',
    min_value=0,
    max_value=10,
    value=0,
)
col71, col72, col73 = st.columns(3)

over_18 = col71.selectbox('Over 18', ('Yes', 'No'))

overtime = col72.selectbox('Overtime', ('Yes', 'No'))

percent_salary_hike = col73.slider(
    'Percent Salary Hike',
    min_value=0,
    max_value=30000,
    value=0,
)
col81, col82, col83 = st.columns(3)

performance_rating = col81.slider(
    'Performance Rating',
    min_value=1,
    max_value=4,
    value=1,
)

relationship_satisfaction = col82.slider(
    'Relationship Satisfaction',
    min_value=1,
    max_value=4,
    value=1,
)

stock_option_level = col83.slider(
    'Stock Option Level',
    min_value=1,
    max_value=4,
    value=1,
)

col91, col92, col93 = st.columns(3)

total_working_years = col91.slider(
    'Total Working Years',
    min_value=0,
    max_value=50,
    value=0,
)
training_times_last_year = col92.slider(
    'Training Times Last Year',
    min_value=0,
    max_value=10,
    value=0,
)
work_life_balance = col93.slider(
    'Work Life Balance',
    min_value=1,
    max_value=4,
    value=1,
)
col101, col102, col103 = st.columns(3)

years_at_company = col101.slider(
    'Years at Company',
    min_value=0,
    max_value=50,
    value=0,
)
years_in_current_role = col102.slider(
    'Years in Current Role',
    min_value=0,
    max_value=50,
    value=0,
)
years_since_last_promotion = col103.slider(
    'Years Since Last Promotion',
    min_value=0,
    max_value=50,
    value=0,
)
col111, col112, col113 = st.columns(3)
years_with_current_manager = col111.slider(
    'Years with Current Manager',
    min_value=0,
    max_value=50,
    value=0,
)

employee = {
    'BusinessTravel': business_travel,
    'Department': department,
    'EducationField': education_field,
    'Gender': gender,
    'JobRole': job_role,
    'MaritalStatus': marital_status,
    'Age': int(age),
    'DailyRate': float(daily_rate),
    'DistanceFromHome': float(distance_from_home),
    'Education': int(education),
    'EnviromentSatisfaction': int(enviroment_satisfaction),
    'HourlyRate': float(hourly_rate),
    'JobInvolvement': int(job_involvement),
    'JobLevel': int(job_level),
    'JobSatisfaction': int(job_satisfaction),
    'MonthlyIncome': float(monthly_income),
    'MonthlyRate': float(monthly_rate),
    'NumCompaniesWorkedPrev': int(num_comp_worked_prev),
    'Over18': str(over_18) == 'Yes',
    'Overtime': str(overtime) == 'Yes',
    'PercentSalaryHike': float(percent_salary_hike),
    'PerformanceRating': int(performance_rating),
    'RelationshipSatisfaction': int(relationship_satisfaction),
    'StockOptionLevel': int(stock_option_level),
    'TotalWorkingYears': int(total_working_years),
    'TrainingTimesLastYear': int(training_times_last_year),
    'WorkLifeBalance': int(work_life_balance),
    'YearsAtCompany': int(years_at_company),
    'YearsInCurrentRole': int(years_in_current_role),
    'YearsSinceLastPromotion': int(years_since_last_promotion),
    'YearsWithCurrentManager': int(years_with_current_manager),
}

pred = pipeline.predict_proba(employee)[0, 1]
pred = float(pred)

col114, col115 = st.columns(2)

col114.write('<p class="big-font"> Probability of leaving: </p> ', unsafe_allow_html=True)
col115.write(
    f"""<p class="big-font">
{pred:0.2f}
</p>
""",
    unsafe_allow_html=True,
)

st.write(
    """
App was devoloped by [Esteban Encina](https://github.com/eeeds)
"""
)
