
import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sb
from wordcloud import WordCloud
import os

# file directory path 
# changing file path
os.chdir('/Users/zainabmakhdum/Downloads/capstone/data')
print("Updated Directory:", os.getcwd())

"""Defining Functions"""

def convert_cols_to_lower_and_snake_case(data_frame):
    """
    Converting column names in each dataframe to lower case and snake_case for consistency.
    """
    data_frame.columns = [col.lower().replace(' ', '_') for col in data_frame.columns]


def df_info(dataset_name):
    """
    Identifying datatypes for each column in all datasets
    """
    dataset_name = dataset_name.info()
    return dataset_name


def missing_values(dataset):
    """
    Getting the total number of missing values in each dataset.
    """
    missing_vals = dataset.isnull().sum().sum()
    return missing_vals


"""Reading all datasets"""

files = [
    "Crimes_Against_Persons_Offenses_Offense_Category_by_State_2022.xlsx",
    "Relationship_of_Victims_to_Offenders_by_Offense_Category_2022.xlsx",
    "Table_8_Offenses_Known_to_Law_Enforcement_by_State_by_City_2022.xlsx",
    "Crimes_Against_Persons_Offenses_Offense_Category_by_Location_2022.xlsx",
    "Victims_Age_by_Offense_Category_2022.xlsx",
    "Victims_Race_by_Offense_Category_2022.xlsx",
    "Victims_Sex_by_Offense_Category_2022.xlsx",
    "Arrestees_Age_by_Arrest_Offense_Category_2022.xlsx",
    "Arrestees_Race_by_Arrest_Offense_Category_2022.xlsx",
    "Arrestees_Sex_by_Arrest_Offense_Category_2022.xlsx",
    "Crimes_Against_Persons_Incidents_Offense_Category_by_Time_of_Day_2022.xlsx",
    "Number_of_Offenses_Completed_and_Attempted_by_Offense_Category_2022.xlsx",
    "National_Rape_Ten_Year_Trend.csv",
    "NIBRS_OFFENSE_TYPE.csv",
    "NIBRS_OFFENDER.csv",
    "NIBRS_OFFENSE.csv",
    "uscities.csv",
    "NIBRS_incident.csv"
]

# empty dict to store dfs
df_dict = {}

for file in files:
    if file.endswith(".xlsx"):
        df_dict[file] = pd.read_excel(file)
    elif file.endswith(".csv"):
        df_dict[file] = pd.read_csv(file)


"""Writing Datasets"""
# offenses by state
offense_by_state = df_dict['Crimes_Against_Persons_Offenses_Offense_Category_by_State_2022.xlsx']

# keeping relevant cols
selected_cols = ['Crimes Against Persons Offenses', 'Unnamed: 2', 'Unnamed: 8']
offense_by_state = offense_by_state[selected_cols]

# renaming cols
new_col_names = ['State', 'population_covered', 'sex_offenses']
offense_by_state.columns = new_col_names

# removing first few irrelevant rows
offense_by_state = offense_by_state.iloc[5:, :]

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(offense_by_state)

# resetting index
offense_by_state.reset_index(drop=True, inplace=True)

code = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
        'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
        'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
        'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
        'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
        'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY'}

offense_by_state['Code'] = offense_by_state['state'].map(code)
offense_by_state.head()

# relationship between victim and offender 
vic_offen_relationship = df_dict['Relationship_of_Victims_to_Offenders_by_Offense_Category_2022.xlsx']

# modifying col headers
# renaming col headers
rel_col_names = ['relationship', 'total_victims', 'family_member', 'family_member_and_other',
                 'ktv_and_other', 'stranger', 'other']
vic_offen_relationship.columns = rel_col_names

# removing first five  and last five (irrelevant) rows
vic_offen_relationship = vic_offen_relationship.iloc[4:, :]

# selecting only sex offenses crime
vic_offen_relationship = vic_offen_relationship[vic_offen_relationship['relationship'].str.contains('Sex Offenses')]

# transposing and resetting index
vic_offen_relationship = vic_offen_relationship.T.reset_index()
    
# making row 1 the col headers
vic_offen_relationship.columns = vic_offen_relationship.iloc[0]
        
# removing row 1
vic_offen_relationship = vic_offen_relationship[2:]

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(vic_offen_relationship)

vic_offen_relationship

# offenses by city
offense_by_city = df_dict["Table_8_Offenses_Known_to_Law_Enforcement_by_State_by_City_2022.xlsx"]

# removing the first two rows
offense_by_city = offense_by_city.iloc[2:, :]

# making the first row the column headers
offense_by_city.columns = offense_by_city.iloc[0, :]

# removing the first row
offense_by_city = offense_by_city.iloc[1:, :]

# removing the last two rows
offense_by_city = offense_by_city.iloc[:-2, :]

# keeping relevant cols
selected_city_cols = ['State', 'City', 'Population', 'Rape']
offense_by_city = offense_by_city[selected_city_cols]

# filling the missing values for state based on most recent non-NaN value
offense_by_city['State'] = offense_by_city['State'].fillna(method='ffill')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(offense_by_city)

# converting the state col to title format
offense_by_city['state'] = offense_by_city['state'].str.title()

# adding state code column
offense_by_city['Code'] = offense_by_city['state'].map(code)

# Assuming your DataFrame is named offense_by_city
offense_by_city['city'] = offense_by_city['city'].str.strip()

# resetting the index
offense_by_city.reset_index(drop=True, inplace=True)
offense_by_city.head()

# crime by location
crime_location = df_dict['Crimes_Against_Persons_Offenses_Offense_Category_by_Location_2022.xlsx']

# keeping relevant cols
selected_location_cols = ['Crimes Against Persons Offenses', 'Unnamed: 6']
crime_location = crime_location[selected_location_cols]

# renaming cols
new_loc_col_names = ['location', 'sex_offenses']
crime_location.columns = new_loc_col_names
crime_location = crime_location.iloc[5:, :]
crime_location.reset_index(drop=True, inplace=True)
crime_location.head()

# Victim Demographics
# victim age
victim_age = df_dict['Victims_Age_by_Offense_Category_2022.xlsx']

# assigning the fourth row as the column headers
victim_age.columns = victim_age.iloc[3, :]

# changing specific col names
victim_age.columns = victim_age.columns.str.replace('10 and\nUnder', '10_and_under')
victim_age.columns = victim_age.columns.str.replace('66 and\nOver', '66_and_over')
victim_age.columns = victim_age.columns.str.replace('Unknown\nAge', 'unknown_age')
victim_age.columns.values[:2] = ['offense_category', 'total_victims']

# filtering for sex offenses in the offense_category
victim_age = victim_age[victim_age['offense_category'] == 'Sex Offenses']


# transposing and resetting index
victim_age = victim_age.T.reset_index()
    
# making row 1 the col headers
victim_age.columns = victim_age.iloc[0]
        
# removing row 1
victim_age = victim_age[1:]

# changing specific col names after transposing
victim_age.columns = victim_age.columns.str.replace('offense_category', 'age')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(victim_age)

# removing row 1
victim_age = victim_age.iloc[1:]

# changing specific col names after transposing
victim_age.columns = victim_age.columns.str.replace('sex_offenses', 'victims')

# Resetting the index
victim_age.reset_index(drop=True, inplace=True)
victim_age.head()

# Victim Demographics
# victim race
victim_race = df_dict['Victims_Race_by_Offense_Category_2022.xlsx']

# assigning the fourth row as the column headers
victim_race.columns = victim_race.iloc[3, :]

# changing specific col names
victim_race.columns.values[:2] = ['offense_category', 'total_victims']
victim_race.columns = victim_race.columns.str.replace('\n', ' ')
victim_race.columns = victim_race.columns.str.replace(' ', '_')

# filtering for sex offenses in the offense_category
victim_race = victim_race[victim_race['offense_category'] == 'Sex Offenses']

# transposing and resetting index
victim_race = victim_race.T.reset_index()
    
# making row 1 the col headers
victim_race.columns = victim_race.iloc[0]

# removing row 1
victim_race = victim_race[1:]

# changing specific col names after transposing
victim_race.columns = victim_race.columns.str.replace('offense_category', 'race')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(victim_race)

# removing row 1
victim_race = victim_race.iloc[1:]

# changing specific col names after transposing
victim_race.columns = victim_race.columns.str.replace('sex_offenses', 'victims')

# Resetting the index
victim_race.reset_index(drop=True, inplace=True)

victim_race

# Victim Demographics
# victim sex

victim_sex = df_dict['Victims_Sex_by_Offense_Category_2022.xlsx']

# assigning the fourth row as the column headers
victim_sex.columns = victim_sex.iloc[3, :]

# changing specific col names
victim_sex.columns.values[:2] = ['offense_category', 'total_victims']
victim_sex.columns = victim_sex.columns.str.replace(' ', '_')

# filtering for sex offenses in the offense_category
victim_sex = victim_sex[victim_sex['offense_category'] == 'Sex Offenses']

# transposing and resetting index
victim_sex = victim_sex.T.reset_index()
    
# making row 1 the col headers
victim_sex.columns = victim_sex.iloc[0]
        
# removing row 1
victim_sex = victim_sex[1:]

# changing specific col names after transposing
victim_sex.columns = victim_sex.columns.str.replace('offense_category', 'sex')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(victim_sex)

# removing row 1
victim_sex = victim_sex.iloc[1:]

# changing specific col names after transposing
victim_sex.columns = victim_sex.columns.str.replace('sex_offenses', 'victims')

# Resetting the index
victim_sex.reset_index(drop=True, inplace=True)

victim_sex

# Arrestee Demographics
# arrestee age

arrestee_age = df_dict['Arrestees_Age_by_Arrest_Offense_Category_2022.xlsx']

# assigning the fourth row as the column headers
arrestee_age.columns = arrestee_age.iloc[3, :]
arrestee_age

# changing specific col names
arrestee_age.columns.values[:2] = ['offense_category', 'total_arrestees']
arrestee_age.columns = arrestee_age.columns.str.replace('\n', ' ')
arrestee_age.columns = arrestee_age.columns.str.replace(' ', '_')

# filtering for sex offenses in the offense_category
arrestee_age['offense_category'] = arrestee_age['offense_category'].str.strip()
arrestee_age = arrestee_age[arrestee_age['offense_category'] == 'Sex Offenses']


# transposing and resetting index
arrestee_age = arrestee_age.T.reset_index()
    
# making row 1 the col headers
arrestee_age.columns = arrestee_age.iloc[0]
        
# removing row 1
arrestee_age = arrestee_age[1:]

# changing specific col names after transposing
arrestee_age.columns = arrestee_age.columns.str.replace('offense_category', 'age')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(arrestee_age)

# removing row 1
arrestee_age = arrestee_age.iloc[1:]

# changing specific col names after transposing
arrestee_age.columns = arrestee_age.columns.str.replace('sex_offenses', 'arrestees')

# making all rows in the age column lowercase
arrestee_age['age'] = arrestee_age['age'].str.lower()

# Resetting the index
arrestee_age.reset_index(drop=True, inplace=True)

arrestee_age.head()

# Arrestee Demographics
# arrestee race

arrestee_race = df_dict['Arrestees_Race_by_Arrest_Offense_Category_2022.xlsx']

# assigning the fourth row as the column headers
arrestee_race.columns = arrestee_race.iloc[3, :]

# changing specific col names
arrestee_race.columns.values[:2] = ['offense_category', 'total_arrestees']
arrestee_race.columns = arrestee_race.columns.str.replace('\n', ' ')
arrestee_race.columns = arrestee_race.columns.str.replace(' ', '_')

# filtering for sex offenses in the offense_category
arrestee_race['offense_category'] = arrestee_race['offense_category'].str.strip()
arrestee_race = arrestee_race[arrestee_race['offense_category'] == 'Sex Offenses']


# transposing and resetting index
arrestee_race = arrestee_race.T.reset_index()
    
# making row 1 the col headers
arrestee_race.columns = arrestee_race.iloc[0]
        
# removing row 1
arrestee_race = arrestee_race[1:]

# changing specific col names after transposing
arrestee_race.columns = arrestee_race.columns.str.replace('offense_category', 'race')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(arrestee_race)

# removing row 1
arrestee_race = arrestee_race.iloc[1:]

# changing specific col names after transposing
arrestee_race.columns = arrestee_race.columns.str.replace('sex_offenses', 'arrestees')

# Resetting the index
arrestee_race.reset_index(drop=True, inplace=True)

arrestee_race

# Arrestee Demographics
# Arrestee Sex

arrestee_sex = df_dict['Arrestees_Sex_by_Arrest_Offense_Category_2022.xlsx']

# assigning the fourth row as the column headers
arrestee_sex.columns = arrestee_sex.iloc[3, :]

# changing specific col names
arrestee_sex.columns.values[:2] = ['offense_category', 'total_arrestees']
arrestee_sex.columns = arrestee_sex.columns.str.replace(' ', '_')

# filtering for sex offenses in the offense_category
arrestee_sex['offense_category'] = arrestee_sex['offense_category'].str.strip()
arrestee_sex = arrestee_sex[arrestee_sex['offense_category'] == 'Sex Offenses']


# transposing and resetting index
arrestee_sex = arrestee_sex.T.reset_index()
    
# making row 1 the col headers
arrestee_sex.columns = arrestee_sex.iloc[0]
        
# removing row 1
arrestee_sex = arrestee_sex[1:]

# changing specific col names after transposing
arrestee_sex.columns = arrestee_sex.columns.str.replace('offense_category', 'sex')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(arrestee_sex)

# removing row 1
arrestee_sex = arrestee_sex.iloc[1:]

# Resetting the index
arrestee_sex.reset_index(drop=True, inplace=True)

# changing specific col names after transposing
arrestee_sex.columns = arrestee_sex.columns.str.replace('sex_offenses', 'arrestees')

arrestee_sex

# recorded offenses by time of day
offenses_time = df_dict['Crimes_Against_Persons_Incidents_Offense_Category_by_Time_of_Day_2022.xlsx']

# assigning the fourth row as the column headers
offenses_time.columns = offenses_time.iloc[3, :]

# changing specific col names
offenses_time.columns.values[:2] = ['time', 'total_incidents']
offenses_time.columns = offenses_time.columns.str.replace('\n', ' ')
offenses_time.columns = offenses_time.columns.str.replace(' ', '_')

# keeping relevant cols
selected_time_cols = ['time', 'Sex_Offenses']
offenses_time = offenses_time[selected_time_cols]

# removing irrelevant rows
offenses_time = offenses_time.iloc[5:, :]
offenses_time = offenses_time.iloc[:-1, :]
offenses_time = offenses_time[offenses_time['time'].isin(['Total A.M. Hours', 'Total P.M. Hours']) == False]

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(offenses_time)

# resetting index
offenses_time.reset_index(drop=True, inplace=True)
offenses_time

# attempt vs complete status
attempt_complete_status = df_dict['Number_of_Offenses_Completed_and_Attempted_by_Offense_Category_2022.xlsx']

# make row 2 the column header
# assigning the fourth row as the column headers
attempt_complete_status.columns = attempt_complete_status.iloc[1, :]

# remove /n from column headers and make it snake_case

# changing specific col names
attempt_complete_status.columns = attempt_complete_status.columns.str.replace('\n', ' ')
attempt_complete_status.columns = attempt_complete_status.columns.str.replace(' ', '_')

# keep only the relevant rows - remove all others
# filtering for sex offenses in the offense_category
attempt_complete_status = attempt_complete_status[attempt_complete_status['Offense_Category'] == 'Sex Offenses']

# transposing and resetting index
attempt_complete_status = attempt_complete_status.T.reset_index()
    
# making row 1 the col headers
attempt_complete_status.columns = attempt_complete_status.iloc[0]
        
# removing row 1
attempt_complete_status = attempt_complete_status[1:]

# changing specific col names after transposing
attempt_complete_status.columns = attempt_complete_status.columns.str.replace('Offense_Category', 'status')

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(attempt_complete_status)

attempt_complete_status

# Nationwide Ten Year Rape Trend
national_rape_trend = df_dict['National_Rape_Ten_Year_Trend.csv']

# keeping only the relevant rows
national_rape_trend = national_rape_trend.head(1)

# changing the value of row one in 'Years' column to 'rape_incidents'
national_rape_trend.loc[0, 'Years'] = 'rape_incidents'

# transposing and resetting index
national_rape_trend = national_rape_trend.T.reset_index()
    
# making row 1 the col headers
national_rape_trend.columns = national_rape_trend.iloc[0]
        
# removing row 1
national_rape_trend = national_rape_trend[1:]

# converting to lower case and snake case by calling on fucntion
convert_cols_to_lower_and_snake_case(national_rape_trend)

national_rape_trend.head()

# Offense Type
offense_type = df_dict['NIBRS_OFFENSE_TYPE.csv']

# selecting relevant offenses (sexual crimes (forcible and non-forcible))
# extracting relevant offense codes that are to be utilized for selecting relevant incidents from offender dataset
offense_type = offense_type[
    (offense_type['offense_category_name'] == 'Sex Offenses') |
    (offense_type['offense_category_name'] == 'Sex Offenses, Non-forcible')]

# resetting  index
offense_type.reset_index(drop=True, inplace=True)

# getting a list of all offense codes related to sex offenses
offense_code_vals = offense_type['offense_code'].unique()
offense_code_list = list(offense_code_vals)
offense_code_list

# offense data
offense_data = df_dict['NIBRS_OFFENSE.csv']

# filtering for offenses by relevant offense_code_list
offense_data = offense_data[offense_data['offense_code'].isin(offense_code_list)]

# removing irrelevant cols
offense_data = offense_data.iloc[:, :-3]

# resetting the index
offense_data.reset_index(drop=True, inplace=True)

# Offender dataset
offenders = df_dict['NIBRS_OFFENDER.csv']

# filtering the 'offenders' dataframe based on the 'incident_id' column in the 'offense_data' dataframe

# extracting unique incident_id values from offense_data
incident_id_unq = offense_data['incident_id'].unique()

# filtering offenders based on incident_id_unq
offenders = offenders[offenders['incident_id'].isin(incident_id_unq)]

# resetting index
offenders.reset_index(drop=True, inplace=True)

# merging the 'offense_code' and 'attempt_complete_flag'cols with the offenders df
merged_offenders = pd.merge(offenders, offense_data[['incident_id', 'offense_code',
                                                              'attempt_complete_flag']],
                            on='incident_id', how='left')

# removing irrelevant cols
remove_irrel_cols = ['offender_seq_num', 'age_id', 'ethnicity_id',
                     'age_range_low_num', 'age_range_high_num', 'race_id']

merged_offenders = merged_offenders.drop(columns=remove_irrel_cols)


# add the 'offense_name' and 'offense_category_name' from the 'offense_type' df to the merged_offenders
merged_offenders = pd.merge(merged_offenders, offense_type[['offense_code', 'offense_name',
                                                            'offense_category_name']],
                            on='offense_code', how='left')

# getting month and datetime data for offenders based on incident_id from incident dataset
incident_df = df_dict['NIBRS_incident.csv']

# extracting date and id columns from incident_df and putting them into a new df
incident_date_df = incident_df[['incident_id', 'incident_date', 'report_date_flag']]

# merging incident_date_df with merged_offenders based on incident_id
merged_offenders = pd.merge(merged_offenders, incident_date_df, on='incident_id', how='left')

# converting incident_date column to datetime dtype
merged_offenders['incident_date'] = pd.to_datetime(merged_offenders['incident_date'], errors='coerce')

# `report_date_flag`column indicates if the agency used the report date as the `incident_date.`
report_date_flag_counts = merged_offenders['report_date_flag'].value_counts()

# removing rows where report_date_flag is 't: 9 rows total
merged_offenders = merged_offenders[merged_offenders['report_date_flag'] != 't']
merged_offenders.head()

"""Data Processing/Cleaning"""

# putting all datasets in a dictionary to access later

all_datasets = {
    'offense_by_state': offense_by_state,
    'vic_offen_relationship': vic_offen_relationship,
    'offense_by_city': offense_by_city,
    'crime_location': crime_location,
    'victim_age': victim_age,
    'victim_race': victim_race,
    'victim_sex': victim_sex,
    'arrestee_age': arrestee_age,
    'arrestee_race': arrestee_race,
    'arrestee_sex': arrestee_sex,
    'offenses_time': offenses_time,
    'national_rape_trend': national_rape_trend,
    'merged_offenders': merged_offenders,
    'attempt_complete_status': attempt_complete_status
}

# Identifying and Changing Datatypes
for dataset_name, dataset in all_datasets.items():
    print("Dataset:", dataset_name)
    print(df_info(dataset))
    print("\n" + "="*50 + "\n")


def change_datatype(data_frame, col_name, modified_datatype):
    """
    Changeing the datatype of a column in a a specific dataframe.
    """
    data_frame[col_name] = data_frame[col_name].astype(modified_datatype)
    return data_frame


offense_by_state = change_datatype(offense_by_state, 'population_covered', int)
offense_by_state = change_datatype(offense_by_state, 'sex_offenses', int)
vic_offen_relationship = change_datatype(vic_offen_relationship, 'sex_offenses', int)

offense_by_city = change_datatype(offense_by_city, 'population', int)
offense_by_city = change_datatype(offense_by_city, 'rape', int)

crime_location = change_datatype(crime_location, 'sex_offenses', int)

attempt_complete_status = change_datatype(attempt_complete_status, 'sex_offenses', int)

victim_age = change_datatype(victim_age, 'victims', int)
victim_race = change_datatype(victim_race, 'victims', int)
victim_sex = change_datatype(victim_sex, 'victims', int)

arrestee_age = change_datatype(arrestee_age, 'arrestees', int)
arrestee_race = change_datatype(arrestee_race, 'arrestees', int)
arrestee_sex = change_datatype(arrestee_sex, 'arrestees', int)

offenses_time = change_datatype(offenses_time, 'sex_offenses', int)

national_rape_trend = change_datatype(national_rape_trend, 'rape_incidents', int)
national_rape_trend = change_datatype(national_rape_trend, 'years', int)

merged_offenders = change_datatype(merged_offenders, 'offender_id', object)
merged_offenders = change_datatype(merged_offenders, 'incident_id', object)

for dataset_name, dataset in all_datasets.items():
    print("Dataset:", dataset_name)
    print(df_info(dataset))
    print("\n" + "="*50 + "\n")

"""Identifying Missing Values"""

"""
    As shown below - no missing values found for any variables in any of the datasets with the exception of
    age_num column where the missing values are shown as 'NS'. Therefore, first, we will redefine 'NS'
    as NaN and then change the dtype of this column to numeric i.e., float. Then, we will examine if there
    are outliers present. If the presence of outliers is detected, then the missing values will be
    replaced by the median, otherwise the mean.
"""

# missing values as 'NS' convert them to NaN and impute them
merged_offenders['age_num'] = merged_offenders['age_num'].replace('NS', np.nan)

# detecting missing values
for dataset_name, dataset in all_datasets.items():
    print("Dataset:", dataset_name)
    print("Total Missing Values:", missing_values(dataset))
    print("-"*50)

"""Now, we can see that there are 169 missing values in the merged_offenders df.
Therefore, we'll take a closer look."""

merged_offenders.isna().sum()

# changing dtype from object to float to visualize distribution of age
merged_offenders = change_datatype(merged_offenders, 'age_num', float)


def boxplot_visualize(data_frame, col_name):
    """
    Createing a boxplot for a specific column in a dataframe to examine distributions.
    """
    fig = px.box(data_frame,
                 y = col_name,
                 title = f'Boxplot for {col_name}',
                 labels = {col_name: col_name})
    
    fig.update_layout(
        title=dict(font=dict(size=14)),
        height = 400,
        width = 400
    )
    
    fig.show()


boxplot_visualize(merged_offenders, 'age_num')

"""Since there are outliers present, we want to replace the missing
values in this column with the median."""

# replacing the missing values with the median for age num column
merged_offenders['age_num'] = merged_offenders['age_num'].fillna(merged_offenders['age_num'].median())

"""Now, we can see that there are no missing values in this dataset."""
print(merged_offenders.isna().sum())