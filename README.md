Analyzing The Dynamics of Sex Offenses In The United States
-------------------------- 

Author: Zainab Makhdum

## File Paths

* All data files are located in the 'capstone/data' folder. 
* All the source code can be found in either of the following locations:
    - 'capstone/src/rape_incidents_capstone.py'
    - 'capstone/docs/blog.qmd'
* The project proposal can be found in the following location: 'capstone/docs/proposal.qmd'
* The references can be found in the following location: 'capstone/docs/references.bib'
* The final version of the blog post can be accessed through the following link: [Analyzing The Dynamics of Sex Offenses In The United States](https://zainabmakhdum.github.io/capstone/blog.html)

## Notes on Extracted Data

* The extracted files contain multiple categories of crimes e.g., robbery, human trafficking, assault, etc, however, for the purpose of this research, we are only interested in 'Rape' and 'Sex Offenses' categories. Therefore, one of the initial steps in the data cleaning process consisted of extracting only 'sex offenses' or 'rape' categories and discarding the rest. 

* The last seven datasets described in the *Data File Name and Description* section, are used for modelling. Relevant information from all of these datasets (specifically the following variables: age, sex, location, and weapons used by offenders) are merged together prior to performing modelling. 

## File Name and Description

- [Offenses By State](data/Crimes_Against_Persons_Offenses_Offense_Category_by_State_2022.xlsx): Displays multiple categories of crimes for each state across the United States. 
- [Offenses By County](data/Table_10_Offenses_Known_to_Law_Enforcement_by_State_by_Metropolitan_and_Nonmetropolitan_Counties_2022.clsx): Displays multiple categories of crimes for each county across the United States. 
- [National Rape Trend](data/National_Rape_Ten_Year_Trend.csv): Displays the total number of rape incidents by year (2012-2022)
- [Offense by Location](data/Crimes_Against_Persons_Offenses_Offense_Category_by_Location_2022.xlsx): Displays multiple categories of crimes based on specific location where the crime took place. 
- [Victim-Offender Relationship](data/Relationship_of_Victims_to_Offenders_by_Offense_Category_2022.xlsx): Displays relationships between victims and offenders by offense category.
- [Attempt vs. Complete Offenses](data/Number_of_Offenses_Completed_and_Attempted_by_Offense_Category_2022.xlsx): Displays the total number of completed vs. attempted offenses. 
- [Victim Age](data/Victims_Age_by_Offense_Category_2022.xlsx): Displays total number of victims by age categories for multiple offenses. 
- [Victim Race](data/Victims_Race_by_Offense_Category_2022.xlsx): Displays total number of victims by race for multiple offenses.
- [Victim Sex](data/Victims_Sex_by_Offense_Category_2022.xlsx): Displays total number of victims by sex for multiple offenses.
- [Arrestee Age](data/Arrestees_Age_by_Arrest_Offense_Category_2022.xlsx): Displays total number of arrestees by age for multiple offenses.
- [Arrestee Race](data/Arrestees_Race_by_Arrest_Offense_Category_2022.xlsx): Displays total number of arrestees by race for multiple offenses.
- [Arrestee Sex](data/Arrestees_Sex_by_Arrest_Offense_Category_2022.xlsx): Displays total number of arrestees by sex for multiple offenses.
- [Offense Incident](data/NIBRS_incident.csv): Displays incidents and their respective information such as the data, type of incident etc. This dataset is used for case study and modelling. 
- [Offense Location Type](data/NIBRS_LOCATION_TYPE.csv): Displays location names associated with respective location codes. This dataset is used for modelling. 
- [Offender Information](data/NIBRS_OFFENDER.csv): Displays offender demographics associated with each unique offender. This dataset is used for modelling. 
- [Offense Type](data/NIBRS_OFFENSE_TYPE.csv): Displays offense information based on unique offense code. This dataset is used for modelling. 
- [Offense Details](data/NIBRS_OFFENSE.csv): Displays offense details - including whether it was attempted or completed, age, sex etc., of offender - that is associated with each offense. This dataset is used for modelling. 
- [Weapon Type](data/NIBRS_WEAPON_TYPE.csv): Displays the unique weapons and their associated descriptions. This dataset is used for modelling. 
- [Weapon Used in Crime](data/NIBRS_WEAPON.csv): Displays details on which weapon was associated with which specific offense. This dataset is used for modelling. 
