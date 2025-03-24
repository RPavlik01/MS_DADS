#Rob Pavlik D599 Task1 TCN1
#Import Libraries
import pandas as pd
#%%
import numpy as np
#%%
#Read the dataset using Pandas
df = pd.read_excel('EmpTurnoverDS.xlsx')
#%%
#Use Shape to see the number of rows and columns 
df.shape
#%%
#Use Head to get a sense of the data in each column 
df.head()
#%%
#View a quick summary of the dataset using Info
df.info
#%%
#Get each Column Datatype using .dtypes
df.dtypes
#%%
#Count the Duplicates in the Dataset
df.duplicated().sum()
#%%
#See all rows that are duplicates
df[df.duplicated()]
#%%
#Drop Duplicate rows keeping the first instance of each row based on the column EmployeeNumber
df = df.drop_duplicates(subset=['EmployeeNumber'], keep='first')
#%%
#Get new column count after dropping duplicates
df.shape
#%%
#Get a count of each column that has a null or missing value (only show columns with null greater than 0)
df.isnull().sum()[df.isnull().sum() > 0]
#%%
#fill in null values with (0.0) for numeric columns
df.loc[:, 'TrainingTimesLastYear'] = df['TrainingTimesLastYear'].fillna(0.0)
df.loc[:, 'NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(0.0)
#%%
#fill in null values with 'Unknown' for object columns
df.loc[:, 'EducationField'] = df['EducationField'].fillna('Unknown')
df.loc[:, 'Gender'] = df['Gender'].fillna('Unknown')
#%%
#Use the .interpolate method to fill in null values in meaningful numeric columns 
df.loc[:, 'MonthlyIncome'] = df['MonthlyIncome'].interpolate()
df.loc[:, 'MonthlyRate'] = df['MonthlyRate'].interpolate()
df.loc[:, 'TotalWorkingYears'] = df['TotalWorkingYears'].interpolate()
df.loc[:, 'YearsSinceLastPromotion'] = df['YearsSinceLastPromotion'].interpolate()
#%%
#Check to ensure null or missing values have been dealt with
df.isnull().sum()[df.isnull().sum() > 0]
#%%
#Identify Inconsistent Entries - Loop through each column and get every unique value 
inconsistent_entries = {}
for column in df.select_dtypes(include=['object']).columns:
    unique_values = df[column].unique()
    inconsistent_entries[column] = df[column].unique()
    print(f"{column}: {(unique_values)}")
#%%
#Use dictionary replace method to take Inconsistent Entries and replace with 'Unknown'
df.loc[:, 'BusinessTravel'] = df['BusinessTravel'].replace({1: 'Unknown', -1: 'Unknown', '00': 'Unknown', ' ': 'Unknown'})
df.loc[:, 'EducationField'] = df['EducationField'].replace({' ': 'Unknown'})
df.loc[:, 'JobRole'] = df['JobRole'].replace({' ': 'Unknown'})
#%%

#%%
#Change column to a numeric datatype and changes any value that cant be converted to a numeric value to a NaN value
df.loc[:, 'YearsWithCurrManager'] = pd.to_numeric(df['YearsWithCurrManager'], errors='coerce')
#%%
#Change values less than 0 to be NaN
df.loc[df['YearsWithCurrManager'] < 0, 'YearsWithCurrManager'] = np.nan
#%%
#Ensure the columns new datatype is set to Float
df.loc[:, 'YearsWithCurrManager'] = df['YearsWithCurrManager'].astype(float)
#%%
#Use Describe to help identify outliers (Mostly looking at Min and Max values in each column)
df.describe()
#%%
#Fix the outliers in 'Age' with capping at 18 and 100 
df.loc[df['Age'] < 18, 'Age'] = 18
df.loc[df['Age'] > 100, 'Age'] = 100
#%%
#Employee Count should only be 1. This fix is to correct the -1s in the data
df.loc[df['EmployeeCount'] < 0, 'EmployeeCount'] = 1
#%%
#Monthly income has a negative value. To fix by applying abs() to the value
df.loc[:,'MonthlyIncome'] = df['MonthlyIncome'].abs()
#%%
#TotalWorkingYears has a -1 for the min. Fix will be to assign -1 to 1
df.loc[df['TotalWorkingYears'] < 0, 'TotalWorkingYears'] = 1
#%%
#Ensure outliers have been dealt with
df.describe()
#%%
#Check the max value of Column with extreme outlier
df['MonthlyRate'].max()
#%%
#Identify all values that are extreme outliers in this column
unique_values = df['MonthlyRate'].unique()
for num in unique_values:
    if num > 2000000:
        print(num)
#%%
#Create a variable containing the known outliers
outlier_values = [872214872214.0, 877411155.0]
#%%
#Find the mean of the all the rows not including the outliers
mean_value = df[df['MonthlyRate'].isin(outlier_values) == False]['MonthlyRate'].mean()
#%%
#Set the two rows with outliers to the calculated mean value
df.loc[df['MonthlyRate'].isin(outlier_values), 'MonthlyRate'] = mean_value
#%%
#Check the new max value for Outlier Column
df['MonthlyRate'].max()
#%%
#Save updated dataset to a new Excel file
df.to_excel('Updated_EmpTurnoverDS2.xlsx', index=False)
#%%
#https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=e0389bf2-bff1-4184-8e7b-b2240179e59a