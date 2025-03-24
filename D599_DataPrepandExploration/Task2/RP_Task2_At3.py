#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy import stats
from scipy.stats import kruskal, shapiro, skew, kurtosis
#%%
df = pd.read_excel('Health Insurance Dataset.xlsx')
#%%
df.head()
#%%
df.info()
#%%
df.isnull().sum()
#%%
#Fill Null Values for both Continuous and Categorical Variables
df['age'] = df['age'].fillna(df['age'].mean())
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['children'] = df['children'].fillna(df['children'].mean())
df['score'] = df['score'].fillna(df['score'].mean())

df['sex'] = df['sex'].fillna('unknown')
df['smoker'] = df['smoker'].fillna('unknown')
df['region'] = df['region'].fillna('unknown')
df['charges'] = df['charges'].fillna('unknown')
df['Level'] = df['Level'].fillna('unknown')
#%%
df.isnull().sum()
#%%
#Check for Inconsistent Entries
inconsistent_entries = {}
for column in df.select_dtypes(include='object').columns:
    unique_values = df[column].unique()
    inconsistent_entries[column] = df[column].unique()
    print(f"{column}: {(unique_values)}")
#%%
df.loc[:,'charges'] = df['charges'].replace({'unknown': '0.0', ' ': '0.0'})
df.loc[:,'region'] = df['region'].replace({' ': 'unknown'})
#%%
#Check for Inconsistent Entries
inconsistent_entries = {}
for column in df.select_dtypes(include='object').columns:
    unique_values = df[column].unique()
    inconsistent_entries[column] = df[column].unique()
    print(f"{column}: {(unique_values)}")
#%%
#Convert Charges Column to a Float Datatype
df.loc[:,'charges'] = df['charges'].astype(float)
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
#%%
df.info()
#%%
########################################## PART 1 - A - 1 Univariate Statistics ###############################################################################################
#CONTINUOUS VARIABLE - AGE

# Histogram and KDE for Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, color='blue', bins=15, alpha=0.6)
plt.title('Distribution of Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%%
df.age.describe()
#%%
ageSkew = df.age.skew()
ageMedian = df.age.median()
print(f'Age Skew: {ageSkew}')
print(f'Age Median: {ageMedian}')
#%%
# Box Plot for Age Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['age'], color='lightblue', width=0.6)
plt.title('Box Plot of Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.xticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
#%%
###################################################################################################
#CONTINUOUS VARIABLE - CHARGES

# Histogram and KDE for Charges Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True, color='blue', bins=15, alpha=0.6)
plt.title('Distribution of Charges', fontsize=16)
plt.xlabel('Charges', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%%
df.charges.describe()
#%%
chargesSkew = df.charges.skew()
chargesMedian = df.charges.median()
print(f'Charges Skew: {chargesSkew}')
print(f'Charges Median: {chargesMedian}')
#%%
# Box Plot for Charges Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['charges'], color='lightblue', width=0.6)
plt.title('Box Plot of Charges', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.xticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
#%%
###################################################################################################
#CATEGORICAL VARIABLES - SEX
df['sex'].describe()
#%%
df['sex'].value_counts()
#%%
df['sex'].value_counts(normalize=True)
#%%
# Distribution of SEX
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='sex', data=df, hue='sex', legend=False)
plt.title('Distribution of Sex')
#%%
###################################################################################################
#CATEGORICAL VARIABLES - SMOKER
df['smoker'].describe()
#%%
df['smoker'].value_counts()
#%%
df['smoker'].value_counts(normalize=True)
#%%
# Distribution of SMOKER
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='smoker', data=df, hue='smoker', legend=False)
plt.title('Distribution of Smoker')
#%%
########################################## PART 1 - B - 1  Bivariate Statistics ###############################################################################################
#CONTINUOUS VARIABLE - Age / Charges
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x='age', y='charges', data=df, hue='age', palette='hls', alpha=0.7, s=80)
plt.legend(title='Age', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Age vs Charges')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Charges', fontsize=14)
plt.show()
spearman_corr = df[['age','charges']].corr(method='spearman')
print ("Spearman Correlation: ")
print(spearman_corr)
#%%
#############################################################ANOTHER WAY TO LOOK HOW CONTINUOUS VARIABLES CORRELATE TO EACH OTHER###################################################################################
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#%%
sns.pairplot(df)
plt.show()
#%%
#####################################################################################################################################################################################
#%%
########################################## PART 1 - B - 1  Bivariate Statistics ###############################################################################################
#CATEGORICAL VARIABLE - SEX / SMOKER
contingency = pd.crosstab(df['sex'], df['smoker'])
print("Contingency Table:")
print(contingency)
#%%
#Perform / Print Chi Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
np.set_printoptions(suppress=True, precision=2)
print(f"\nChi-Square Test Results:\nChi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")
print("\nExpected Frequencies Table:")
print(expected)
#%%
#Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, cmap='Blues', fmt='d')
plt.title('Heatmap of Sex vs Smoker')
plt.xlabel('Smoker')
plt.ylabel('Sex')
plt.show()
#%%
# Contingency Table for 'region' and 'smoker'
contingency2 = pd.crosstab(df['region'], df['smoker'])
print("\nContingency Table for Region and Smoker:")
print(contingency2)
#%%
#Perform a Chi Square Test
chi2, p, dof, expected = chi2_contingency(contingency2)
np.set_printoptions(suppress=True, precision=2)
print(f"\nChi-Square Test Results:\nChi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")
print("\nExpected Frequencies Table:")
print(expected)
#%%
#Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency2, annot=True, cmap='Greens', fmt='d')
plt.title('Heatmap of Region vs Smoker')
plt.xlabel('Smoker')
plt.ylabel('Region')
plt.show()
#%%
########################################################################################### PART 2 : PARAMETRIC STATISTICAL TESTING ############################################################################################
#%%
########################################## PART 2 - C - 1 ###############################################################################################
#%%
#--------------------------------------Does smoking have an effect on a person's BMI?----------------------------------------------
#%%
# Plot Histogram for BMI
sns.histplot(df['bmi'], kde=True)
plt.title('Histogram of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()
#%%
# Separate BMI data for smokers and non-smokers
smokers_bmi = df[df['smoker'] == 'yes']['bmi']
non_smokers_bmi = df[df['smoker'] == 'no']['bmi']
#%%
# Perform t-test on BMI
t_stat, p_value = stats.ttest_ind(smokers_bmi, non_smokers_bmi, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpretation of results
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in BMI between smokers and non-smokers.")
else:
    print("Fail to reject the null hypothesis: No significant difference in BMI between smokers and non-smokers.")
#%%
#Plot Using Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='bmi', data=df, palette="Set2", hue='smoker')
plt.title('Medical Charges by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Medical Charges')
plt.show()
#%%
########################################################################################### PART 3 : NONPARAMETRIC STATISTICAL TESTING ############################################################################################
#%%
########################################## PART 3 - G - 1 ###############################################################################################
#----------------------Is there a significant difference in insurance charges based on gender and BMI categories?
#%%
# Create BMI categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

df['bmi_category'] = df['bmi'].apply(categorize_bmi)
#%%
plt.figure(figsize=(10,6))
sns.boxplot(x='bmi_category', y='charges', hue='sex', data=df, palette='Set2')
plt.title('Insurance Charges by BMI Category and Gender')
plt.xlabel('BMI Category')
plt.ylabel('Insurance Charges')
plt.show()
#%%
# Separate the data by gender
male_data = df[df['sex'] == 'male']
female_data = df[df['sex'] == 'female']

# Perform Kruskal-Wallis test for each gender
kruskal_results_male = kruskal(male_data[male_data['bmi_category'] == 'Underweight']['charges'],
                               male_data[male_data['bmi_category'] == 'Normal weight']['charges'],
                               male_data[male_data['bmi_category'] == 'Overweight']['charges'],
                               male_data[male_data['bmi_category'] == 'Obese']['charges'])

kruskal_results_female = kruskal(female_data[female_data['bmi_category'] == 'Underweight']['charges'],
                                 female_data[female_data['bmi_category'] == 'Normal weight']['charges'],
                                 female_data[female_data['bmi_category'] == 'Overweight']['charges'],
                                 female_data[female_data['bmi_category'] == 'Obese']['charges'])

print("Kruskal-Wallis Test for Males:")
print(f"Statistic: {kruskal_results_male.statistic}, P-value: {kruskal_results_male.pvalue}")

print("Kruskal-Wallis Test for Females:")
print(f"Statistic: {kruskal_results_female.statistic}, P-value: {kruskal_results_female.pvalue}")