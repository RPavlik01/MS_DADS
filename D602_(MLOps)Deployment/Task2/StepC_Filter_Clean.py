import pandas as pd

#Load formatted dataset
file_path = 'Data/formatted_data.csv'
df = pd.read_csv(file_path)

#Drop rows with missing values
df.dropna(subset=['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'], inplace=True)

#Convert Columns from FLOAT to INT
int_columns = ['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']
for col in int_columns:
    df[col] = df[col].astype(int)

#Remove Duplicate Rows
df.drop_duplicates(inplace=True)

#Save cleaned dataset
df.to_csv('Data/cleaned_data.csv', index=False)

print("Data cleaning complete. Cleaned dataset saved as 'cleaned_data.csv'.")