import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Import and format airport data")
parser.add_argument("--data", type=str, default="Data/Default.csv",
                    help="Path to the dataset CSV file")
# You can add other parameters if needed, e.g. num_alphas for later steps
# parser.add_argument("--num_alphas", type=int, default=20, help="Number of alpha increments")
args = parser.parse_args()

# Use the provided data file path
file_path = args.data

# Load the dataset
df = pd.read_csv(file_path)

#Column Names Used in the Poly Regressor Script
# | YEAR | MONTH | DAY | DAY_OF_WEEK | ORG_AIRPORT | DEST_AIRPORT | SCHEDULED_DEPARTURE | DEPARTURE_TIME | DEPARTURE_DELAY | SCHEDULED_ARRIVAL | ARRIVAL_TIME | ARRIVAL_DELAY |
# |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
# | integer | integer | integer | integer | string | string | integer | integer | integer | integer | integer | integer |

#Rename columns in dataset to match model script
df.rename(columns={
    'DAY_OF_MONTH': 'DAY',
    'ORIGIN': 'ORG_AIRPORT',
    'DEST': 'DEST_AIRPORT',
    'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE',
    'DEP_TIME': 'DEPARTURE_TIME',
    'DEP_DELAY': 'DEPARTURE_DELAY',
    'CRS_ARR_TIME': 'SCHEDULED_ARRIVAL',
    'ARR_TIME': 'ARRIVAL_TIME',
    'ARR_DELAY': 'ARRIVAL_DELAY'
}, inplace=True)

#Filter Dataset for Miami International Airport (MIA) departures
df = df[df['ORG_AIRPORT'] == 'MIA']

#Save the Formatted Dataset
df.to_csv('Data/formatted_data.csv', index=False)

print('Formatted data saved successfully')