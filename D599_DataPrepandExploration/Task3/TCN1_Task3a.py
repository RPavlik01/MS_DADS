#%%
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')
#%%
df = pd.read_csv('Megastore_Dataset_Task_3 3.csv')
#%%
# Select relevant columns for analysis
selected_columns = ['OrderPriority', 'CustomerOrderSatisfaction', 'ProductName', 'Region']
selected_data = df[selected_columns]
#%%
# Ordinal Encoding for OrderPriority and CustomerOrderSatisfaction
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
selected_data.loc[:, 'OrderPriority_Encoded'] = ordinal_encoder.fit_transform(selected_data[['OrderPriority']])
#%%
ordinal_encoder = OrdinalEncoder(categories=[['Dissatisfied', 'Very Dissatisfied', 'Prefer not to answer', 'Satisfied', 'Very Satisfied']])
selected_data.loc[:, 'CustomerOrderSatisfaction_Encoded'] = ordinal_encoder.fit_transform(selected_data[['CustomerOrderSatisfaction']])
#%%
# Label Encoding for 'Region'
label_encoder = LabelEncoder()
selected_data.loc[:, 'Region_Encoded'] = label_encoder.fit_transform(selected_data['Region'])
#%%
# One-Hot Encoding for 'ProductName'
one_hot_encoded_products = pd.get_dummies(selected_data['ProductName'], prefix='Product')
#%%
# Combine the encoded columns with the original data
encoded_data = pd.concat([selected_data, one_hot_encoded_products], axis=1)
#%%
# Save the cleaned dataset to a CSV file
encoded_data.to_csv('Cleaned_Megastore_Data.csv', index=False)
#%%
# Group data by OrderID to create transactions
transaction_data = df.groupby('OrderID')['ProductName'].apply(list)
#%%
# Convert transactions into a binary matrix format
transactional_df = pd.get_dummies(transaction_data.apply(pd.Series).stack()).groupby(level=0).sum()
#%%
# Ensure binary encoding in the transactional data
transactional_df = transactional_df.applymap(lambda x: 1 if x > 0 else 0)
#%%
# Run the Apriori algorithm
frequent_itemsets = apriori(transactional_df, min_support=0.01, use_colnames=True)
#%%
# Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))
#%%
# Display the association rules
print("Association Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
#%%
# Extract the top 3 rules with the highest lift
top_rules = rules.sort_values(by='lift', ascending=False).head(3)
#%%
print("Top 3 Rules:\n", top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])