#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

%matplotlib inline
#%%
df = pd.read_csv('D600 Task 1 Dataset 1 Housing Information.csv')
#%%
df.head()
#%%
#C1 - Identify the Dependent and Independent Variables
dependent_variable = 'Price'
independent_variables = ['CrimeRate', 'SchoolRating', 'DistanceToCityCenter', 'EmploymentRate', 'LocalAmenities', 'TransportAccess']

selected_columns = [dependent_variable] + independent_variables
df_selected = df[selected_columns]

print(f"Dependent Variable: {dependent_variable}\nIndependent Variables: {independent_variables}")
#%%
#C2 - Descriptive Statistics for Dependent and Independent Variables
df_selected.describe()
#%%
#C3 - Univariate Visualization for Dependent and Independent Variables (Histograms)
for col in selected_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_selected[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
#%%
#C3 - Bivariate Visualization for Dependent and Independent Variables (Scatterplots)
for col in independent_variables:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_selected, x=col, y=dependent_variable, color='green')
    plt.title(f'{dependent_variable} vs {col}')
    plt.xlabel(col)
    plt.ylabel(dependent_variable)
    plt.show()
#%%
#D1 - Split the Dataset into two datasets - only include the selected variables
train_ratio = 0.8

# Split the data
train_data, test_data = train_test_split(df_selected, test_size=1-train_ratio, random_state=42)

# Export the datasets to CSV files
train_data.to_csv('training_dataset.csv', index=False)
test_data.to_csv('test_dataset.csv', index=False)
#%%
#D2 - Initial Linear Regression Model
X_train = train_data[independent_variables]
y_train = train_data[dependent_variable]

# Add a constant term for the intercept
X_train = sm.add_constant(X_train)

# Build the initial regression model
model = sm.OLS(y_train, X_train).fit()

# Display the model summary
print(model.summary())
#%%
#D2 - Optimization: Using Backward Elimination
def backward_elimination(X, y, significance_level=0.05):
    X = sm.add_constant(X)
    while True:
        model = sm.OLS(y, X).fit()
        max_p_value = model.pvalues.max()
        if max_p_value > significance_level:
            excluded_feature = model.pvalues.idxmax()
            print(f"Dropping '{excluded_feature}' with p-value {max_p_value}")
            X = X.drop(columns=[excluded_feature])
        else:
            break
    return model

# Perform backward elimination
optimized_model = backward_elimination(X_train, y_train)

# Display the summary of the optimized model
print(optimized_model.summary())
#%%
#D2 - Extracting Key Metrics
r_squared = optimized_model.rsquared
adjusted_r_squared = optimized_model.rsquared_adj
f_stat = optimized_model.fvalue
prob_f_stat = optimized_model.f_pvalue
coefficients = optimized_model.params
p_values = optimized_model.pvalues

# Print the metrics
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
print(f"F-statistic: {f_stat}, Probability F-statistic: {prob_f_stat}")
print("Coefficients:")
print(coefficients)
print("P-values:")
print(p_values)
#%%
#D3 - Give the Mean Squared Error (MSE) of the Optimized Model Using the Training Set
X_train_optimized = X_train[['const', 'SchoolRating', 'DistanceToCityCenter', 'LocalAmenities', 'TransportAccess']]
y_train_pred = optimized_model.predict(X_train_optimized)

# Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_train_pred)

# Display the MSE
print(f"Mean Squared Error (MSE) on Training Set: {mse_train}")
#%%
#D4 - Run the Prediction on the Test Dataset using the Optimized Regression Model
X_test = test_data[['SchoolRating', 'DistanceToCityCenter', 'LocalAmenities', 'TransportAccess']]
X_test = sm.add_constant(X_test)  # Add constant for intercept
y_test = test_data[dependent_variable]

# Predict on the test set
y_test_pred = optimized_model.predict(X_test)

# Calculate Mean Squared Error (MSE) on the test set
mse_test = mean_squared_error(y_test, y_test_pred)

# Display the MSE
print(f"Mean Squared Error (MSE) on Test Set: {mse_test}")
#%%
