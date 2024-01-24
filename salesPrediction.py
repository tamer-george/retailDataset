import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import warnings

pd.set_option("display.max.columns", 500)
warnings.filterwarnings("ignore")
data = pd.read_csv("train.csv")

"""
print(data.columns)
['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
       'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Sales']

"""

column_selection = ["Order Date", 'Category', 'Product ID', 'Sub-Category', 'State', 'Ship Mode', "Sales"]

df = data[column_selection]
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")


# Feature engineering
# df["DayOfWeek"] = df["Order Date"].dt.dayofweek
df["Month"] = df["Order Date"].dt.month
# df['day_of_month'] = df['Order Date'].dt.day
df['Sales_log'] = np.log(df['Sales'])
# fig = plt.figure(figsize=(12,4))
# ax1 = fig.add_subplot(121)
# stats.probplot(df['Sales'], dist="norm", plot=ax1)
# ax1.set_title('Outliers before Log transformation')
# ax2 = fig.add_subplot(122)
# stats.probplot(df['Sales_log'],dist="norm", plot=ax2)
# ax2.set_title('Outliers after Log transformation')
# plt.show()


# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Category', 'Sub-Category', 'Product ID', 'Ship Mode', 'State'],
                            drop_first=True)

# print(df_encoded.head())

# Split data into features (X) and target variable (y)
X = df_encoded.drop(['Order Date', "Sales", 'Sales_log'], axis=1)
y = df_encoded['Sales_log']

# print(X.head())


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')


# # Example prediction for a new data point
# new_data_point = {'Product_Name_B': [1], 'Day_of_Week': [2], 'Month': [5], 'Year': [2024]}
# new_prediction = model.predict(pd.DataFrame(new_data_point))
# print(f'Predicted Sales for new data point: {new_prediction[0]}')
