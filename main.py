import pandas as pd
import proFunctions as pF
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

column_selection = ['Customer ID', 'Order ID', "Order Date", "Ship Date", "Ship Mode", "Segment", "State",
                    "Region", "Category", "Sub-Category", "Sales"]

df = data[column_selection]
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y", errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
df["DayOfWeek"] = df["Order Date"].dt.dayofweek
df["Month"] = df["Order Date"].dt.strftime("%b")
df["Year"] = df["Order Date"].dt.year
df['Quarter'] = df['Order Date'].dt.to_period('Q')
# print(df.head())
# print(df.shape)
# print(df.info())


#         --- Percentage of missing values ---

print(df.isnull().sum() * 100 / len(df))

#         --- Sum of sales per month ---

pF.sum_of_sales_per_month(df)

#         --- Analyzing Segment Trends ---

pF.segment_trend(df)

#         --- Total Sales Across Quarters ---

pF.quarter_sales(df)

#         --- Analyzing Top and Bottom Performing States ---

pF.top_performing_states(data=df)


#         --- Segment Contribution to Overall Sales ---

pF.segment_contribution_to_overall_sales(df)

#         --- Average Sales per region ---

pF.average_sales_per_region(df)


#         --- Top Performing Categories ---

pF.top_performing_categories(df)


#         --- Customer Behavior ---

pF.customer_behaviour(df)

#         --- Customer Segmentation ---

pF.customer_segmentation(df)


#         --- Shipping Method Distribution ---

pF.shipping_mode(df)


