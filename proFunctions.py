import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def show_values(axs, orient="v", space=.01):
    def _single(axes2):
        if orient == "v":
            for p in axes2.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.0f}'.format(p.get_height())
                axes2.text(_x, _y, value, ha="center", color='#000000', fontsize="8", fontweight="heavy")
        elif orient == "h":
            for p in axes2.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                axes2.text(_x, _y, value, ha="left", color='#000000')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def format_thousands(x, pos):
    return f"{int(x/1000)}k"


def sum_of_sales_per_month(df):
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sales_sum_by_year = df.groupby('Month')['Sales'].sum().reindex(month_order).reset_index()

    # Plotting with Seaborn line plot
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(data=sales_sum_by_year, x='Month', y='Sales', marker='o', color='skyblue')
    plt.title('Total Sales Per Month'.title(), fontweight="heavy", fontsize="16")
    plt.ylabel('Total Sales')
    plt.xlabel(None)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    sns.despine(offset=10, trim=True)
    plt.grid(False)
    plt.grid(axis="y", alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    # plt.savefig("sales_per_month", dpi=300, bbox_inches="tight")
    plt.show()


def segment_trend(df):
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Extract month from OrderDate
    df['Month'] = df['Order Date'].dt.strftime('%b')  # Full month names

    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
    segment_trends = df.groupby(['Month', 'Segment']).agg(
        AvgSpending=('Sales', 'mean'),
    ).reset_index()

    # Plotting Segment Trends
    plt.figure(figsize=(8, 5))
    segments = df['Segment'].unique()

    for segment in segments:
        segment_data = segment_trends[segment_trends['Segment'] == segment]
        plt.plot(segment_data['Month'], segment_data['AvgSpending'], label=f'Segment {segment}', marker='o')

    plt.title('Segment Trends over time'.title(), fontweight="heavy", fontsize="16")
    plt.xlabel(None)
    plt.ylabel('Average Spending')
    plt.legend()
    plt.grid(False)
    plt.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=45, ha='right')
    sns.despine(top=True, left=True)
    plt.savefig("segment_trends", dpi=300, bbox_inches="tight")
    plt.show()


def quarter_sales(df):
    # Pivot the DataFrame for easy plotting
    pivot_df = df.pivot_table(index=['Year', 'Quarter'], values='Sales', aggfunc='sum').reset_index()

    # Plotting the data
    fig, ax = plt.subplots(figsize=(8, 5))
    for year in pivot_df['Year'].unique():
        year_data = pivot_df[pivot_df['Year'] == year]
        ax.bar(year_data['Quarter'].dt.strftime('%Y-Q%q'), year_data['Sales'], label=str(year))

    # Adding labels and legend
    ax.set_ylabel('Total Sales')
    ax.set_title('Total Sales Across Quarters'.title(), fontweight="heavy", fontsize="16")
    ax.legend()
    plt.grid(False)
    show_values(ax, "v")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis="y", alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    sns.despine(top=True, left=True)
    plt.savefig("quarter_sales", dpi=300, bbox_inches="tight")
    plt.show()


def top_performing_states(data):
    top_performing_state = data.groupby('State')['Sales'].sum().nlargest(5)
    bottom_performing_states = data.groupby('State')['Sales'].sum().nsmallest(5)

    # Plotting Top Performing States
    plt.figure(figsize=(8, 5))

    # Top 5 Performing States
    plt.subplot(1, 2, 1)
    ax = top_performing_state.plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title('Top 5 Performing States'.title(), fontweight="heavy", fontsize="16")
    plt.xlabel('State')
    plt.ylabel('Total Sales')
    plt.grid(False)
    plt.xticks(rotation=45)
    sns.despine(top=True, left=True)
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    plt.xlabel(None)

    # Bottom 5 Performing States
    plt.subplot(1, 2, 2)
    bottom_performing_states.plot(kind='bar', color='coral', alpha=0.7)
    plt.title('Bottom 5 Performing States'.title(), fontweight="heavy", fontsize="16")
    plt.ylabel(None)
    plt.grid(False)
    plt.xticks(rotation=45)
    plt.xlabel(None)
    plt.tight_layout()
    sns.despine(top=True, left=True)
    plt.savefig("top_performing_states", dpi=300, bbox_inches="tight")
    plt.show()


def segment_contribution_to_overall_sales(df):
    # Segment Contribution to Overall Sales
    segment_contribution = df.groupby('Segment')['Sales'].sum() / df['Sales'].sum() * 100

    # Data Visualization: Pie chart of Segment Contribution to Overall Sales
    plt.figure(figsize=(8, 5))
    plt.pie(segment_contribution, labels=segment_contribution.index, autopct='%1.1f%%', startangle=90,
            colors=['lightcoral', 'lightblue', 'lightgreen'])
    circle = plt.Circle((0, 0), 0.45, fc='white', edgecolor='#CAC9CD')
    plt.gca().add_artist(circle)
    plt.text(0, 0, "Total Sales\n2261537", ha="center", va="center", fontsize=12)
    # Add a circle in the center (to create a donut effect)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('Segment Contribution to Overall Sales'.title(), fontweight="heavy", fontsize="16")
    plt.savefig("Segment_Contribution_to_Overall_Sales", dpi=300, bbox_inches="tight")
    plt.show()


def average_sales_per_region(df):
    average_sales_per_state = df.groupby("Region")['Sales'].mean().reset_index().sort_values(by="Sales",
                                                                                             ascending=False)
    # Data Visualization: Bar chart of Average Sales per State
    plt.figure(figsize=(8, 5))
    plt.bar(average_sales_per_state["Region"], average_sales_per_state['Sales'], color='skyblue')
    plt.title('Average Sales per region'.title(), fontweight="heavy", fontsize="16")
    plt.xlabel(None)
    plt.ylabel('AVG Sales')
    plt.grid(False)
    plt.xticks(rotation=45)
    sns.despine(top=True, left=True)
    plt.savefig("avg_sales_per_region", dpi=300, bbox_inches="tight")
    plt.show()


def top_performing_categories(df):
    # Calculate the sum of sales per category
    sales_sum_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)

    # Plotting
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=sales_sum_by_category.index, y=sales_sum_by_category.values, palette='viridis')
    plt.title('Top Performing Categories'.title(), fontweight="heavy", fontsize="16")
    plt.xlabel(None)
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Apply the custom formatter to the y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    plt.grid(False)
    sns.despine(top=True, left=True)
    show_values(ax, "v")
    plt.savefig("top_performing_categories", dpi=300, bbox_inches="tight")
    plt.show()


def customer_behaviour(data):
    customer_stats = data.groupby("Customer ID").agg(AvgSpending=("Sales", "mean"),
                                                     OrderFrequency=('Order ID', "count"),
                                                     CustomerTenure=(
                                                     "Order Date", lambda x: (x.max() - x.min()).days)).reset_index()

    plt.figure(figsize=(10, 5))

    # Average Spending
    plt.subplot(1, 3, 1)
    sns.histplot(customer_stats['AvgSpending'], bins=10, kde=True, color='skyblue')
    plt.title('Average Spending\nper Customer'.title(), fontweight="heavy", fontsize="14")
    plt.xlabel('Average Spending')
    plt.grid(False)
    sns.despine(top=True, left=True)

    # Order Frequency
    plt.subplot(1, 3, 2)
    sns.histplot(customer_stats['OrderFrequency'], bins=15, kde=True, color='lightgreen')
    plt.title('Order Frequency\nper Customer'.title(), fontweight="heavy", fontsize="14")
    plt.xlabel('Order Frequency')
    plt.grid(False)
    sns.despine(top=True, left=True)

    # Tenure (using count as a placeholder)
    plt.subplot(1, 3, 3)
    sns.histplot(customer_stats['CustomerTenure'], bins=15, kde=True, color='coral')
    plt.title('Customer Tenure'.title(), fontweight="heavy", fontsize="14")
    plt.xlabel('Tenure (count)')
    plt.grid(False)
    sns.despine(top=True, left=True)

    plt.tight_layout()
    plt.savefig("customer_behavior", dpi=300, bbox_inches="tight")
    plt.show()


def customer_segmentation(df):
    customer_segmentations = pd.qcut(df['Sales'], q=[0, 0.2, 0.8, 1], labels=['Low', 'Mid', 'High'])

    # Plotting
    plt.figure(figsize=(8, 5))

    # Customer Segmentation
    ax = customer_segmentations.value_counts().sort_index().plot(kind='bar', color='coral', alpha=0.7)
    plt.title('Customer Segmentation by Sales'.title(), fontweight="heavy", fontsize="16")
    plt.xlabel('Customer Segment')
    plt.ylabel('Number of Orders')
    plt.grid(False)
    plt.xticks(rotation=45, ha='right')
    sns.despine(top=True, left=True)
    show_values(ax, "v")
    plt.savefig("customer_segmentations", dpi=300, bbox_inches="tight")
    plt.show()


def shipping_mode(df):
    ax = sns.countplot(x='Ship Mode', data=df, order=df['Ship Mode'].value_counts().index)
    plt.title('Ship Mode Distribution'.title(), fontweight="heavy", fontsize="16")
    plt.xlabel(None)
    plt.ylabel('# of Orders')
    plt.grid(False)
    plt.xticks(rotation=45, ha='right')
    sns.despine(top=True, left=True)
    show_values(ax, "v")
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    # plt.savefig("ship_mode_distribution", dpi=300, bbox_inches="tight")
    plt.show()



