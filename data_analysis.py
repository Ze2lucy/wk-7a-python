import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
df = pd.read_csv(url, header=None)

# Set column names for the dataset
df.columns = ['sepal_length', 'sepal_width',
              'petal_length', 'petal_width', 'species']

# Display the first few rows to inspect the data
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nData types and missing values:")
print(df.dtypes)
print(df.isnull().sum())

# Clean the dataset if necessary (fill missing values with the mean)
# Replace missing values in numeric columns only
df[df.select_dtypes(include='number').columns] = df.select_dtypes(
    include='number').fillna(df.mean(numeric_only=True))


# Task 2: Basic Data Analysis
# Compute basic statistics of the numerical columns
print("\nBasic statistics of the numerical columns:")
print(df.describe())

# Perform groupings by 'species' and compute the mean for each group
grouped = df.groupby('species').mean()
print("\nAverage measurements by species:")
print(grouped)

# Task 3: Data Visualization

# 1. Line chart showing trends over time (Not available in the Iris dataset, so this will be skipped)

# 2. Bar chart comparing the average sepal length by species
plt.figure(figsize=(8, 6))
df.groupby('species')['sepal_length'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Histogram of sepal length distribution
plt.figure(figsize=(8, 6))
df['sepal_length'].plot(kind='hist', bins=20,
                        color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot to visualize the relationship between sepal length and petal length
plt.figure(figsize=(8, 6))
plt.scatter(df['sepal_length'], df['petal_length'], color='purple', alpha=0.5)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.tight_layout()
plt.show()

# Observations and findings:
# You can manually add your observations in the script or in a separate report.
# For example:
# - "The average sepal length is highest in the Iris-virginica species."
# - "The distribution of sepal lengths is slightly skewed to the right."
# - "There is a positive correlation between sepal length and petal length, suggesting that larger flowers tend to have larger petals."
