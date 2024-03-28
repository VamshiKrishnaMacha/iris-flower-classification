# Iris Flower Classification Project

## Introduction

This project explores the fascinating world of machine learning through the lens of the Iris flower dataset, one of the most famous datasets used for classification tasks. Our objective is to build a predictive model capable of distinguishing between the three species of Iris flowers — setosa, versicolor, and virginica — based on the physical dimensions of their petals and sepals. By applying machine learning techniques, we aim to uncover the patterns that define the uniqueness of each species.

## Data Loading and Preprocessing

The dataset is loaded from a CSV file named `Iris (1).csv`. The preprocessing steps involve removing the 'Id' column, as it does not contribute to the model's ability to learn the classification task. Additionally, the categorical 'Species' column is encoded into numerical form, enabling the machine learning algorithms to process the labels.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset directly as it is in the current working directory
iris_data = pd.read_csv('Iris (1).csv')

# Drop the 'Id' column as it's not needed for modeling
iris_data.drop('Id', axis=1, inplace=True)

# Encode the 'Species' column
le = LabelEncoder()
iris_data['Species'] = le.fit_transform(iris_data['Species'])

# Split the dataset into features (X) and the target variable (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

# Initialize the k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)
