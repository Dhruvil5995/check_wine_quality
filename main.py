import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class WineQualityClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.dataset_path)

    def explore_data(self):
        print("Number of rows & columns:", self.data.shape)

        # Display first 5 rows
        print(self.data.head())

        # Check for missing values
        print(self.data.isnull().sum())

        # Statistical measures of the dataset
        print(self.data.describe())

        # Plot number of values for each quality
        sns.catplot(x='quality', data=self.data, kind='count')
        plt.title("Number of values for each quality")
        plt.show()

        # Plot volatile acidity vs Quality
        self._plot_barplot('quality', 'volatile acidity')

        # Plot citric acid vs Quality
        self._plot_barplot('quality', 'citric acid')

        # Plot heatmap of correlations between columns
        self._plot_heatmap()

    def _plot_barplot(self, x_col, y_col):
        plot = plt.figure(figsize=(5, 5))
        sns.barplot(x=x_col, y=y_col, data=self.data)
        plt.show()

    def _plot_heatmap(self):
        correlation = self.data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
        plt.show()

    def preprocess_data(self):
        # Separate the data and labels
        x = self.data.drop('quality', axis=1)
        y = self.data['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

        # Split data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=3)

        return X_train, X_test, Y_train, Y_test

    def train_model(self, X_train, Y_train):
        self.model = RandomForestClassifier()
        self.model.fit(X_train, Y_train)

    def evaluate_model(self, X_test, Y_test):
        X_test_prediction = self.model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        print('Accuracy:', test_data_accuracy)

    def predict_quality(self, input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = self.model.predict(input_data_reshaped)

        if prediction[0] == 1:
            print('Good Quality Wine')
        else:
            print('Bad Quality Wine')


if __name__ == '__main__':
    wine_classifier = WineQualityClassifier("E:\\python_projects_CV\\wine quality\\winequality-red.csv")
    wine_classifier.load_data()
    wine_classifier.explore_data()
    X_train, X_test, Y_train, Y_test = wine_classifier.preprocess_data()
    wine_classifier.train_model(X_train, Y_train)
    wine_classifier.evaluate_model(X_test, Y_test)
    wine_classifier.predict_quality((7.3,	0.65,	0	,1.2,	0.065,	15	,21	,0.9946	,3.39,	0.47,	10))
