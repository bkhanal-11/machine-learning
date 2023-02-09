import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 

from linear_regression import LinearRegression as LR

def load_dataset(filepath):
    # Importing the dataset
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[:, :-1].values # get a copy of dataset exclude last column
    y = dataset.iloc[:, 1].values # get array of dataset in column 1st

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

    return X_train, X_test, y_train, y_test

def main():
    # Loading the data 
    lr = LR()
    X_train, X_test, y_train, y_test = load_dataset('datasets/salary_data.csv')
    lr.fit(X_train, y_train)

    # Visualizing the Training set results
    viz_train = plt
    viz_train.scatter(X_train, y_train, color='red')
    viz_train.plot(X_train, lr.predict(X_train), color='blue')
    viz_train.title('Salary VS Experience (Training set)')
    viz_train.xlabel('Year of Experience')
    viz_train.ylabel('Salary')
    viz_train.savefig('assets/train.png')
    viz_train.show()

    # Visualizing the Test set results
    viz_test = plt
    viz_test.scatter(X_test, y_test, color='red')
    viz_test.plot(X_train, lr.predict(X_train), color='blue')
    viz_test.title('Salary VS Experience (Test set)')
    viz_test.xlabel('Year of Experience')
    viz_test.ylabel('Salary')
    viz_test.savefig('assets/test.png')
    viz_test.show()


if __name__ == "__main__":
    main()