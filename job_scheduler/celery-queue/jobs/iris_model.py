from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class IrisModel():

    def load_data(self):
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns = iris.feature_names)
        df['target'] = iris.target
        targets = iris.target_names
        X_train = df.drop(['target'], axis=1)
        y_train = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.33, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def fit(self):
        X_train, X_test, y_train, y_test = self.load_data()
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)
