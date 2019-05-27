import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, accuracy_score


class Models:
    def __init__(self, input_filepath):
        self.df = pd.read_csv(input_filepath)
        self.data = np.genfromtxt(input_filepath, delimiter=',', skip_header=True)

    def preprocessing(self):
        features = self.df.iloc[:, 2:4].values
        ground_truth= self.df.iloc[:, 1].values
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        self.train, self.test, self.train_labels, self.test_labels = train_test_split(features, ground_truth, test_size=0.3, random_state=42)

    def SGDClassifier(self):
        self.sg = SGDClassifier(random_state=42)
        self.sg.fit(self.train, self.train_labels)
        pred = self.sg.predict(self.test)
        print(classification_report(self.test_labels, pred))
        print(accuracy_score(self.test_labels, pred))

    def GaussianNB(self):
        self.gnb = GaussianNB()
        self.gnb.fit(self.train, self.train_labels)
        pred = self.gnb.predict(self.test)
        print(pred)
        print(classification_report(self.test_labels, pred))
        print(accuracy_score(self.test_labels, pred))

    def quickresult_GaussianNB(self):
        testlog = open(r"C:/Users/Gebruiker/Documents/HexClassifier/data/processed/testlog_Gauss.csv", 'w')

        pred = self.gnb.predict(self.df.iloc[:, 2:4].values)
        for i in range(len(self.data)):
            testlog.write('{}, {}, {}\n'.format(i, self.data[i, 1], pred[i]))
        testlog.close()

    def quickresult_SGD(self):
        testlog = open(r"C:/Users/Gebruiker/Documents/HexClassifier/data/processed/testlog_SGD.csv", 'w')

        pred = self.sg.predict(self.df.iloc[:, 2:4].values)
        for i in range(len(self.data)):
            testlog.write('{}, {}, {}\n'.format(i, self.data[i, 1], pred[i]))
        testlog.close()


if __name__ == '__main__':
    input_path = r"C:/Users/Gebruiker/Documents/HexClassifier/data/interim/features_stitched/features.csv"

    M = Models(input_path)
    M.preprocessing()
    M.GaussianNB()
    M.quickresult_GaussianNB()
    M.SGDClassifier()
    M.quickresult_SGD()
