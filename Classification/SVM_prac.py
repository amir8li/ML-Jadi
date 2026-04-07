import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('../data/cell_samples.csv')

df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype(int)

X = df.drop(['ID', 'Class'], axis=1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)

kernels = ['rbf', 'linear', 'poly', 'sigmoid']
accuracies = []

for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    accuracy = f1_score(y_test, y_hat, average='weighted')
    accuracies.append(accuracy)

for kernel, acc in zip(kernels, accuracies):
    print(kernel, acc)
