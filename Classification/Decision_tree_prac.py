import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/buy_comp.csv")
# print(df.head())

# print(df.columns)

X = df.drop(["Buys_Computer"], axis=1).values
# print(type(X))

y = df["Buys_Computer"]

le_student = LabelEncoder()
X[:,2] = le_student.fit_transform(X[:,2])

le_credit = LabelEncoder()
X[:,3] = le_credit.fit_transform(X[:,3])

# print(df['Student'].value_counts())
# print(X[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

drugTree = DecisionTreeClassifier(criterion='entropy',max_depth=4)
drugTree.fit(X_train, y_train)

y_hat = drugTree.predict(X_test)
# print(y_hat[0:5])
# print(y_test[0:5].values)

print(accuracy_score(y_hat, y_test))

plt.figure(figsize=(30,20))
plot_tree(drugTree, feature_names=df.columns[0:5], class_names=[str(c) for c in drugTree.classes_],
          filled=True, rounded=True)
plt.show()