import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import model_selection, linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("../../data/study_exam_1000.csv")

# pd.plotting.scatter_matrix(df, figsize=(10,10))
# plt.show()

# recognized the linear relation btw study hours and exam score

X = np.array(df[["study_hours"]])
y = np.array(df[["exam_score"]])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# print(regr.coef_)
# print(regr.intercept_)

# plt.scatter(X_train, y_train, color="b")
# plt.plot(X_train, X_train*regr.coef_[0] + regr.intercept_, '-r')
# plt.xlabel("study hours")
# plt.ylabel("exam score")
# plt.show()

predicts = regr.predict(X_test)

print(f"MAE: {np.mean(np.absolute(predicts - y_test)):.2f}")
print(f"MSE: {np.mean((predicts - y_test)**2):.2f}")
print(f"R2: {r2_score(y_test, predicts)}")