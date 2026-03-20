import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("../data/FuelConsumption.csv")
cdf = df[["ENGINESIZE", "CYLINDERS", 
          "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]

# cdf = cdf[cdf["ENGINESIZE"] < 8]

np.random.seed(42)
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# pd.plotting.scatter_matrix(train, figsize=(10,10))
# plt.show()

train_x = np.array(train[["ENGINESIZE"]])
test_x = np.array(test[["ENGINESIZE"]])
train_y = np.array(train[["CO2EMISSIONS"]])
test_y = np.array(test[["CO2EMISSIONS"]])

# print(train_x[:3])
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
# print(train_x_poly)

regr = linear_model.LinearRegression()
regr.fit(train_x_poly, train_y)

# print(regr.coef_)
# print(regr.intercept_)

plt.scatter(train_x, train_y, color="blue")
xx = np.arange(0, 10, 0.1)
yy = regr.intercept_[0] + regr.coef_[0][1]*xx + regr.coef_[0][2]*np.power(xx,2) + regr.coef_[0][3]*np.power(xx,3)
plt.plot(xx, yy, '--r')
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 EMISSIONS")
plt.show()


test_x_poly = poly.fit_transform(test_x)
y_predicted = regr.predict(test_x_poly) 

print(f"R2: {r2_score(test_y, y_predicted): .2f}")


def predict_model(x):
    return regr.intercept_[0] + regr.coef_[0][1]*x + regr.coef_[0][2]*np.power(x,2) + regr.coef_[0][3]*np.power(x,3)

a_to_predict =  5.7
print(predict_model(a_to_predict))