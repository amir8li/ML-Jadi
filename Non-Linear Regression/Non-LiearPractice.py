import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 

df = pd.read_csv("../data/china_gdp.csv")

X = np.array(df["Year"])
y = np.array(df["Value"])
# plt.plot(X, y, "ro")
# plt.xlabel("Year")
# plt.ylabel("GDP")
# plt.show()


def sigmoid(X, a, b, c):
    y = a/(1+np.exp(-b*(X-c)))
    return y 

x_data = X/max(X)
y_data = y/max(y)

# b1 = 0.4
# b2 = 2007
# y_pred = sigmoid(x_data, b1, b2)
# plt.plot(x_data, y_pred*15*np.power(10,12))
# plt.plot(x_data,y_data, "ro")
# plt.show()

popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[1,10,0.9])
print(popt)

x = np.linspace(1960, 2015, 55)
x_ = x/max(X)
y_ = sigmoid(x_, *popt)
# print(y_)
# plt.plot(x_,y_,label='fit')
# plt.plot(x_data, y_data, 'ro', label='data')
# plt.xlabel("Year")
# plt.ylabel("GDP")
# plt.show()

r2 = r2_score(y_data, y_)
print(r2)


future_year = 2025
future_norm = future_year / max(X)

y_pred_norm = sigmoid(future_norm, *popt)
y_pred = y_pred_norm * max(y)
print(y_pred)