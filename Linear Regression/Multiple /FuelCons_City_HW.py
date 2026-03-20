import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumption.csv")
cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY","CO2EMISSIONS"]]

# pd.plotting.scatter_matrix(cdf, figsize=(10,10))
# plt.show()
X = np.array(cdf.drop(['CO2EMISSIONS'], axis=1))
y = np.array(cdf['CO2EMISSIONS'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[mask]
# test = cdf[~msk]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# print(regr.coef_)

y_hat = regr.predict(X_test)
print(r2_score(y_test, y_hat))


