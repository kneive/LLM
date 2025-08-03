import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

datapath = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(datapath + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# linear regression
model = LinearRegression()
model.fit(X,y)

X_new = [[37_655.2]]
print(model.predict(X_new))

# K nearest neighbors
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)

X_new = [[37_655.2]]
print(model.predict(X_new))