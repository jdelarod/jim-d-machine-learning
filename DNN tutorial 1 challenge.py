# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 20:52:19 2018

@author: james.delaroderie
"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt', names=["X", "Y"])
x_values = dataframe[['X']]
y_values = dataframe[['Y']]

#train model on data
challenge_reg = linear_model.LinearRegression()
challenge_reg.fit(x_values, y_values)

#error
from sklearn.metrics import r2_score
coefficient_of_determination = r2_score(y_values, challenge_reg.predict(x_values))

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, challenge_reg.predict(x_values))
plt.show()