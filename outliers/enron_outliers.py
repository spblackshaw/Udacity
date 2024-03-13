#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
sys.path.append(os.path.abspath("../tools/"))
from tools.feature_format import featureFormat, targetFeatureSplit
import pandas as pd


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("./final_project/final_project_dataset.pkl", "rb") )
data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
df = pd.DataFrame.from_dict(data_dict, orient="index").reset_index().replace("NaN", 0)
plt.scatter(data[:, 0], data[:, 1])
plt.show()
print("here")
