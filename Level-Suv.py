import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

Data_train = pd.read_csv("C:/pythonProject6/Data/train.csv")

fig = plt.figure()
fig.set(alpha=0.2)



S_0 = Data_train.Pclass[Data_train.Survived ==0].value_counts()
S_1 = Data_train.Pclass[Data_train.Survived ==1].value_counts()
df = pd.DataFrame({'Y':S_1, 'N':S_0})
df.plot(kind='bar', stacked=True)
plt.title("Level-YN")
plt.xlabel("Level")
plt.ylabel("number")
plt.show()