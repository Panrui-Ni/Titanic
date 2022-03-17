import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

Data_train = pd.read_csv("C:/pythonProject6/Data/train.csv")

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))# 两行三列小图，第(0,0)位置的图
Data_train.Survived.value_counts().plot(kind='bar')#柱状图
plt.title("Survived situation")
plt.ylabel("people counting")

plt.subplot2grid((2,3),(0,1))
Data_train.Pclass.value_counts().plot(kind='bar')
plt.title("people counting")
plt.ylabel("level of ticket")

plt.subplot2grid((2,3),(0,2))
plt.scatter(Data_train.Survived, Data_train.Age)#散点图
plt.ylabel("pc")
plt.title("Age")

plt.subplot2grid((2,3),(1,0),colspan=2)# (1,0)-(1,1)位置图
Data_train.Age[Data_train.Pclass==1].plot(kind='kde')#密度图
Data_train.Age[Data_train.Pclass==2].plot(kind='kde')
Data_train.Age[Data_train.Pclass==3].plot(kind='kde')
plt.xlabel("Age")
plt.ylabel("density")
plt.title("distri of diff ages")
plt.legend(("1", "2", "3"),loc='best')

plt.subplot2grid((2,3),(1,2))
Data_train.Embarked.value_counts().plot(kind='bar')
plt.title("pc from dif pots")
plt.ylabel("cp")

plt.show()

