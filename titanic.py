from sklearn.ensemble import RandomForestRegressor ##缺失值处理
from sklearn import linear_model ## 线性回归
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
import sklearn.preprocessing as preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

Data_train = pd.read_csv("C:/pythonProject6/Data/train.csv")

def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']] #df中所有数值型特征

    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values # 缺失值

    y = known_age[:, 0]# y为年龄值所在列
    x = known_age[:, 1:]# x为其他特征属性值

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1) # 固定随机分支的方法，2000棵树，全部内核处理

    rfr.fit(x,y) # 将已知数据 fit 到随机森林中

    predictAges = rfr.predict(unknown_age[:, 1::]) #输入年龄缺失的人的其他属性值，预测年龄

    df.loc[(df.Age.isnull()), 'Age'] = predictAges # 将预测的值填回 df

    return df, rfr



def set_Cabin_type(df): #将 Cabin 是否缺失作为新的属性
    df.loc[(df.Cabin.notnull()), 'Cabin']='Y'
    df.loc[(df.Cabin.isnull()),'Cabin']='N'
    return df


Data_train, rfr = set_missing_ages(Data_train) # 输入 Data_train，令 Data_train, rfr 等于返回的 df, rfr，即补上缺失的年龄

Data_train = set_Cabin_type(Data_train) # 输入 Data_train，输出 Cabin 是否为空


dummies_Cabin = pd.get_dummies(Data_train['Cabin'], prefix='Cabin')# 由于回归预测需要数值型数据，将 Cabin 的取值 Y 和 N 转化为(1,0)-(0,1)哑变量，列前缀为 Cabin
dummies_Embarked = pd.get_dummies(Data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(Data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(Data_train['Pclass'], prefix='Pclass') ## 同理把 Embarked，Sex，Pclass 转化为哑变量

# 处理人名里面的Title
Data_train['Titile'] = Data_train.Name.str.extract('([A-Za-z]+)\.', expand = False) # 将 Name 里'.'前面的 Title 提取出来
Data_train['Titile'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)
#把前面一堆出现的字段替换成 Rare
Data_train['Titile'].replace('Mlle', 'Miss', inplace=True)
Data_train['Titile'].replace('Ms', 'Miss', inplace=True)
Data_train['Titile'].replace('Mme', 'Mrs', inplace=True)
dummies_Title = pd.get_dummies(Data_train['Titile'], prefix='Title')


df = pd.concat([Data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Title], axis=1) #把得到的数据按列拼接（按行 axis=0）
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)# 去掉已经处理的和非数值的信息


# 归一化！！
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))# 将 Age 归一化
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param) # 不加 reshape 输出的是一维数值，加了输出的是列向量
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

### 以上为数据处理

# 以下用 sklearn 逻辑回归

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')# 通过 filter 取出预处理后的值
train_np = train_df.values #转化为矩阵

y=train_np[:, 0]

x=train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)# 惩罚项为2-范数，系数为1，迭代误差为1e-6

clf.fit(x,y)
clf

#下面开始处理测试集数据

Data_test = pd.read_csv("C:/pythonProject6/Data/test.csv")

Data_test.loc[(Data_test.Fare.isnull()), 'Fare'] = 0 # 将 Fare 空白处填为0

tmp_df = Data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[Data_test.Age.isnull()].values #将 Age 缺失的人的信息做成矩阵
x = null_age[:, 1:]
predictedAges = rfr.predict(x) # 用之前生成的随机森林，基于 null_age 除 Age 以外的信息预测 Age 的值
Data_test.loc[(Data_test.Age.isnull()), 'Age'] = predictedAges

Data_test = set_Cabin_type(Data_test) # 调用之前的 set...函数，处理 Cabin 列
dummies_Cabin = pd.get_dummies(Data_test['Cabin'], prefix='Cabin') #依然把 Cabin 换成哑变量
dummies_Embarked = pd.get_dummies(Data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(Data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(Data_test['Pclass'], prefix='Pclass')


Data_test.loc[(Data_test.Name.isnull()), 'Name'] = 'N.'
Data_test['Titile'] = Data_test.Name.str.extract('([A-Za-z]+)\.', expand = False) # 将 Name 里'.'前面的 Title 提取出来
Data_test['Titile'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'N'], 'Rare', inplace=True)
#把前面一堆出现的字段替换成 Rare
Data_test['Titile'].replace('Mlle', 'Miss', inplace=True)
Data_test['Titile'].replace('Ms', 'Miss', inplace=True)
Data_test['Titile'].replace('Mme', 'Mrs', inplace=True)
dummies_Title = pd.get_dummies(Data_test['Titile'], prefix='Title')
#print(dummies_tit)

df_test = pd.concat([Data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Title], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
age_scale_param = scaler.fit(df_test['Age'].values.reshape(-1,1))# 将 Age 归一化
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df_test['Fare'].values.reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

# 下面开始测试
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = test.values #将基本信息转化为矩阵
predictions = clf.predict(x)# 输入逻辑回归函数
result = pd.DataFrame({'PassengerId': Data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)}) #复制 Data_test 'PassengerId' 列，'Survived'列为 predictions
#result.to_csv("C:/pythonProject6/Data/submission.csv") #生成结果文件


##优化
# 以训练集做交叉验证
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
all_Data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = all_Data.values[:,1:]
y = all_Data.values[:,0]
#print(cross_val_score(clf,x,y,cv=10)) #打分，用 df 中的数据进行 5 折交叉验证

# 按训练数据：验证数据=7:3 查找坏数据
split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')#将训练样本中的数据提取出来
#clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
clf.fit(train_df.values[:,1:], train_df.values[:,0])

#开始预测
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
cv_df_va = cv_df.values
pre = clf.predict(cv_df_va[:, 1:]) # 通过 cv 集上的信息预测
bad_cases = Data_train.loc[Data_train['PassengerId'].isin(split_cv[pre != cv_df_va[:,0]]['PassengerId'].values)]
#print(bad_cases)


# Bagging 集成学习
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
train_np = train_df.values

y = train_np[:, 0]
x = train_np[:, 1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
# 以逻辑回归作为基学习器，20个学习器，抽0.8比例的样本作为训练集，抽1.0比例的特征样本训练，bootstrap=True采用自助抽样，特征集不放回抽样，全部内核处理
bagging_clf.fit(x,y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = test.values #将基本信息转化为矩阵
predictions = bagging_clf.predict(x)# 输入逻辑回归函数
result = pd.DataFrame({'PassengerId': Data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)}) #复制 Data_test 'PassengerId' 列，'Survived'列为 predictions
result.to_csv("C:/pythonProject6/Data/submission.csv") #生成结果文件

# 再次交叉验证
all_Data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = all_Data.values[:,1:]
y = all_Data.values[:,0]
#print(cross_val_score(bagging_clf,x,y,cv=10)) #打分，用 df 中的数据进行 5 折交叉验证



# XGBoost 预测
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
train_np = train_df.values

y = train_np[:, 0]
x = train_np[:, 1:]
xgcl = XGBClassifier(n_estimators=20, max_depth=5, n_jobs=-1)
xgcl.fit(x,y,eval_metric=['logloss', 'auc', 'error'])

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = test.values #将基本信息转化为矩阵
predictions = xgcl.predict(x)# 输入逻辑回归函数
result = pd.DataFrame({'PassengerId': Data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)}) #复制 Data_test 'PassengerId' 列，'Survived'列为 predictions
#result.to_csv("C:/pythonProject6/Data/submission.csv") #生成结果文件

# 再次交叉验证
all_Data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = all_Data.values[:,1:]
y = all_Data.values[:,0]
#print(cross_val_score(xgcl,x,y,cv=10)) #打分，用 df 中的数据进行 5 折交叉验证

# 随机森林预测
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
train_np = train_df.values

y = train_np[:, 0]
x = train_np[:, 1:]
rfc = RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=-1)
rfc.fit(x,y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = test.values #将基本信息转化为矩阵
predictions = rfc.predict(x)# 输入逻辑回归函数
result = pd.DataFrame({'PassengerId': Data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)}) #复制 Data_test 'PassengerId' 列，'Survived'列为 predictions
#result.to_csv("C:/pythonProject6/Data/submission.csv") #生成结果文件

# 再次交叉验证
all_Data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = all_Data.values[:,1:]
y = all_Data.values[:,0]
#print(cross_val_score(rfc,x,y,cv=10)) #打分，用 df 中的数据进行 5 折交叉验证

# 以下用 sklearn 支持向量机

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')# 通过 filter 取出预处理后的值
train_np = train_df.values #转化为矩阵

y=train_np[:, 0]

x=train_np[:, 1:]

rfc = RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=-1)
bagging_clf = BaggingRegressor(rfc, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
# 以逻辑回归作为基学习器，20个学习器，抽0.8比例的样本作为训练集，抽1.0比例的特征样本训练，bootstrap=True采用自助抽样，特征集不放回抽样，全部内核处理
bagging_clf.fit(x,y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = test.values # 将基本信息转化为矩阵
predictions = bagging_clf.predict(x) # 输入逻辑回归函数
result = pd.DataFrame({'PassengerId': Data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)}) # 复制 Data_test 'PassengerId' 列，'Survived'列为 predictions
#result.to_csv("C:/pythonProject6/Data/submission.csv") # 生成结果文件

all_Data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
x = all_Data.values[:,1:]
y = all_Data.values[:,0]
#print(cross_val_score(bagging_clf,x,y,cv=10)) #打分，用 df 中的数据进行 5 折交叉验证