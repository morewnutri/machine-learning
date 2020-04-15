#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:13:21 2020

@author: wuhaitao
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('~/Desktop/random/train.csv',index_col=0,engine='python')
data=df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked']]
#更改分类变量对应的值
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
#同理，更改Embarked对应的值
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
te=data[data['Embarked'].notnull()]#非空的embarked对应的行
te_X=te[['Survived','Pclass','Sex','SibSp','Parch','Fare']]#设定输入的X
te_Y=te[['Embarked']]#设定输入的Y
te_X=te_X.astype(float)#转换数据类型，不转换成数值型的，到后面输入模型会报错。
te_Y=te_Y.astype(float)#转换数据类型，不转换成数值型的，到后面输入模型会报错。
tr=data[data['Embarked'].isnull()]
tr_X=tr[['Survived','Pclass','Sex','SibSp','Parch','Fare']].astype(float)
tr_Y=tr['Embarked'].astype(float)
fc=RandomForestClassifier()
fc.fit(te_X,te_Y)
pr=fc.predict(tr_X)

data[data['Embarked'].isnull(),'Embarked']=pr#将预测的缺失值补充到原来的缺失的位置