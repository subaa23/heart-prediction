import pandas as pd
import numpy as np
df=pd.read_csv("hearts.csv")
print(df)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Age']=le.fit_transform(df['Age'])
df['Sex']=le.fit_transform(df['Sex'])
df['ChestPainType']=le.fit_transform(df['ChestPainType'])
df['RestingBP']=le.fit_transform(df['RestingBP'])
df['Cholesterol']=le.fit_transform(df['Cholesterol'])
df['FastingBS']=le.fit_transform(df['FastingBS'])
df['RestingECG']=le.fit_transform(df['RestingECG'])
df['MaxHR']=le.fit_transform(df['MaxHR'])
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
df['Oldpeak']=le.fit_transform(df['Oldpeak'])
df['ST_Slope']=le.fit_transform(df['ST_Slope'])
x=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print("x_test",x_test.shape)
print("x_train",x_train.shape)
print("y_test",x_test.shape)
print("y_train",y_train.shape)
print(df.shape)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train,y_train)
y_pred=NB.predict(x_test)
print("y_pred",y_pred)
print("y_test",y_test)
from sklearn.metrics import accuracy_score
print("ACCURACY IS",accuracy_score(y_test,y_pred))

