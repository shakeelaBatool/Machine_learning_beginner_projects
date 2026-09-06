import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data= pd.read_csv(r'diabetes.csv')
print(data.head(5))
# Data cleaning process
# print(data.isnull().sum())
data= data.drop_duplicates()
print("************")
print(data)
print(data.info())
print(f"Data type :\n{type(data)}")
print(f'Shape of the data:\n{data.shape}')

data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

mean_=data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].mean()
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].fillna(mean_)
print(data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']])
count=0
for i in data['BMI']:
    if i==0:
        count+=1
print(count)

# Feature Selection


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = data[['Glucose', 'BMI', 'Age']] 
Y = data['Outcome'] 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("*********************")
print(X_train.shape)              # Kitne patients (rows/column) training me hain
print(X_test.shape)
print(y_train.shape)               # Training data ke outputs kitne hain
print(y_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("LogisticRegression Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

from sklearn.metrics import accuracy_score
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))


if  accuracy_score(y_test, y_pred)> accuracy_score(y_test, y_pred_knn):
    print("LogisticsRegression is better then knn") 
else:
    print(" knn is better then LogisticsRegression") 
choose=input("Press enter to visulize graphs: ")

if choose=="":

    # visulization
    plt.hist(data['Glucose'], bins=30, color='blue')
    plt.title("Graph 1")
    plt.xlabel("Glucose")
    plt.ylabel("frequency")
    plt.show()
    plt.hist(data['BMI'], bins=30, color='Blue')
    plt.title("Graph 2")
    plt.xlabel("BMI")
    plt.ylabel("frequency")
    plt.show()  
    plt.hist(data['Age'], bins=31, color='Red')
    plt.title("Graph 3")
    plt.xlabel("Age")
    plt.ylabel("frequency")
    plt.show()
    
else:
    print("End program")