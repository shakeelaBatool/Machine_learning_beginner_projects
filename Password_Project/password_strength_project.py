import re            # help to recognize pattern of expression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv(r'data.csv')
# print(data.head(5))
data=data[['password', 'strength']]               # remove unwanted columns
# print(data.head(6))

# Data cleaning process
data=data.dropna()
data=data.drop_duplicates()
print(data)
print(data.shape)
def feature_extract(password):
    return [
        len(password),  # total length
        len(re.findall(r'[A-Z]', password)),  # uppercase
        len(re.findall(r'[a-z]', password)),  # lowercase
        len(re.findall(r'[0-9]', password)),  # digits
        len(re.findall(r'[^a-zA-Z0-9]', password))  # special char
    ]
X = data["password"].apply(feature_extract).tolist()          # X ----> input
y = data["strength"]                                          # y ----> output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

model = LogisticRegression(max_iter=1000)           # used for classification
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")
def predict_strength(password):
    features = feature_extract(password)
    result = model.predict([features])[0]
    
    if result == 0:
        return "Weak"
    elif result == 1:
        return "Medium"
    else:
        return "Strong"

print("Test password:", predict_strength(password=input("Enter: ")))

sns.countplot(x='strength', data=data)
plt.xticks([0,1,2], ["Weak", "Medium", "Strong"])
plt.title("Password Strength Distribution")
plt.show()
