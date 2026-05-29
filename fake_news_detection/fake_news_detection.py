# =========================
# FAKE NEWS DETECTION
# SIMPLE BEGINNER CODE
# =========================

import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# download stopwords
nltk.download('stopwords')

# load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# add labels
fake["label"] = 0
real["label"] = 1

# merge data
data = pd.concat([fake, real])

# make text lowercase
data["text"] = data["text"].str.lower()

# remove symbols and numbers
data["text"] = data["text"].apply(
    lambda x: ''.join(
        c for c in x
        if c.isalpha() or c == " "
    )
)

# remove stopwords
stop_words = stopwords.words("english")

data["text"] = data["text"].apply(
    lambda x: " ".join(
        w for w in x.split()
        if w not in stop_words
    )
)

# features and labels
x = data["text"]
y = data["label"]

# split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# convert text to numbers
vec = CountVectorizer()

x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

# train model
model = LogisticRegression()
model.fit(x_train, y_train)

# prediction
pred = model.predict(x_test)

# accuracy
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# user input
news = input("Enter news: ")

news = news.lower()

news = ''.join(
    c for c in news
    if c.isalpha() or c == " "
)

news = " ".join(
    w for w in news.split()
    if w not in stop_words
)

news_vec = vec.transform([news])

result = model.predict(news_vec)

if result[0] == 0:
    print("Fake News")
else:
    print("Real News")