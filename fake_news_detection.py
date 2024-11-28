import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Load datasets
data_fake = pd.read_csv('../Fake-News-Detection/Datasets/datasets/Fake.csv')
data_true = pd.read_csv('../Fake-News-Detection/Datasets/datasets/True.csv')

# Assign labels
data_fake["class"] = 0
data_true["class"] = 1

# Remove last 10 rows for manual testing
data_fake_manual_testing = data_fake.tail(10)
data_fake = data_fake.iloc[:-10]

data_true_manual_testing = data_true.tail(10)
data_true = data_true.iloc[:-10]

# Combine and shuffle data
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1).sample(frac=1).reset_index(drop=True)

# Text preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

# Train-test split
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, pred_lr))

# # Decision Tree Classifier
# DT = DecisionTreeClassifier()
# DT.fit(xv_train, y_train)
# pred_dt = DT.predict(xv_test)
# print("\nDecision Tree Classification Report:")
# print(classification_report(y_test, pred_dt))

# # Gradient Boosting Classifier
# GB = GradientBoostingClassifier(random_state=0)
# GB.fit(xv_train, y_train)
# pred_gb = GB.predict(xv_test)
# print("\nGradient Boosting Classification Report:")
# print(classification_report(y_test, pred_gb))

# # Random Forest Classifier
# RF = RandomForestClassifier(random_state=0)
# RF.fit(xv_train, y_train)
# pred_rf = RF.predict(xv_test)
# print("\nRandom Forest Classification Report:")
# print(classification_report(y_test, pred_rf))

# Manual Testing
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test["text"])
    pred_LR = LR.predict(new_xv_test)
    #pred_DT = DT.predict(new_xv_test)
    #pred_GB = GB.predict(new_xv_test)
    #pred_RF = RF.predict(new_xv_test)

    print("\nManual Testing Predictions:")
    print(f"Logistic Regression: {output_label(pred_LR[0])}")
    #print(f"Decision Tree: {output_label(pred_DT[0])}")
    #print(f"Gradient Boosting: {output_label(pred_GB[0])}")
    #print(f"Random Forest: {output_label(pred_RF[0])}")

if __name__ == "__main__":
    while True:
        news = input("\nEnter news text to classify (or type 'exit' to quit): ")
        if news.lower() == 'exit':
            print("Exiting...")
            break
        manual_testing(news)
