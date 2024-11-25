import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#NLP Text Libraries
import nltk
nltk.download('punkt_tab', force=True)
nltk.download("punkt", force=True)
nltk.download("stopwords", force=True)
nltk.download("wordnet", force=True)
import joblib
import pandas as pd
import numpy as npcle
import matplotlib.pyplot as plt
import string
import re
import nltk.corpus
from nltk.stem import WordNetLemmatizer

"""**EDA Analysis**"""

# Text Polarity
from textblob import TextBlob

# Text Vectorizer
from sklearn.feature_extraction.text import CountVectorizer


"""**Feature Engineering**"""

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Resampling
from imblearn.over_sampling import SMOTE
from collections import Counter

# Splitting Dataset
from sklearn.model_selection import train_test_split

"""**Model Selection and Evaluation**"""

# Model Building
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Model Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Dataset

dataset = pd.read_csv("Instruments_Reviews.csv")

"""**Shape of The Dataset**"""

dataset.shape

#From this, we can infer that the dataset consists of 10261 rows and 9 columns.*
# Data Preprocessing

#Checking Null Values**
dataset.isnull().sum()

#Filling Missing Values

dataset['reviewText'] = dataset['reviewText'].fillna("").astype(str)

#Concatenate `reviewText` and `summary` Columns

dataset["reviews"] = dataset["reviewText"] + " " + dataset["summary"]
dataset.drop(columns = ["reviewText", "summary"], axis = 1, inplace = True)

#*Statistic Description of The Dataset

dataset.describe(include = "all")



#Percentages of Ratings Given from The Customers**


dataset.overall.value_counts().plot(kind = "pie", legend = False, autopct = "%1.2f%%", fontsize = 10, figsize=(8,8))
plt.title("Percentages of Ratings Given from The Customers", loc = "center")
plt.show()

#Labelling Products Based On Ratings Given

def Labelling(Rows):
  if(Rows["overall"] > 3.0):
    Label = "Positive"
  elif(Rows["overall"] < 3.0):
    Label = "Negative"
  else:
    Label = "Neutral"
  return Label

dataset["sentiment"] = dataset.apply(Labelling, axis = 1)

dataset["sentiment"].value_counts().plot(kind = "bar", color = "blue")
plt.title("Amount of Each Sentiments Based On Rating Given", loc = "center", fontsize = 15, color = "red", pad = 25)
plt.xlabel("Sentiments", color = "green", fontsize = 10, labelpad = 15)
plt.xticks(rotation = 0)
plt.ylabel("Amount of Sentiments", color = "green", fontsize = 10, labelpad = 15)
plt.show()



# Text Preprocessing
#Text Cleaning


def Text_Cleaning(Text):
  if not isinstance(Text, str):
    return ""
  # Lowercase the texts
  Text = Text.lower()

  # Cleaning punctuations in the text
  punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  Text = Text.translate(punc)

  # Removing numbers in the text
  Text = re.sub(r'\d+', '', Text)

  # Remove possible links
  Text = re.sub('https?://\S+|www\.\S+', '', Text)

  # Deleting newlines
  Text = re.sub('\n', '', Text)

  return Text

#Text Processing

# Stopwords
Stopwords = set(nltk.corpus.stopwords.words("english")) - set(["not"])

def Text_Processing(Text):
  Processed_Text = list()
  Lemmatizer = WordNetLemmatizer()

  # Tokens of Words
  Tokens = nltk.word_tokenize(Text)

  # Removing Stopwords and Lemmatizing Words
  # To reduce noises in our dataset, also to keep it simple and still 
  # powerful, we will only omit the word `not` from the list of stopwords

  for word in Tokens:
    if word not in Stopwords:
      Processed_Text.append(Lemmatizer.lemmatize(word))

  return(" ".join(Processed_Text))

"""**Applying The Functions**"""

dataset["reviews"] = dataset["reviews"].apply(lambda Text: Text_Cleaning(Text))
dataset["reviews"] = dataset["reviews"].apply(lambda Text: Text_Processing(Text))

"""

# Exploratory Data Analysis

**Overview of The Dataset**
"""

dataset.head(n = 10)

#About Other Features

dataset.describe(include = "all")

#Polarity, Review Length, and Word Counts**


dataset["polarity"] = dataset["reviews"].map(lambda Text: TextBlob(Text).sentiment.polarity)

dataset["polarity"].plot(kind = "hist", bins = 40, edgecolor = "blue", linewidth = 1, color = "orange", figsize = (10,5))
plt.title("Polarity Score in Reviews", color = "blue", pad = 20)
plt.xlabel("Polarity", labelpad = 15, color = "red")
plt.ylabel("Amount of Reviews", labelpad = 20, color = "green")

plt.show()

#Review Length
dataset["length"] = dataset["reviews"].astype(str).apply(len)

dataset["length"].plot(kind = "hist", bins = 40, edgecolor = "blue", linewidth = 1, color = "orange", figsize = (10,5))
plt.title("Length of Reviews", color = "blue", pad = 20)
plt.xlabel("Length", labelpad = 15, color = "red")
plt.ylabel("Amount of Reviews", labelpad = 20, color = "green")

plt.show()


#Word Counts

dataset["word_counts"] = dataset["reviews"].apply(lambda x: len(str(x).split()))

dataset["word_counts"].plot(kind = "hist", bins = 40, edgecolor = "blue", linewidth = 1, color = "orange", figsize = (10,5))
plt.title("Word Counts in Reviews", color = "blue", pad = 20)
plt.xlabel("Word Counts", labelpad = 15, color = "red")
plt.ylabel("Amount of Reviews", labelpad = 20, color = "green")

plt.show()

#Encoding Our Target Variable

Encoder = LabelEncoder()
dataset["sentiment"] = Encoder.fit_transform(dataset["sentiment"])

dataset["sentiment"].value_counts()

#TF-IDF Vectorizer

# Defining our vectorizer with total words of 5000 and with bigram model
TF_IDF = TfidfVectorizer(max_features = 5000, ngram_range = (2, 2))

# Fitting and transforming our reviews into a matrix of weighed words
# This will be our independent features
X = TF_IDF.fit_transform(dataset["reviews"])

# Check our matrix shape
X.shape

# Declaring our target variable
y = dataset["sentiment"]


#Resampling Our Dataset

Counter(y)

Balancer = SMOTE(random_state = 42)
X_final, y_final = Balancer.fit_resample(X, y)

Counter(y_final)

#Splitting Our Dataset
print("X_final shape:", X_final.shape)
print("y_final shape:", y_final.shape)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Model Selection and Evaluation
#Model Building**


#A Decision Tree Classifier is a supervised machine learning algorithm used for classification tasks. 
# It predicts the class label of a given input by learning simple decision rules inferred from the training data.
DTree = DecisionTreeClassifier()

#Linear Regression is a supervised learning algorithm used to predict a continuous target variable based on one or more input features.
#  It assumes a linear relationship between the dependent variable (target) and the independent variables (features).
LogReg = LogisticRegression()

Models = [DTree, LogReg]
Models_Dict = {0: "Decision Tree", 1: "Logistic Regression", 2: "SVC", 3: "Random Forest", 4: "Naive Bayes", 5: "K-Neighbors"}

for i, model in enumerate(Models):
  print("{} Test Accuracy: {}".format(Models_Dict[i], cross_val_score(model, X, y, cv = 10, scoring = "accuracy").mean()))

Classifier = LogisticRegression(random_state = 42, C = 6866.488450042998, penalty = 'l2', max_iter=1000)

# Train the model using the training data
Classifier.fit(X_train, y_train)

joblib.dump(Classifier, 'logistic_regression_model.pkl')  # Save your model
joblib.dump(TF_IDF, 'tfidf_vectorizer.pkl')  # Save your vectorizer
# Make predictions on the test set
Prediction = Classifier.predict(X_test)

# Evaluate the model

# Accuracy Score
accuracy = accuracy_score(y_test, Prediction)
print("Accuracy on Test Set: {:.2f}%".format(accuracy * 100))

# Classification Report (Precision, Recall, F1-score, etc.)
print("\nClassification Report:")
print(classification_report(y_test, Prediction))

