# Supervised learning classification problem
# Task 1
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import pandas as pd

# Task 2
# importing data & splitting into training and test data
data = pd.read_csv("WhosTweeting.csv")
tweet_df = pd.DataFrame(data)
y = tweet_df.author
X_train, X_test, y_train, y_test = train_test_split(tweet_df.status, y, random_state=53, test_size=.33)

# Task 3
# Why add the parameter stop_words='english', when the project ask me to remove english stop words?
# As far as I can see the parameter stop_words='english' adds a list of english stop words
count_vectorizer = CountVectorizer(stop_words='english', min_df=0.05, max_df=0.9)
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# Why add the parameter stop_words='english', when the project ask me to remove english stop words?
# As far as I can see the parameter stop_words='english' adds a list of english stop words
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0.05, max_df=0.9)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Task 4
# Creating model and training
tfidf_nb = MultinomialNB()
tfidf_nb.fit(tfidf_train, y_train)

# Getting prediction and score
tfidf_nb_pred = tfidf_nb.predict(tfidf_test)
tfidf_nb_score = metrics.accuracy_score(tfidf_nb_pred, y_test)

# Creating model and training
count_nb = MultinomialNB()
count_nb.fit(count_train, y_train)

# Getting prediction and score
count_nb_pred = count_nb.predict(count_test)
count_nb_score = metrics.accuracy_score(count_nb_pred, y_test)

# Printing score
print('NaiveBayes Tfidf Score: ', tfidf_nb_score)
print('NaiveBayes Count Score: ', count_nb_score)

# Task 5
