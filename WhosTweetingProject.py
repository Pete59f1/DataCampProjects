# Supervised learning classification problem
# Task 1
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import pandas as pd
from pprint import pprint
from WhosTweetingHelper_functions import plot_confusion_matrix
from WhosTweetingHelper_functions import plot_and_return_top_features

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
tfidf_nb_cm = metrics.confusion_matrix(y_test, tfidf_nb_pred, labels=['Donald J. Trump', 'Justin Trudeau'])
count_nb_cm = metrics.confusion_matrix(y_test, count_nb_pred, labels=['Donald J. Trump', 'Justin Trudeau'])

plot_confusion_matrix(tfidf_nb_cm, classes=['Donald J. Trump', 'Justin Trudeau'], title="TF-IDF NB Confusion Matrix")
plot_confusion_matrix(count_nb_cm, classes=['Donald J. Trump', 'Justin Trudeau'], title="Count NB Confusion Matrix", figure=1)

# Task 6
# Creating and training model
tfidf_svc = LinearSVC()
tfidf_svc.fit(tfidf_train, y_train)

# Predicting and getting score
tfidf_svc_pred = tfidf_svc.predict(tfidf_test)
tfidf_svc_score = metrics.accuracy_score(tfidf_svc_pred, y_test)

print("LinearSVC Score:   %0.3f" % tfidf_svc_score)

svc_cm = metrics.confusion_matrix(y_test, tfidf_svc_pred, labels=['Donald J. Trump', 'Justin Trudeau'])
plot_confusion_matrix(svc_cm, classes=['Donald J. Trump', 'Justin Trudeau'], title="TF-IDF LinearSVC Confusion Matrix")

# Task 7
top_features = plot_and_return_top_features(tfidf_svc, tfidf_vectorizer)
pprint(top_features)

# Task 8
trump_tweet = "Build the wall"
trudeau_tweet = "Now its Canada"

trump_tweet_vectorized = tfidf_vectorizer.transform([trump_tweet])
trudeau_tweet_vectorized = tfidf_vectorizer.transform([trudeau_tweet])

trump_tweet_pred = tfidf_svc.predict(trump_tweet_vectorized)
trudeau_tweet_pred = tfidf_svc.predict(trudeau_tweet_vectorized)

print("Predicted Trump tweet", trump_tweet_pred)
print("Predicted Trudeau tweet", trudeau_tweet_pred)
