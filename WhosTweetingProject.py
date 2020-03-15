# Supervised learning classification problem
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import pandas as pd

# importing data & splitting into training and test data
data = pd.read_csv("WhosTweeting.csv")
tweet_df = pd.DataFrame(data)
y = tweet_df.author
X_train, X_test, y_train, y_test = train_test_split(tweet_df.status, y, random_state=53, test_size=.33)

count_vectorizer = CountVectorizer(min_df=0.05, max_df=0.9)
count_train = count_vectorizer.fit_transform()
count_test = count_vectorizer.transform()

tfidf_vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.9)
tfidf_train = tfidf_vectorizer.fit_transform()
tfidf_test = tfidf_vectorizer.transform()