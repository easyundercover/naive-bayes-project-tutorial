#Set up workspace
import pandas as pd
import numpy as np
import sys
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

#Import data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

#Clean data
df_trans = df_raw.drop("package_name", axis = 1)
df_trans['review'] = df_trans['review'].str.lower()
df_trans['review'] = df_trans['review'].str.strip() 

stop = stopwords.words('english')
def remove_stopwords(review):
  if review is not None:
    words = review.strip().split()
    words_filtered = []
    for word in words:
      if word not in stop:
        words_filtered.append(word)
    result = " ".join(words_filtered) 
  else:
      result = None
  return result

  df_trans['review'] = df_trans['review'].apply(remove_stopwords)

  #Copy df into a new one
  df = df_trans.copy()

  #Split data
  X = df['review']
y = df['polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 25)

#Create pipeline and models
clf_1 = Pipeline([('cont_vect', CountVectorizer()), ('clf', MultinomialNB())])
clf_1.fit(X_train, y_train)
pred_1 = clf_1.predict(X_test)

clf_2 = Pipeline([('tfidf_vect', TfidfVectorizer()), ('clf', MultinomialNB())])
clf_2.fit(X_train, y_train)
pred_2 = clf_2.predict(X_test)

clf_3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
clf_3.fit(X_train, y_train)
pred_3 = clf_3.predict(X_test)

#Hyperparameter tuning
n_iter_search = 4
parameters = {'cont_vect__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': (1e-2, 1e-3)}
gs_clf_1 = RandomizedSearchCV(clf_1, parameters, n_iter = n_iter_search)
gs_clf_1.fit(X_train, y_train)
pred_1_grid = gs_clf_1.predict(X_test)

n_iter_search = 2
parameters = {'clf__alpha': (1e-2, 1e-3)}
gs_clf_2 = RandomizedSearchCV(clf_2, parameters, n_iter = n_iter_search)
gs_clf_2.fit(X_train, y_train)
pred_2_grid = gs_clf_2.predict(X_test)

n_iter_search = 4
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf_3 = RandomizedSearchCV(clf_3, parameters, n_iter = n_iter_search)
gs_clf_3.fit(X_train, y_train)
pred_3_grid = gs_clf_3.predict(X_test)

#Save best model
pickle.dump(bmodel, open('../models/bmodel.csv', 'wb'))