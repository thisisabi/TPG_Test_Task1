
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB

import string
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

def create_labels(df):
    labels = df['product_category'].unique()
    i = 0
    idx2class = {}
    class2idx = {}
    for tp in labels:
        idx2class[i] = tp
        class2idx[tp] = i
        i += 1
    return class2idx

def bonus_task():


if __name__ == "__main__":
    train_df = pd.read_json("data/train_test/dataset_en_train.json", lines=True)

    # data is too much for local machine

    df_rat1 = train_df[train_df.stars == 1].head(5000)
    df_rat2 = train_df[train_df.stars == 2].head(5000)
    df_rat3 = train_df[train_df.stars == 3].head(5000)
    df_rat4 = train_df[train_df.stars == 4].head(5000)
    df_rat5 = train_df[train_df.stars == 5].head(5000)

    final_train_df = pd.concat([df_rat1, df_rat2, df_rat3, df_rat4, df_rat5])

    print (final_train_df.shape)


    # using local machine, so cutting down the review body size to 100 characters
    final_train_df['review_body_5000'] = final_train_df['review_body'].str.slice(0, 5000)

    final_train_df['label'] = final_train_df['product_category'].replace(create_labels(final_train_df))

    count_vec = CountVectorizer()
    bow = count_vec.fit_transform(final_train_df['review_body_5000'])

    # making a matrix
    bow = np.array(bow.todense())

    X = bow  # The features we want to analyse
    y = final_train_df['label']  # The labels, in this case feedback

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

    # cutting some corners here as I really don't have much time -
    # am skipping to the step after performing cross validation and hyperparameter optimization
    # once above is done, we will fit the model with ideal parameters and serialise it

    model = MultinomialNB()

    model.fit(X_train, y_train)


















