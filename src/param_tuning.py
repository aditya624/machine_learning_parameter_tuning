import json as j
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

json_data = None
with open('../data/yelp_academic_dataset_review.json') as data_file:
    lines = data_file.readlines()
    joined_lines = "[" + ",".join(lines) + "]"

    json_data = j.loads(joined_lines)

data = pd.DataFrame(json_data)
data.head()

data = data[data.stars != 3]
data['sentiment'] = data['stars'] >= 4

X_train, X_test, y_train, y_test = train_test_split(data, data.sentiment, test_size=0.2)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_train.text, y_train, scoring='accuracy', cv=5, n_jobs=-1)

mean = scores.mean()
std = scores.std()
print(mean)
print(std)

print(pipeline.get_params())

from sklearn.model_selection import GridSearchCV
grid = {
    'vectorizer__ngram_range': [(1, 1), (2, 1)],
    'vectorizer__stop_words': [None, 'english'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [1.0, 0.8],
    'classifier__class_weight': [None, 'balanced'],
    'classifier__n_jobs': [-1]
}

grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='accuracy', n_jobs=-1, cv=5)
grid_search.fit(X=X_train.text, y=y_train)

print("-----------")
print(grid_search.best_score_)
print(grid_search.best_params_)

pipeline2 = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(2, 1))),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression(C=1.0, class_weight=None, n_jobs=-1, penalty='l1'))
])

model = pipeline.fit(X_train.text, y_train)
model2 = pipeline2.fit(X_train.text, y_train)

predicted = model.predict(X_test.text)
predicted2 = model2.predict(X_test.text)

print("model1: " + str(np.mean(predicted == y_test)))
print("model2: " + str(np.mean(predicted2 == y_test)))