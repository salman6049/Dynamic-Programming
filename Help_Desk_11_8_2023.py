#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import joblib
import os
import pandas as pd


save_path = r'C:\Work\trained_model'
# Function for text transformation
def transform_text(series, vectorizer=None, max_features=1000, prefix=''):
    series = series.fillna('Unknown').astype(str)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features)
        transformed_data = vectorizer.fit_transform(series).toarray()
    else:
        transformed_data = vectorizer.transform(series).toarray()
    return pd.DataFrame(transformed_data, columns=[prefix + '_' + col for col in vectorizer.get_feature_names_out()]), vectorizer

# Derivatives and Derived columns
Derivative = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']
Derived = ['col_7', 'col_8', 'col_9', 'col_10']

# Prepare the input features
vectorizers = {col: TfidfVectorizer() for col in Derivative}
Xs = []

for col in Derivative:
    X_col =  vectorizers[col].fit_transform(df[col])
    Xs.append(pd.DataFrame(X_col.toarray(), columns=vectorizers[col].get_feature_names_out()))

X = pd.concat(Xs, axis=1)

# Parameters for GridSearch
param_grid = {
    'C': [1,10],
    'kernel': ['linear'],
    'gamma': ['auto'],
    'class_weight': ['balanced']}

# Define the directory where you want to save the files
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Training and saving models
for derived in Derived:
    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(mlb.fit_transform(df[derived].fillna('Unknown').values.reshape(-1, 1)), columns=mlb.classes_)
    
    print("Performing train-validation split ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svc_model = SVC(decision_function_shape='ovr')
    
    print('starting grid search...')
    grid_search = GridSearchCV(estimator=svc_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.argmax(axis=1))
    best_svc = grid_search.best_estimator_
    
    print('Validatiing the model...')
    y_pred = best_svc.predict(X_test)

    print(f"\nSupport Vector Classifier Results for {derived}:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=mlb.classes_))

    # Save models and transformers
    joblib.dump(best_grid, os.path.join(save_path, f'best_grid_{derived}.pkl'))
    joblib.dump(mlb, os.path.join(save_path, f'mlb_{derived}.pkl'))
    for col in Derivative:
        joblib.dump(vectorizers[col], os.path.join(save_path, f'vectorizer_{col}.pkl'))


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os
import pandas as pd

save_path = r'C:\Work\trained_model'

def transform_text(series, vectorizer=None, max_features=1000, prefix=''):
    series = series.fillna('Unknown').astype(str)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features)
        transformed_data = vectorizer.fit_transform(series).toarray()
    else:
        transformed_data = vectorizer.transform(series).toarray()
    return pd.DataFrame(transformed_data, columns=[prefix + '_' + col for col in vectorizer.get_feature_names_out()]), vectorizer

Derivative = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']
Derived = ['col_7', 'col_8', 'col_9', 'col_10']

vectorizers = {col: TfidfVectorizer() for col in Derivative}
Xs = []

for col in Derivative:
    X_col, vectorizers[col] = transform_text(df[col], vectorizer=vectorizers[col])
    Xs.append(X_col)

X = pd.concat(Xs, axis=1)

param_grid = {
    'estimator__C': [1,10],
    'estimator__kernel': ['linear'],
    'estimator__gamma': ['auto'],
    'estimator__class_weight': ['balanced']
}

if not os.path.exists(save_path):
    os.makedirs(save_path)

for derived in Derived:
    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(mlb.fit_transform(df[derived].fillna('Unknown').str.split('|')), columns=mlb.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ovr_svc = OneVsRestClassifier(SVC(decision_function_shape='ovr'))
    
    grid_search = GridSearchCV(estimator=ovr_svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_ovr_svc = grid_search.best_estimator_

    y_pred = best_ovr_svc.predict(X_test)

    print(f"\nSupport Vector Classifier Results for {derived}:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=mlb.classes_))
    
    joblib.dump(best_ovr_svc, os.path.join(save_path, f'best_ovr_svc_{derived}.pkl'))
    joblib.dump(mlb, os.path.join(save_path, f'mlb_{derived}.pkl'))
    for col in Derivative:
        joblib.dump(vectorizers[col], os.path.join(save_path, f'vectorizer_{col}.pkl'))


# In[14]:


from Levenshtein import distance as levenshtein_distance
import re

def compare_digit_sequences(seq1, seq2):
    # Strips leading zeros and compares the integer values
    int_seq1 = int(seq1) if seq1 else 0
    int_seq2 = int(seq2) if seq2 else 0
    return int_seq1 == int_seq2

def calculate_similarity_score(str1, str2):
    # Split the strings into parts of digits and non-digits
    parts1 = re.split('(\d+)', str1)
    parts2 = re.split('(\d+)', str2)
    
    # Pad the shorter list of parts with empty strings to match the length of the longer list
    length_difference = len(parts1) - len(parts2)
    if length_difference > 0:
        parts2 += [''] * length_difference
    elif length_difference < 0:
        parts1 += [''] * (-length_difference)
    
    # Initialize score and counts
    total_length = 0
    total_similarity = 0
    
    # Calculate similarity for each part
    for part1, part2 in zip(parts1, parts2):
        if part1.isdigit() and part2.isdigit():
            # For digit parts, check if they are equal when leading zeros are ignored
            if compare_digit_sequences(part1, part2):
                # Consider exact match for the digits, add max length of the two parts to the score
                total_similarity += max(len(part1), len(part2))
            total_length += max(len(part1), len(part2))
        else:
            # For non-digit parts, use Levenshtein distance
            dist = levenshtein_distance(part1, part2)
            total_similarity += (max(len(part1), len(part2)) - dist)
            total_length += max(len(part1), len(part2))
    
    # Calculate final score
    if total_length == 0:
        return 1.0 if str1 == str2 else 0.0
    score = total_similarity / total_length
    return score

# Example usage:
str_1 = '2017AP001261'
str_2 = '2017AP261'
print(f"Similarity score between '{str_1}' and '{str_2}':", calculate_similarity_score(str_1, str_2))

str_1 = '23ACR1001'
str_2 = '23ACR1051'
print(f"Similarity score between '{str_1}' and '{str_2}':", calculate_similarity_score(str_1, str_2))

str_1 = '23ACR509'
str_2 = '23ACR01509'
print(f"Similarity score between '{str_1}' and '{str_2}':", calculate_similarity_score(str_1, str_2))

str_1 = '23ACR257'
str_2 = '23ACR01257'
print(f"Similarity score between '{str_1}' and '{str_2}':", calculate_similarity_score(str_1, str_2))


# In[ ]:


from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import re
from Levenshtein import distance as levenshtein_distance

def compare_digit_sequences(seq1, seq2):
    # Strips leading zeros and compares the integer values
    int_seq1 = int(seq1) if seq1 else 0
    int_seq2 = int(seq2) if seq2 else 0
    return int_seq1 == int_seq2

@pandas_udf(DoubleType())
def dock_num_similarity_score_udf(str_1: pd.Series, str_2: pd.Series) -> pd.Series:
    def calculate_similarity_score(num1, num2):
        # Split the strings into parts of digits and non-digits
        parts1 = re.split('(\d+)', num1)
        parts2 = re.split('(\d+)', num2)

        # Pad the shorter list of parts with empty strings to match the length of the longer list
        length_difference = len(parts1) - len(parts2)
        if length_difference > 0:
            parts2 += [''] * length_difference
        elif length_difference < 0:
            parts1 += [''] * (-length_difference)

        # Initialize score and counts
        total_length = 0
        total_similarity = 0

        # Calculate similarity for each part
        for part1, part2 in zip(parts1, parts2):
            if part1.isdigit() and part2.isdigit():
                # For digit parts, check if they are equal when leading zeros are ignored
                if compare_digit_sequences(part1, part2):
                    # Consider exact match for the digits, add max length of the two parts to the score
                    total_similarity += max(len(part1), len(part2))
                total_length += max(len(part1), len(part2))
            else:
                # For non-digit parts, use Levenshtein distance
                dist = levenshtein_distance(part1, part2)
                total_similarity += (max(len(part1), len(part2)) - dist)
                total_length += max(len(part1), len(part2))

        # Calculate final score
        if total_length == 0:
            return 1.0 if num1 == num2 else 0.0
        score = total_similarity / total_length
        return score
    
    return str_1.combine(str_2, calculate_similarity_score)

# When using in a Spark DataFrame operation, you would apply it like this:
# df.withColumn('similarity_score', dock_num_similarity_score_udf(df['str_col1'], df['str_col2']))

