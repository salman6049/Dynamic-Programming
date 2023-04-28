#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load your dataframe (replace with your own dataframe)
# df = pd.read_csv('your_data.csv')

# Preprocess the text data (you can add more preprocessing steps)
df['news_headline'] = df['news_headline'].str.lower()
df['News_Full_story'] = df['News_Full_story'].str.lower()

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['news_headline'] + " " + df['News_Full_story'])

# Encode the target variable
encoder = LabelEncoder()
y = encoder.fit_transform(df['Name'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Predict and verify using the 'Correct' column
predicted_names = encoder.inverse_transform(y_pred)
df_test = df.iloc[y_test.index]
df_test['predicted_name'] = predicted_names
df_test['prediction_correct'] = (df_test['Name'] == df_test['predicted_name']) & (df_test['Correct'] == "Yes")

# Display the verification results
print("Number of correct predictions:", df_test['prediction_correct'].sum())
print("Total predictions:", len(df_test))
print("Correct prediction ratio:", df_test['prediction_correct'].sum() / len(df_test))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load your dataframe (replace with your own dataframe)
# df = pd.read_csv('your_data.csv')

# Preprocess the text data (you can add more preprocessing steps)
df['news_headline'] = df['news_headline'].str.lower()
df['News_Full_story'] = df['News_Full_story'].str.lower()

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['news_headline'] + " " + df['News_Full_story'])

# Encode the target variable
encoder = LabelEncoder()
y = encoder.fit_transform(df['Name'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# ... (Same imports, preprocessing, training, and evaluation code as before)

# Plot confusion matrix with class names
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(conf_mat, cmap='coolwarm')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.title("Confusion Matrix")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ... (Same code as before for calculating macro-average ROC curve and ROC area)

# Plot macro-average ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})', color='navy')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-average ROC Curve')
plt.legend(loc='lower right')
plt.show()

