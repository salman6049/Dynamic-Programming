#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Create 5 features with some randomness
np.random.seed(0)
df = pd.DataFrame({
    'feature_1': [' '.join(np.random.choice(['dog', 'cat', 'mouse', 'parrot', 'fish'], size=np.random.randint(1,5))) for _ in range(1000)],
    'feature_2': [' '.join(np.random.choice(['blue', 'red', 'yellow', 'green'], size=np.random.randint(1,5))) for _ in range(1000)],
    'feature_3': [' '.join(np.random.choice(['apple', 'banana', 'cherry', 'date'], size=np.random.randint(1,5))) for _ in range(1000)],
    'feature_4': [' '.join(np.random.choice(['car', 'bus', 'bicycle', 'train'], size=np.random.randint(1,5))) for _ in range(1000)],
    'feature_5': [' '.join(np.random.choice(['circle', 'square', 'triangle', 'rectangle'], size=np.random.randint(1,5))) for _ in range(1000)],
})

# Generate three "derived" features
df['derived_1'] = df['feature_1'] + ' ' + df['feature_2'] # concatenation of feature_1 and feature_2
df['derived_2'] = df['feature_3'].apply(lambda x: str(x.count('apple'))) # count of 'apple' in feature_3
df['derived_3'] = df['feature_4'].apply(lambda x: ' '.join([word[0] for word in x.split()])) # start letters of words in feature_4

# Define a function to transform a dataframe of strings into a matrix of word counts
def transform_text(df, max_features=1000):
    vectorizer = CountVectorizer(max_features=max_features)
    transformed_data = vectorizer.fit_transform(df.values.ravel()).toarray()
    return pd.DataFrame(transformed_data, columns=vectorizer.get_feature_names_out())


# In[2]:


df.head()


# In[3]:


df.describe(include='all')


# In[4]:


df.isnull().sum()


# In[5]:


df.duplicated().sum()


# In[8]:


# Transform all features
X = pd.concat([transform_text(df[col]) for col in ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']], axis=1)

# Transform all derived features
mlb = MultiLabelBinarizer()
y = pd.concat([pd.DataFrame(mlb.fit_transform(df[col].str.split()), columns=mlb.classes_) for col in ['derived_1', 'derived_2', 'derived_3']], axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model for each target
for i in range(y_test.shape[1]):
    print(f"Accuracy for target {i+1}: {accuracy_score(y_test.iloc[:, i], y_pred[:, i])}")


# In[9]:


import scipy.stats as ss
import numpy as np

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# In[14]:


correlation_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
for i in df.columns:
    for j in df.columns:
        correlation_matrix.loc[i, j] = cramers_v(df[i], df[j])

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".2f", cmap='viridis')
plt.show()


# In[15]:


from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
import lightgbm as lgb

# Create and train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Predict and evaluate the RandomForest model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier Results:")
for i in range(y_test.shape[1]):
    print(f"\nClassification report for target {i+1}:")
    print(classification_report(y_test.iloc[:, i], y_pred_rf[:, i], zero_division=0))

# Create and train the XGBoost model with OneVsRest strategy
xgb_model = OneVsRestClassifier(xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
xgb_model.fit(X_train, y_train)
# Predict and evaluate the XGBoost model
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Classifier Results:")
for i in range(y_test.shape[1]):
    print(f"\nClassification report for target {i+1}:")
    print(classification_report(y_test.iloc[:, i], y_pred_xgb[:, i], zero_division=0))

# Create and train the LightGBM model with OneVsRest strategy
lgb_model = OneVsRestClassifier(lgb.LGBMClassifier(n_estimators=100, random_state=42))
lgb_model.fit(X_train, y_train)
# Predict and evaluate the LightGBM model
y_pred_lgb = lgb_model.predict(X_test)
print("\nLightGBM Classifier Results:")
for i in range(y_test.shape[1]):
    print(f"\nClassification report for target {i+1}:")
    print(classification_report(y_test.iloc[:, i], y_pred_lgb[:, i], zero_division=0))


# In[16]:


from sklearn.metrics import classification_report

# Evaluate the model for each target
for i in range(y_test.shape[1]):
    print(f"\nClassification report for target {i+1}:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))#zero_division=0 is used to handle the case where there are no positive samples in y_true or y_pred, resulting in undefined precision, recall or F-score.


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Calculate classification report for each model
report_rf = classification_report(y_test, y_pred_rf, zero_division=0, output_dict=True)
report_xgb = classification_report(y_test, y_pred_xgb, zero_division=0, output_dict=True)
report_lgb = classification_report(y_test, y_pred_lgb, zero_division=0, output_dict=True)

# Combine reports into a single DataFrame
report = pd.concat([
    pd.DataFrame(report_rf).transpose().assign(Model='Random Forest'),
    pd.DataFrame(report_xgb).transpose().assign(Model='XGBoost'),
    pd.DataFrame(report_lgb).transpose().assign(Model='LightGBM')
])

# Reset index
report.reset_index(inplace=True)

# Rename columns
report.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support', 'Model']

# Melt DataFrame to long format for easier plotting
report_melt = report.melt(id_vars=['Model', 'Class'], value_vars=['Precision', 'Recall', 'F1-Score'], var_name='Metric', value_name='Score')

# List of unique metrics
metrics = report_melt['Metric'].unique()

# Loop through each metric and create a separate bar plot
for metric in metrics:
    plt.figure(figsize=(10, 6))
    data = report_melt[report_melt['Metric'] == metric]
    sns.barplot(x='Class', y='Score', hue='Model', data=data, palette='viridis', alpha=0.6)
    plt.title(metric)
    plt.ylabel('Score')
    plt.xticks(rotation=90)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# In[ ]:




