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
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, df.index, test_size=0.2, random_state=42, stratify=y)

# Train a classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Predict and verify using the 'Correct' column
predicted_names = encoder.inverse_transform(y_pred)
df_test = df.iloc[test_index].copy()
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

# Calculate macro-average ROC curve and ROC area
n_classes = len(np.unique(y))
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
y_score = clf.decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Average the TPR values and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
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



######################## with Grid Search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# Load your dataframe (replace with your own dataframe)
# df = pd.read_csv('your_data.csv')

# ... (Same preprocessing, feature extraction, and encoding code as before)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers and parameter grids
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier()
}

param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    'Random Forest': {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'sigmoid']
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 2, 4],
        'subsample': [0.5, 0.8, 1]
    }
}

# Perform grid search for each classifier
best_params = {}
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params[name] = grid_search.best_params_

# Update classifiers with the best parameters and evaluate them
for name, clf in classifiers.items():
    clf.set_params(**best_params[name])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy:.2f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=class_names))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from itertools import cycle
from sklearn.preprocessing import label_binarize

# Load your dataframe (replace with your own dataframe)
# df = pd.read_csv('your_data.csv')

# ... (Same preprocessing, feature extraction, encoding, and train-test split code as before)

# ... (Same classifiers, parameter grids, and grid search code as before)

# Update classifiers with the best parameters and evaluate them
for name, clf in classifiers.items():
    clf.set_params(**best_params[name])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test) if hasattr(clf, 'decision_function') else clf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy:.2f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix with class names
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_mat, cmap='coolwarm')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Calculate ROC curve and ROC area for each class
    n_classes = len(np.unique(y))
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    y_score_bin = label_binarize(y_score, classes=np.arange(n_classes)) if len(y_score.shape) == 1 else y_score

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {name}')
    plt.legend(loc="lower right")
    plt.show()

### Graph 1
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

# Set figure size
plt.figure(figsize=(10, 6))

# Plot training and validation accuracies with customized line styles and markers
plt.plot(history.history['accuracy'], linestyle='--', marker='o', markersize=8, linewidth=2, label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], linestyle='-', marker='s', markersize=8, linewidth=2, label='Validation Accuracy', color='orange')

# Set axis labels and font size
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

# Set title and font size
plt.title('Training vs. Validation Accuracy', fontsize=18)

# Set legend location and font size
plt.legend(loc='lower right', fontsize=12)

# Customize tick font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add a grid for better visualization
plt.grid(True)

# Display the plot
plt.show()

### Graph2
import matplotlib.pyplot as plt
import seaborn as sns

def plot_multiple_model_accuracies(sorted_models):
    num_models = len(sorted_models)
    num_rows = 2
    num_cols = 3
    
    # Set Seaborn style
    sns.set_style("whitegrid")

    # Set figure size
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 12), sharex=True)

    for idx, (history, model_name) in enumerate(sorted_models):
        row, col = divmod(idx, num_cols)
        
        # Plot training and validation accuracies for each model
        axes[row, col].plot(history['accuracy'], linestyle='--', marker='o', markersize=10, linewidth=3, label=f'Training Accuracy', alpha=0.8, color='blue')
        axes[row, col].plot(history['val_accuracy'], linestyle='-', marker='s', markersize=10, linewidth=3, label=f'Validation Accuracy', alpha=0.8, color='orange')

        # Set axis labels and font size
        axes[row, col].set_xlabel('Epoch', fontsize=14, labelpad=10)
        axes[row, col].set_ylabel('Accuracy', fontsize=14, labelpad=10)

        # Set title and font size
        axes[row, col].set_title(f'{model_name}', fontsize=18, pad=15)

        # Set legend location and font size
        axes[row, col].legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)

        # Customize tick font size
        axes[row, col].tick_params(axis='both', which='major', labelsize=12)

        # Add a grid for better visualization
        axes[row, col].grid(True)

        # Remove top and right spines for a cleaner look
        sns.despine(top=True, right=True, ax=axes[row, col])

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Display the plot
    plt.show()

##Graph 3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

def plot_classifier_accuracies(classifiers, best_params, X_train, y_train, X_test, y_test):
    num_models = len(classifiers)
    num_rows = 2
    num_cols = 3
    
    # Set Seaborn style
    sns.set_style("whitegrid")

    # Set figure size
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 12), sharex=True)

    for idx, (name, clf) in enumerate(classifiers.items()):
        row, col = divmod(idx, num_cols)
        
        clf.set_params(**best_params[name])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Plot training and validation accuracies for each model
        axes[row, col].bar(['Accuracy'], [accuracy], color=['blue'])

        # Set axis labels and font size
        axes[row, col].set_xlabel('Metric', fontsize=14, labelpad=10)
        axes[row, col].set_ylabel('Score', fontsize=14, labelpad=10)

        # Set title and font size
        axes[row, col].set_title(f'{name}', fontsize=18, pad=15)

        # Customize tick font size
        axes[row, col].tick_params(axis='both', which='major', labelsize=12)

        # Add a grid for better visualization
        axes[row, col].grid(True)

        # Remove top and right spines for a cleaner look
        sns.despine(top=True, right=True, ax=axes[row, col])

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Display the plot
    plt.show()


