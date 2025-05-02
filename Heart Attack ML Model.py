#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv(r"C:\Users\mrfun\Downloads\HA2022.csv")
#dataset.head()

feature_col = ['Sex', 'GeneralHealth', 'PhysicalHealthDays',
               'MentalHealthDays', 'SleepHours',
                'HadStroke', 'HadDepressiveDisorder',
               'HadDiabetes', 'SmokerStatus', 'AgeCategory',
               'BMI', 'AlcoholDrinkers']
        
        
X = dataset[feature_col]
y = dataset['HadHeartAttack']
#y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Confusion Matrix:",conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[ ]:




