#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Part 1: Data Exploration

# In[33]:


# importing dataset as dataframe
df = pd.read_csv("alzheimers_disease_data.csv")
df


# <b>About the Dataset:</b>
# 
# Name: Alzheimer's Disease Dataset 
# 
# Source: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data
# 
# Author: Rabie El Khaoura
# 
# License: Attribution 4.0 International
# 
# Some Relevant Fields: (Taken from kaggle)
# 
# Demographics
# - Age: (The age of the patients ranges from 60 to 90 years)
# - Gender: (0 represents Male and 1 represents Female)
# - Ethnicity: (0: Caucasian 1: African American2: Asian 3: Other)
# 
# Lifestyle
# - AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20
# - PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10
# - SystolicBP: Systolic blood pressure, ranging from 90 to 180 mmHg
# 
# Clinical Meausurments
# - DiastolicBP: Diastolic blood pressure, ranging from 60 to 120 mmHg
# - CholesterolTotal: Total cholesterol levels, ranging from 150 to 300 mg/dL
# - CholesterolLDL: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL
# - CholesterolHDL: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL
# - CholesterolTriglycerides: Triglycerides levels, ranging from 50 to 400 mg/dL
# 
# Cognitive and Functional Assessments
# - MMSE: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment
# - ADL: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment
# 
# Diagnosis Information
# - Diagnosis (target feature): Diagnosis status for Alzheimer's Disease, where 0 indicates No and 1 indicates Yes

# In[36]:


# list of column names
print(df.columns)


# In[38]:


# shape of dataset (rows,cols)
df.shape


# In[40]:


# data types & non-null count
df.info()


# In[42]:


# statistical overview of data
df.describe()


# In[43]:


# count of nulls
df.isnull().sum()


# ### Visualizations:

# In[47]:


# Plotting Age Distribution of Alzheimer's pateints 

# creating histogram
  
plt.hist(df['Age'], edgecolor='black')

# adding labels and title
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Patients")

plt.show()


# In[48]:


# pie chart showing gender breakdown

# mapping values for labels
status_counts = df['Gender'].map({0: 'Male', 1: 'Female'}).value_counts()

# plotting pie chart
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',startangle = 90)
# adding title
plt.title('Gender Breakdown')

plt.show()


# In[49]:


# Bar chart showing distribution of ethnicities

# Counting occurrences of each category
class_counts = df["Ethnicity"].map ({0:"Caucasian", 1: "African-American", 2: "Asian", 3: "Other"}).value_counts()

plt.bar(class_counts.index, class_counts.values, color='navy',zorder=3)           

# title

plt.title("Distribution of Ethnicities")

# rotating x-axis labels
plt.xticks(rotation=40)

# adding grid
plt.grid(axis='y')

plt.show()


# In[51]:


# Distribution of MMSE (Mini-Mental State Examination) scores (lower the score, more cognitive decline)

# creating histogram
  
plt.hist(df['MMSE'], edgecolor='black',color='green')

# adding labels and title
plt.xlabel("MMSE Score")
plt.ylabel("Frequency")
plt.title("Distribution of MMSE Scores")

plt.show()


# In[53]:


# Histogram showing Distribution of MMSE (Mini-Mental State Examination) scores (lower the score more cognitive decline) of patients who have presence of forgetfullnes or not

# Split MMSE scores by forgetfulness = 1 is yes or 0 is no
mmse_forgetful = df[df['Forgetfulness'] == 1]['MMSE']
mmse_not_forgetful = df[df['Forgetfulness'] == 0]['MMSE']

# Plotting histogram
plt.hist(mmse_forgetful, alpha=0.8, label='Forgetful', color='red', edgecolor='black')
plt.hist(mmse_not_forgetful,  alpha=0.4, label='Not Forgetful', color='grey', edgecolor='black')

# Labels and title
plt.xlabel("MMSE Score")
plt.ylabel("Frequency")
plt.title("MMSE Score Distribution: Forgetful vs. Not Forgetful")

# adding legend
plt.legend()

plt.show()


# In[54]:


# stacked bar chart showing education levels by diagnosis

# Counting patients grouped by EducationLevel and Diagnosis
edu_counts = df.groupby(['EducationLevel', 'Diagnosis']).size().unstack(fill_value=0)

# Plotting stacked bar chart and determinning colours
edu_counts.plot(kind='bar', stacked=True, color=['turquoise', 'red'], edgecolor='black')

# addingle lables and title
plt.xticks(ticks=range(4), labels=['None', 'High School', "Bachelor's", 'Higher'], rotation=45)
plt.xlabel("Education Level")
plt.ylabel("Number of Patients")
plt.title("Education Level by Diagnosis")

# adding a legend
plt.legend(['No Alzheimer\'s', 'Alzheimer\'s'])

plt.show()


# In[56]:


# Line plot showing mean ADL scores by age 

# grouping by  age and getting mean ADL score
avg_adl_by_age = df.groupby('Age')['ADL'].mean()

# plotting the line chart and adding colour
plt.plot(avg_adl_by_age.index, avg_adl_by_age.values, linestyle='-', color='royalblue')

# Labels and title
plt.xlabel("Age")
plt.ylabel("Average ADL Score")
plt.title("ADL Score by Age")

# adding a grid
plt.grid(True)
plt.show()


# In[57]:


# bar chart showing patients that have depression or not

# Counting how many patients have and don’t have depression
depression_counts = df['Depression'].value_counts().sort_index()  # 0 = No, 1 = Yes

# plotting bar chart
plt.bar(['No Depression', 'Has Depression'], depression_counts, color=['lightgrey', 'darkred'], edgecolor='black')

# adding title and lables
plt.title("Count of Patients with Depression")
plt.ylabel("Number of Patients")

plt.show()


# In[58]:


# bar chart showing alzhemiers cases by those who have family history of it and not 

# Filtering only patients diagnosed with Alzheimer's
alzheimers_only = df[df['Diagnosis'] == 1]

# Countjng how many of them have family history or not
fh_diag_counts = alzheimers_only['FamilyHistoryAlzheimers'].value_counts().sort_index()

# Plotting bar chart
plt.bar(['No Family History', 'Has Family History'], fh_diag_counts, color='orange', edgecolor='black')

# adding title and lable
plt.title("Alzheimer's Cases by Family History")
plt.ylabel("Number of Alzheimer's Diagnoses")

plt.show()


# In[60]:


# Pie chart showing overall distribution of cases of alzhemiers in the dataset

# Counting diagnosis categories
diagnosis_counts = df['Diagnosis'].value_counts().sort_index()

# adding lable and colour
labels = ['No Alzheimer\'s', 'Alzheimer\'s']
colors = ['#A6CEE3', '#FB9A99']  # Soft blue & muted red

# Plotting pie chart
plt.pie(diagnosis_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
# title
plt.title("Distribution of Alzheimer's Diagnoses")

plt.show()


# ## Part 2: Data Preprocessing

# In[62]:


# Dropping irrelevant features
df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

df.head()


# In[63]:


# plotting correlation heatmap to choose highest correlated features 

# corr is pandas df method used to calculate correlation between numerical columns in a dataframe
corr = df.corr()[['Diagnosis']].sort_values(by='Diagnosis', ascending=False)

# imshow is a matplotlib function that displayes image like data, like a matrix grid
plt.imshow(corr, cmap='coolwarm', aspect='auto')
# adding side colourbar
plt.colorbar()

# adding ticks and title 
plt.xticks([0], ['Diagnosis'])
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation with Alzheimer's Diagnosis")
# increasing size
plt.figure(figsize=(15, 16))  

plt.show()


# In[65]:


# new dataframe with newly chosen features 

df2 = df[["FunctionalAssessment","ADL","MMSE","MemoryComplaints","BehavioralProblems","DietQuality","SleepQuality","PhysicalActivity","CholesterolTotal","EducationLevel",
          "Diagnosis" ]]
df2.head()
     


# #### Preprocessing explanation:
# 
# - Dropped IDs and identifiers (PatientID, DoctorInCharge) as they are non-informative for prediction and clean up the dataset
# - Used a correlation heatmap to visualize linear relationships between features and Alzheimer’s diagnosis
# - Selected the top 12 features with the strongest positive or negative correlation to the target variable (Diagnosis)
# - Created a separate DataFrame (df2) containing only the selected features for testing and modeling purposes

# # Part 3a: Model Selection & Training (SVM)

# ### Test 1

# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Splitting into features and target
X = df2.drop("Diagnosis", axis=1)  
y = df2["Diagnosis"]  

print(X)
print(y)


# In[73]:


# spliting into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# verifying split ratios
print(len(X_train))  
print(len(X_test))


# In[76]:


from sklearn.metrics import accuracy_score, classification_report

# Creating a SVM model with a linear kernel
clf1 = SVC(kernel="linear", C=0.01, max_iter=1000)

# training the model
clf1.fit(X_train, y_train)


# In[78]:


# predicting training accuracy
y_pred_train = clf1.predict(X_train)
# predicting test accuracy
y_pred_test = clf1.predict(X_test)

# printing accuracy scores
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))


# In[81]:


# printing classification report
print( classification_report(y_test, y_pred_test))


# In[82]:


from sklearn.metrics import confusion_matrix

# Generating predictions
y_pred = clf1.predict(X_test)

# initilaizing the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Ploting the confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(conf_matrix, cmap='viridis')
fig.colorbar(cax)

#  setting classification: 0 = No Alzheimer's, 1 = Alzheimer's
class_names = ["No Alzheimer's", "Alzheimer's"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# Axis labels and title
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("Confusion Matrix")

# Annotate each cell with count
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white', fontsize=14)


plt.show()


# ### Test 2:

# In[84]:


from sklearn.preprocessing import StandardScaler
# Scaling the data using standard scalar 
scaler = StandardScaler()

# scaling correctly to avoid data leakage (seperating train for test)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating 2nd classifer and chaging kernel method
clf2 = SVC(kernel="rbf", C=1, gamma="scale", max_iter=5000, random_state=42)

# Training classifier 2
clf2.fit(X_train_scaled, y_train)


# In[85]:


# predicting train and test sets
y_pred2 = clf2.predict(X_test_scaled)
y_pred2_train = clf2.predict(X_train_scaled)

print("Train Accuracy:", accuracy_score(y_train, y_pred2_train))
print("Test Accuracy :", accuracy_score(y_test, y_pred2))


# In[86]:


# printng classification report
print(classification_report(y_test, y_pred2))


# In[87]:


# Generating confusion matrix for 2nd classifier 
conf_matrix2 = confusion_matrix(y_test, y_pred2)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(conf_matrix2, cmap='viridis')
fig.colorbar(cax)

# Labeling and adding ticks 
class_names = ["No Alzheimer's", "Alzheimer's"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# titles and axis lables
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("Confusion Matrix (clf2)")

# Annotate values
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(conf_matrix2[i, j]), ha='center', va='center', color='white', fontsize=14)


plt.show()


# #### SVM Justification:
# 
# - Performs well on structured, tabular datasets like this one with a moderate number of features and observations
# 
# - Effective at handling both linear and nonlinear boundaries through kernel tricks such as the RBF kernel
# 
# - Well-suited for binary classification problems and benefits from standardized features, which were applied during preprocessing
# 
# 

# # Part 3b: Model Selection & Training (MLP)

# ### Test 3:

# In[91]:


from sklearn.neural_network import MLPClassifier

# initializing basic MLP classifier 
clf3 = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", max_iter=1000, random_state=42)
clf3.fit(X_train_scaled, y_train)


# In[92]:


# Predicting accuracies on test set and training set
y_pred3_test = clf3.predict(X_test_scaled)
y_pred3_train = clf3.predict(X_train_scaled)

# printing accuracy scores 
print("Train Accuracy:", accuracy_score(y_train, y_pred3_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred3_test))
print( classification_report(y_test, y_pred3_test))


# In[94]:


# Plotting the loss curve

plt.plot(clf3.loss_curve_)

# adding titles and lables
plt.title("MLP Loss Curve (clf3)")
plt.xlabel("Iterations")
plt.ylabel("Loss")

# adding grid
plt.grid(True)

plt.show()


# In[96]:


# Computing confusion matrix
conf_matrix3 = confusion_matrix(y_test, y_pred3_test)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(conf_matrix3, cmap='viridis')
fig.colorbar(cax)

# adding ticks 
class_names = ["No Alzheimer's", "Alzheimer's"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# Axis titles and lables
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("Confusion Matrix (MLP)")

# Annotate values
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(conf_matrix3[i, j]), ha='center', va='center', color='white', fontsize=14)


plt.show()


# ### Test 4:

# In[100]:


# making final MLP classifier 

clf4 = MLPClassifier(hidden_layer_sizes=(12,6), activation="relu",random_state=17,learning_rate_init=0.1,max_iter=1000,verbose = True)

# training classifier 
clf4.fit(X_train_scaled, y_train)


# In[102]:


# Plotting final loss curve for clf4
plt.plot(clf4.loss_curve_)

# adding titles and lables
plt.title("MLP Loss Curve (clf4)")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.grid(True)

plt.show()


# In[104]:


# predicting accuracies for train and testing sets

y_pred4 = clf4.predict(X_test_scaled)

# printing train and test accuracies 
print("Train Accuracy:", accuracy_score(y_train, clf4.predict(X_train_scaled)))
print("Test Accuracy:", accuracy_score(y_test, y_pred4))

# printing classification  report
print( classification_report(y_test, y_pred4))


# In[106]:


# Compute confusion matrix
conf_matrix4 = confusion_matrix(y_test, y_pred4)

# Plotting matrix 
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(conf_matrix4, cmap='viridis')
fig.colorbar(cax)

# Class labels
class_names = ["No Alzheimer's", "Alzheimer's"]

# adding ticks 
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# adding axis labels and title
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("Confusion Matrix (clf4)")

# Annotate counts
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(conf_matrix4[i, j]), ha='center', va='center', color='white', fontsize=14)

plt.show()


# #### MLP Justification:
# 
# - Performs well on structured datasets like this one, especially when features are a mix of standardized numeric and categorical variables
# 
# - Capable of modeling complex, nonlinear relationships between cognitive, behavioral, and lifestyle-related features
# 
# - Learns hierarchical feature interactions through multiple hidden layers, making it more flexible than linear models
# 
# - Suitable for binary classification tasks such as Alzheimer's diagnosis prediction and benefits significantly from feature scaling, which was done during preprocessing
# 
# 

# In[111]:


# Clustered bar chart showing all 4 tests with train and test accuracies 

# Actual train and test accuracy scores
data = {
    "SVM (Normal)": [0.822, 0.803],
    "SVM (Standardized)": [0.9186, 0.851],
    "MLP (Normal)": [0.883, 0.859],
    "MLP (Tuned)": [0.899, 0.896]
}
# Create DataFrame
df_acc = pd.DataFrame(data, index=["Train Accuracy", "Test Accuracy"]).T

# Plotting clusterd bar chart and deciding size
ax = df_acc.plot(kind="bar", figsize=(10, 6))

# adding titles and lables
plt.title("Train vs Test Accuracy: SVM and MLP Models")
plt.ylabel("Accuracy")

# adding limit to y-axis
plt.ylim(0.75, 1.00)
# addign a legend for better readibility
plt.legend(title="Metric")
# adding a grid
plt.grid(axis='y')

plt.show()


# ## Part 4: Conclusion 

# - The Multi-Layer Perceptron (MLP) model outperformed all others, reaching a test accuracy of approximately 90%, with balanced precision and recall, making it the most effective model for Alzheimer’s prediction in this dataset.
# 
# - The Support Vector Machine (SVM) with RBF kernel and scaled features also performed strongly with around 87% test accuracy, though it had slightly lower recall for Alzheimer’s cases.
# 
# - The basic linear SVM model, trained on unscaled data, achieved only 80% test accuracy, highlighting the importance of data scaling and appropriate kernel selection.
# 
# - Feature selection using correlation analysis helped isolate the most predictive features, improving model efficiency and interpretability.
# 
# - Overall, the project demonstrated that combining structured preprocessing, model tuning, and metric-based evaluation leads to a reliable and interpretable classification pipeline for medical prediction tasks.
