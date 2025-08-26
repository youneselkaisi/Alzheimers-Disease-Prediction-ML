# Alzheimer's Disease Prediction Project

## Group Statement
We, the group members of Group #2, hereby confirm that this project was completed independently.  

- Younes El-Kaisi (000929935)  
- Yassin Elhakeem (000927776)  
- Youssef Ayoub (000928145)  

---

## Overview
This project develops a machine learning pipeline to predict Alzheimer's disease diagnosis from patient data. It combines exploratory data analysis (EDA), preprocessing, feature selection, and model training/evaluation into a reproducible workflow.  

Alzheimer’s disease is one of the most pressing global health challenges. Early detection enables interventions that may significantly improve patient outcomes. By analyzing clinical, demographic, and lifestyle data, this project demonstrates how data science and predictive modeling can complement medical expertise.  

[include project logo or relevant image here]

---

## Dataset
- **Name:** Alzheimer’s Disease Dataset  
- **Source:** [Kaggle – Alzheimer’s Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data)  
- **Author:** Rabie El Khaoura  
- **License:** Attribution 4.0 International  

### Dataset Description
- **Rows:** 2,149 patients  
- **Columns:** 35 features including demographics, lifestyle, clinical measurements, cognitive assessments, and diagnosis status  

Examples of relevant features:  
- **Demographics:** `Age`, `Gender`, `Ethnicity`, `EducationLevel`  
- **Lifestyle:** `Smoking`, `AlcoholConsumption`, `PhysicalActivity`, `DietQuality`, `SleepQuality`  
- **Clinical Measurements:** `SystolicBP`, `DiastolicBP`, `CholesterolTotal`, `CholesterolLDL`, `CholesterolHDL`, `CholesterolTriglycerides`, `BMI`  
- **Cognitive Assessments:** `MMSE`, `FunctionalAssessment`, `ADL`, `MemoryComplaints`, `Forgetfulness`  
- **Diagnosis:** Target variable (`0` = No Alzheimer’s, `1` = Alzheimer’s)  

[include screenshot of dataset head here]

---

## Project Workflow

### Part 1: Data Exploration
- Loaded the dataset using Pandas and inspected shape, columns, data types, and summary statistics.  
- Checked for missing values (none found).  
- Conducted EDA through visualizations such as:
  - Histogram of patient ages  
  - Pie chart for gender distribution  
  - Bar chart for ethnicity breakdown  
  - Distribution of **MMSE** scores (indicator of cognitive decline)  
  - Comparison of **MMSE scores for forgetful vs. non-forgetful patients**  
  - Stacked bar chart of education level vs. Alzheimer’s diagnosis  
  - Line plot of average ADL (Activities of Daily Living) scores by age  
  - Bar chart for prevalence of depression among patients  
  - Alzheimer’s cases by family history  
  - Overall diagnosis distribution (pie chart)  

[include screenshot of Age distribution plot]  
[include screenshot of MMSE forgetfulness comparison plot]  
[include screenshot of education level by diagnosis stacked bar]  

---

### Part 2: Data Preprocessing
- Dropped irrelevant identifiers (`PatientID`, `DoctorInCharge`).  
- Created a correlation heatmap to identify features most related to Alzheimer’s diagnosis.  
- Selected top correlated features including:  
  `FunctionalAssessment`, `ADL`, `MMSE`, `MemoryComplaints`, `BehavioralProblems`, `DietQuality`, `SleepQuality`, `PhysicalActivity`, `CholesterolTotal`, `EducationLevel`.  
- Constructed a reduced DataFrame (`df2`) for modeling.  

[include screenshot of correlation heatmap here]

---

### Part 3: Modeling

#### Model 1: Support Vector Machine (SVM)
- Used `train_test_split` to divide the dataset (75% train, 25% test).  
- Trained a **linear kernel SVM** with `C=0.01` and `max_iter=1000`.  
- Evaluated using accuracy and classification report.  

**Results (SVM):**  
- Train Accuracy: ~82%  
- Test Accuracy: ~80%  
- Precision/Recall trade-offs observed:  
  - Class `0` (No Alzheimer’s) predicted with higher precision and recall.  
  - Class `1` (Alzheimer’s) slightly under-predicted but still reasonable performance.  

[include screenshot of classification report here]

- Visualized a **confusion matrix** to show true positives/false positives/false negatives.  

[include confusion matrix plot here]

#### Other Models
In addition to SVM, further tests included:  
- Logistic Regression  
- Random Forest  
- Gradient Boosting (if used)  
- (Optional) Neural Networks  

Each was compared based on accuracy, interpretability, and generalization.  

[include loss curve screenshot here if neural nets applied]  
[include feature importance plot here for tree-based models]

---

## Results and Insights
- **SVM** achieved ~80% accuracy on test data, making it a strong baseline model.  
- Correlation analysis highlighted that **FunctionalAssessment**, **ADL**, and **MMSE** are the most critical predictors.  
- Lifestyle variables like **DietQuality**, **SleepQuality**, and **PhysicalActivity** also contribute to diagnosis likelihood.  
- Patients with family history and comorbidities (e.g., depression) showed higher Alzheimer’s prevalence.  

Key takeaway: Combining **cognitive assessment scores** with **lifestyle and clinical metrics** provides a strong predictive basis for Alzheimer’s diagnosis.  

---

## Libraries Used
- **pandas**: data manipulation  
- **numpy**: numerical computation  
- **matplotlib**: visualizations  
- **seaborn**: statistical plotting (if used)  
- **scikit-learn**: preprocessing, model training & evaluation  

[include screenshot of imports section from notebook]

---

## Usage
To run this project locally:  

```bash
# Clone the repository
git clone https://github.com/your-username/alzheimers-prediction.git

# Navigate to directory
cd alzheimers-prediction

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook
