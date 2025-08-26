# Alzheimer's Disease Prediction Using Machine Learning

## Overview
This project explores how machine learning can assist in predicting Alzheimer’s Disease (AD) based on health, lifestyle, and cognitive information. Alzheimer’s is the leading form of dementia worldwide, and early detection remains a major challenge in improving care and outcomes. By applying predictive models on clinical and behavioral data, this project demonstrates how data-driven approaches can provide insights for healthcare research and early intervention strategies.

The project uses a dataset of over 2,100 patients and compares different machine learning models — specifically **Support Vector Machines (SVMs)** and **Multi-Layer Perceptrons (MLPs)** — to assess their ability to classify Alzheimer’s vs. non-Alzheimer’s patients.




## Why This Project is Useful
- **Early Detection:** Assists in identifying patients at risk earlier in the progression.
- **Feature Insights:** Reveals which medical, lifestyle, and cognitive variables are most correlated with Alzheimer’s.
- **Model Comparisons:** Highlights the strengths and weaknesses of traditional classifiers (SVM) vs. neural networks (MLP).
- **Healthcare Applications:** Serves as a foundation for applying AI/ML techniques in clinical prediction tasks.

---

## Dataset
The dataset is sourced from **Kaggle** (created by *Rabie El Khaoura*).  
It includes **2,149 patients** with **35 features** spanning demographics, lifestyle, medical history, and cognitive/behavioural metrics.

Dataset link: [Alzheimer’s Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

**Feature Categories:**
- Demographics: Age, Gender, Ethnicity, Education
- Lifestyle Factors: Smoking, Alcohol, Diet, Sleep, Physical Activity
- Medical History: Diabetes, Depression, Hypertension, Cardiovascular Disease
- Clinical Measurements: Blood Pressure, Cholesterol levels
- Cognitive & Behavioral: MMSE scores, Forgetfulness, ADL (Activities of Daily Living), Memory Complaints, Behavioural Problems
- Target Variable: Diagnosis (0 = No Alzheimer’s, 1 = Alzheimer’s)

---

## Exploratory Data Analysis (EDA)
Before model training, several visualizations and summaries were created:

- **Class Balance:** 35% Alzheimer’s, 65% No Alzheimer’s.  
- **Age Distribution:** Majority between 80–90 years old.  
- **Education vs Diagnosis:** Higher education levels correlate with lower Alzheimer’s rates.  
- **Depression and Diagnosis:** Depression is more prevalent among Alzheimer’s patients.  
- **Family History:** Not always predictive — many patients without family history still developed Alzheimer’s.  


## Visualizations

### Distribution by Diagnosis
![Distribution by diagnosis](https://github.com/youneselkaisi/Alzheimers-Disease-Prediction-ML/blob/main/Visuals/Distribution%20by%20diagnosis.png?raw=true)

### Age Distribution of Patients
![Age distribution of patients](https://github.com/youneselkaisi/Alzheimers-Disease-Prediction-ML/blob/main/Visuals/Age%20distribution%20of%20patients.png?raw=true)

### Education Level by Diagnosis
![Education level by diagnosis](https://github.com/youneselkaisi/Alzheimers-Disease-Prediction-ML/blob/main/Visuals/Edu%20lvl%20by%20diagnosis.png?raw=true)

### Alzheimer's by Family History
![Alzheimer's by family history](https://github.com/youneselkaisi/Alzheimers-Disease-Prediction-ML/blob/main/Visuals/Alzheimers%20by%20family%20history.png?raw=true)

### Count of Patients with Depression
![Count of patients with depression](https://github.com/youneselkaisi/Alzheimers-Disease-Prediction-ML/blob/main/Visuals/Count%20of%20patients%20w%20depression.png?raw=true)


---

## Data Preprocessing
Steps taken to prepare the data:

- Checked for missing values (none found).  
- Generated **correlation heatmap** to identify strong predictors.  
- Selected **Top 10 Features** most correlated with Alzheimer’s (both positive and negative).  
- Created a refined dataset for modelling with these features.  
- Standardized features where necessary for SVM and MLP models.  

[include screenshot of correlation heatmap here]  
[include screenshot of top 10 features selection here]  

---

## Machine Learning Models

### Support Vector Machine (SVM)
- **Linear Kernel (Unscaled):** Achieved ~80% accuracy but struggled with recall (false negatives high).  
- **RBF Kernel (Scaled):** Improved test accuracy to 85%, reduced false negatives, but showed signs of overfitting.  

[include confusion matrix for SVM models here]  

### Multi-Layer Perceptron (MLP)
- **Basic MLP (1 hidden layer):** ~86% accuracy, decent recall but still missed some cases.  
- **Tuned MLP (2 hidden layers, adjusted learning rate):** ~90% accuracy, balanced precision and recall, best overall performance.  

[include screenshot of MLP loss curve here]  
[include tuned MLP confusion matrix here]  

---

## Model Comparison
- **SVM (Linear):** Simplest, but weakest results (~80% test accuracy).  
- **SVM (RBF + Standardization):** Stronger but prone to overfitting.  
- **MLP (Default):** Good results with convergence shown in loss curve.  
- **MLP (Tuned):** Best-performing model with ~90% test accuracy and balanced sensitivity.  

[include chart comparing model accuracies here]  

---

## Key Insights
- Standardization significantly improves performance for both SVM and MLP.  
- Neural networks (MLP) were more effective than linear classifiers for this dataset.  
- Education level, MMSE, sleep quality, and memory complaints were among the most predictive features.  
- The tuned MLP achieved the best balance between accuracy and generalization.  
- False negatives remain the most concerning error type in medical contexts, underscoring the importance of recall.  

---

## Conclusion
This project demonstrates the potential of machine learning in predicting Alzheimer’s Disease using real-world patient data. While no model is perfect, the tuned MLP achieved the most promising results with nearly 90% accuracy. Beyond accuracy, the project highlights the importance of feature selection, data preprocessing, and balancing interpretability with predictive power.  

Future work could involve:
- Incorporating larger and more diverse datasets.  
- Testing advanced neural network architectures (e.g., CNNs, LSTMs).  
- Applying explainable AI techniques to understand how features contribute to predictions.  

---

## Libraries Used
- **Python 3.10+**
- **Pandas** (data cleaning & wrangling)  
- **NumPy** (numerical operations)  
- **Matplotlib & Seaborn** (visualizations, EDA)  
- **Scikit-learn** (SVM, MLP, model evaluation)  

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/alzheimers-prediction-ml.git
