# Alzheimer's Disease Prediction Using Machine Learning

## Overview
This project explores how machine learning can assist in predicting Alzheimer’s Disease (AD) based on health, lifestyle, and cognitive information. Alzheimer’s is the leading form of dementia worldwide, and early detection remains a major challenge in improving care and outcomes. By applying predictive models on clinical and behavioral data, this project demonstrates how data-driven approaches can provide insights for healthcare research and early intervention strategies.

The project uses a dataset of over 2,100 patients and compares different machine learning models, specifically **Support Vector Machines (SVMs)** and **Multi-Layer Perceptrons (MLPs)**, to assess their ability to classify Alzheimer’s vs. non-Alzheimer’s patients.




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

- Checked for missing values.  
- Generated **correlation heatmap** to identify strong predictors.  
- Selected **Top 10 Features** most correlated with Alzheimer’s (both positive and negative).  
- Created a refined dataset for modelling with these features.  
- Standardized features where necessary for SVM and MLP models.  

 **correlation heatmap** 
 
 <img width="479" height="321" alt="image" src="https://github.com/user-attachments/assets/e35f6d4f-b6cf-421f-9ba3-e922236fceb8" />
 
  **Updated dataframe with new features** 

 <img width="743" height="215" alt="image" src="https://github.com/user-attachments/assets/ae138c2c-117f-4fe5-9897-40c6542e87c3" />

---

## Machine Learning Models

### Support Vector Machine (SVM)
- **Linear Kernel (Test 1: Unscaled):** Achieved ~80% accuracy but struggled with recall (false negatives high).
  <img width="764" height="198" alt="image" src="https://github.com/user-attachments/assets/f5608efa-30af-4371-9f6b-ecc400c12cc1" />
<img width="454" height="72" alt="image" src="https://github.com/user-attachments/assets/15abfd35-e85c-4714-830a-059686aa4fac" />
<img width="595" height="285" alt="image" src="https://github.com/user-attachments/assets/4971030d-5ea5-4a37-99e4-d778f32803e6" />
<img width="867" height="629" alt="image" src="https://github.com/user-attachments/assets/05516173-7b10-4012-af85-16c5c5768c12" />


 



- **RBF Kernel (Test 2: Scaled):** Improved test accuracy to 85%, reduced false negatives, but showed signs of overfitting.  

<img width="600" height="440" alt="image" src="https://github.com/user-attachments/assets/ebc96d97-9a50-412d-88e4-0f211852a8c4" />
  <img width="588" height="219" alt="image" src="https://github.com/user-attachments/assets/ec7f5632-186a-45e9-bdd7-9120399f42d0" />
<img width="620" height="639" alt="image" src="https://github.com/user-attachments/assets/623510b2-f17a-4bac-8892-e6ff698789cd" />

#### SVM Justification:

- Performs well on structured, tabular datasets like this one with a moderate number of features and observations

- Effective at handling both linear and nonlinear boundaries through kernel tricks such as the RBF kernel

- Well-suited for binary classification problems and benefits from standardized features, which were applied during preprocessing

### Multi-Layer Perceptron (MLP)
- **Basic MLP (Test 1: One hidden layer):** ~86% accuracy, decent recall but still missed some cases.  
<img width="620" height="257" alt="image" src="https://github.com/user-attachments/assets/aba22583-c437-48c6-9ce7-4db53b3faebf" />

<img width="620" height="588" alt="image" src="https://github.com/user-attachments/assets/1633da9a-4e9a-47ae-a52b-b10023a7a1b7" />

<img width="620" height="621" alt="image" src="https://github.com/user-attachments/assets/29699c87-a61a-47a8-8cdd-5dd54c8acf0b" />



- **Tuned MLP (Test 2: Two hidden layers, adjusted learning rate):** ~90% accuracy, balanced precision and recall, best overall performance.  

<img width="620" height="122" alt="image" src="https://github.com/user-attachments/assets/2ebbee88-948b-4fe7-9cfd-ff51d3abbca2" />

<img width="620" height="590" alt="image" src="https://github.com/user-attachments/assets/78693308-0a9c-40b7-9fb7-fc29fcd67725" />

<img width="576" height="190" alt="image" src="https://github.com/user-attachments/assets/ac2d8b1b-22c9-4bd4-a872-13dda8978eb9" />

<img width="600" height="620" alt="image" src="https://github.com/user-attachments/assets/f6ededf1-e9d9-48ae-9375-a983c9e0153c" />

#### MLP Justification:

- Performs well on structured datasets like this one, especially when features are a mix of standardized numeric and categorical variables

- Capable of modeling complex, nonlinear relationships between cognitive, behavioral, and lifestyle-related features

- Learns hierarchical feature interactions through multiple hidden layers, making it more flexible than linear models

- Suitable for binary classification tasks such as Alzheimer's diagnosis prediction and benefits significantly from feature scaling, which was done during preprocessing


## Model Comparison
- **SVM (Linear):** Simplest, but weakest results (~80% test accuracy).  
- **SVM (RBF + Standardization):** Stronger but prone to overfitting.  
- **MLP (Default):** Good results with convergence shown in loss curve.  
- **MLP (Tuned):** Best-performing model with ~90% test accuracy and balanced sensitivity.  

## Model comparison chart

<img width="546" height="412" alt="image" src="https://github.com/user-attachments/assets/72bad492-9592-4b99-a9aa-1aaec8f1674b" />

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

## Acknowledgments

Dataset by Rabie El Khaoura on Kaggle.

Guidance from Prof. Afraz S. (Machine Learning course, Mohawk College).

Project members: Younes El-Kaisi, Yassin Elhakeem, Youssef Ayoub.

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn.

Inspiration from healthcare ML research in NCBI


