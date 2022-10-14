<h1 align="center">
  <br>
Heart Attack Risk Prediction Research <br> + <br> Auto ML <br>

</h1>


<h3 align="center">
  Built with
  <br>
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" height="30">
    <img src="https://github.com/boramorka/usercontent/blob/main/heart-risk/EvalML.png?raw=true" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/matplotlib.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/numpy.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/pandas.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/scikit-learn.svg" height="30">
    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=green" height="30">
</h3>

<p align="center">
  <a href="#main-goals-of-this-project">Main goals of this project</a> •
  <a href="#data">Data</a> •
  <a href="#eda">EDA</a> •
  <a href="#models-building">Models building</a> •
  <a href="#verdict-about-standart-models">Verdict about standart models</a> •
  <a href="#auto-ml">Auto ML</a>
</p>

## Main goals of this project:
* Build and evaluate the best algorithm for heart attack risk prediction
* Apply Auto ML tool from Eval ML to automate model building
* Compare results of custom models and Auto ML

## Data
Link: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

### About this dataset
- Age : Age of the patient
  
- Sex : Sex of the patient
- exang: exercise induced angina (1 = yes; 0 = no)
- ca: number of major vessels (0-3)
- cp : Chest Pain type chest pain type
  - Value 1: typical angina
  - Value 2: atypical angina
  - Value 3: non-anginal pain
  - Value 4: asymptomatic
<br><br>
- trtbps : resting blood pressure (in mm Hg)
- chol : cholestoral in mg/dl fetched via BMI sensor
- fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- rest_ecg : resting electrocardiographic results
  - Value 0: normal
  - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
  - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
<br><br>
- thalach : maximum heart rate achieved
- target : 0= less chance of heart attack 1= more chance of heart attack

![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/Screenshot_1.png?raw=true)

## EDA

### Correlations in data
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/output2.png?raw=true)
### Age distribution
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/3.png?raw=true)
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/4.png?raw=true)
### Types of pain
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/5.png?raw=true)
### ECG Data distribution
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/6.png?raw=true)
### Pairplot by all features
- Orange: 1 (more chance heart attack)
- Blue: 0 (less chance heart attack)
Full Size Pic: https://github.com/boramorka/usercontent/blob/main/heart-risk/7.png?raw=true

![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/7.png?raw=true)
### Min and Max blood pressure fistributions
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/8.png?raw=true)
### Cholesterol distribution
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/9.png?raw=true)


## Models building

### Logistic Regression
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/10.png?raw=true)
### Decision Tree
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/11.png?raw=true)
### Random Forest
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/12.png?raw=true)
### K Nearest Neighbours
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/13.png?raw=true)
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/14.png?raw=true)
### Support Vector Machine (SVM)
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/15.png?raw=true)
### Adaboost  Classifier
![pic](https://github.com/boramorka/usercontent/blob/main/heart-risk/16.png?raw=true)
### Verdict about standart models
**Every life matters!** So if we make a False Negative mistake it will cost us much more! We need to chose model with best True Positive results. And it's a:
- KNN Models
- SVM models

## Auto ML

EvalML is an open-source AutoML library written in python that automates a large part of the machine learning process and we can easily evaluate which machine learning pipeline works better for the given set of data.

- Eval ML Library will do all the pre processing techniques for us and split the data for us
- There are different problem type parameters in Eval ML, we have a Binary type problem here, that's why we are using Binary as a input
- It has awesome API

```python
#train/test split
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(x, y, problem_type='binary')

#all types of problems
evalml.problem_types.ProblemTypes.all_problem_types

```

```
output:
[<ProblemTypes.BINARY: 'binary'>,
 <ProblemTypes.MULTICLASS: 'multiclass'>,
 <ProblemTypes.REGRESSION: 'regression'>,
 <ProblemTypes.TIME_SERIES_REGRESSION: 'time series regression'>,
 <ProblemTypes.TIME_SERIES_BINARY: 'time series binary'>,
 <ProblemTypes.TIME_SERIES_MULTICLASS: 'time series multiclass'>]
```

### Running the Auto ML to select best Algorithm

```python
from evalml.automl import AutoMLSearch
automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')
automl.search()
```

### Building an AutoML model
```python
automl_auc = AutoMLSearch(X_train=X_train, y_train=y_train,
                          problem_type='binary',
                          objective='auc',
                          additional_objectives=['f1', 'precision'],
                          max_batches=1,
                          optimize_thresholds=True)

automl_auc.search()
```

### Display rankings and best pipeline
```python
automl.rankings

best_pipeline = automl.best_pipeline
best_pipeline

automl.describe_pipeline(automl.rankings.iloc[0]["id"])

best_pipeline.score(X_test, y_test, objectives=["auc","f1","Precision","Recall"])
```

### Result
**We got an 91.2 % AUC Score which is the highest of all**