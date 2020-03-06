# NHANE-CVD-Death-Prediction
This Project is to predict Cardiovascular Disease (CVD) Death based on NHANE Open data (National Health and Nutrition Examination Survey, https://wwwn.cdc.gov/nchs/nhanes/default.aspx). By adding **Nutrition Dataset**, the model performance is increased by **8%**. This project is awarded as **First Prize** in the SCOR Datathon in Paris, France in February 2020. <br>

![alt text](https://user-images.githubusercontent.com/52146042/76103162-80af5800-5fd1-11ea-81d3-7b1567389bf2.png)

We only consider people above 20 years old with no CVD prior and death from no CVD is treated as censored.

### Data
All the data is crawled from the offical website https://wwwn.cdc.gov/nchs/nhanes/default.aspx by R, with the name of './data.csv'. (This dataset is merged by Morality, Demographics, Occupation, Laboratory, Medical Condition, Medical Examination, Diabete, Smoking, Dietary). 
1. The folder [Model Data](https://github.com/LakiLiu/NHANE-CVD-Death-Prediction/tree/master/Data/Model_Data) is the data used by the model training, where is Data with Nutrition dataset and Data without nutrition dataset. Also, since the model considers using KNN to impute missing values, and sometimes it is time-consuming, I also upload the dataset after KNN imputation.
2. The folder [tempdata](https://github.com/LakiLiu/NHANE-CVD-Death-Prediction/tree/master/Data/tempdata)is the dataset temporarily saved during the scripts.
  

### Model:
#### A. Classic Models
1. Gradient Boosting
2. LGBM
3. Random Forest
4. Logistic Rression
#### B. Survival Analysis
1. Cox proportional hazard model
2. Gradient boosting for survival analysis

### Evaluation Metrics
1. Gradient boosting for survival analysis
2. Cumulative dynamic AUC for right-censored time-to-event data.


This repo is composed by two parts, Data and JupyterNote scripts. 

**Scipts:** Three scripts is included. 
1. [Data Exploration](https://github.com/LakiLiu/NHANE-CVD-Death-Prediction/blob/master/JupyterScript/Data_exploration.ipynb) is to explore the insights from the dataset, with a large volume of data visulisation. 
2. [Classic_Model](https://github.com/LakiLiu/NHANE-CVD-Death-Prediction/blob/master/JupyterScript/Classic_Models.ipynb) used the classic Machine Learning Algorithms to predict the probability of death. 
3. [Survival_Analysis](https://github.com/LakiLiu/NHANE-CVD-Death-Prediction/blob/master/JupyterScript/Survival_Analysis.ipynb) applied the survival analysis to improve the model performance, with the final results shown. 
