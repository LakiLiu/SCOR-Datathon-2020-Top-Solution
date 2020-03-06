# NHANE-CVD-Death-Prediction
This Project is to predict Cardiovascular Disease (CVD) Death based on NHANE Open data (National Health and Nutrition Examination Survey, https://wwwn.cdc.gov/nchs/nhanes/default.aspx). By adding **Nutrition Dataset**, the model performance is increased by **8%**. This project is awarded as **First Prize** in the SCOR Datathon in Paris, France in February 2020. <br>

![alt text](https://user-images.githubusercontent.com/52146042/76103162-80af5800-5fd1-11ea-81d3-7b1567389bf2.png)

This repo is composed by two parts, Data and JupyterNote scripts. 

**Data:** All the data is crawled from the offical website https://wwwn.cdc.gov/nchs/nhanes/default.aspx by R, with the name of './data.csv'. (This dataset is merged by Morality, Demographics, Occupation, Laboratory, Medical Condition, Medical Examination, Diabete, Smoking, Dietary). The folder <tempdata> is the dataset temporarily saved during the scripts.
  
**Scipts:** Three scripts is included. 
1. [Data Exploration!](https://github.com/LakiLiu/NHANE-CVD-Death-Prediction/blob/master/JupyterScript/Data_exploration.ipynb) is to explore the insights from the dataset, with a large volume of data visulisation. 
2. The <Classic_Model> used the classic Machine Learning Algorithms to predict the probability of death. 
3. The <Survival_Analysis> applied the survival analysis to improve the model performance, with the final results shown. 
