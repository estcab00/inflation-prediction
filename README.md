# Inflation forecasting with Machine Learning Methods in Peru

## 1. Introduction:
This repository contains the code for a resarch on inflation forecasting with different Machine Learning methods in Peru.

## 2. Repository Structure
The repository is divided as follows:
- ```code```: This folder contains two subfolders ```headline_inflation``` and ```core_inflation```. In each one, there is the corresponding code for the prediction task for two periods ```2019``` and ```2023```. Each one of this folders is divided in this sections:
  -  ```1_DataExtraction_###.ipynb```: In this notebook we use the API interface of the Central Bank of Reserve of Peru (BCRP) to extract our data. We do the corresponding transformations to each series and append them in a dataframe ```df_raw_###.csv```, which contains contemporary variables and will be used for visualization, and ```df_lags_###.csv```, which additionally contains lagged variables and will be used for prediction tasks.
  -  ```2_DataVisualization_###.ipynb```: In this notebook we load the files created at the first notebook and do different visualization techniques in order to analize and understand more our data and the relationship between the variables. We do a pairplot and some heatmaps, as well as a graph of our input variables.
  -  ```3_Regression_###.ipynb```: This notebook contains the different regression and prediction tasks for all models

- ```input```: CSV files created at the ```1_DataExtraction_###.ipynb``` notebook and later used in other notebooks are saved here.
- ```output```: The results from the each jupyter notebook are saved here in a corresponding folder.
- ```modules```: Our functions are defined here.
- ```report```: The PDF of the research is found here.

The ```###``` at the end of each jupyter notebook indicates in which subfolder the notebook is located.
- ```C##```: core_inflation
  - ```C19```: core_inflation/2019
  - ```C23```: core_inflation/2023
- ```H##```: headline_inflation
  - ```H19```: headline_inflation/2019
  - ```H23```: headline_inflation/2023
 
The same interpretation can be used for ```###``` at the end of ```.csv``` and ```.png``` files in the ```output``` folder to determine the corresponding variable and year. 

This repository is updated by @estcab00
