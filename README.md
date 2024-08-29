# Inflation forecasting with Machine Learning Methods in Peru

## 1. Introduction:
This repository contains a resarch on inflation forecasting with different Machine Learning methods in Peru. We are predicting headline and core inflation for two periods: 2019 and 2023. 

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

## 3. Methodology

### 3.1 General Framework

Given the monthly price level \( P_{t} \), we define the monthly inflation as \( \pi_{t} = 100 \times (\ln(P_{t}) - \ln(P_{t-1})) \). Let's assume a \( K \times 1 \) vector \( X_{t} \) of predictors. Our objective is to predict inflation \( h \) periods forward \( \pi_{t+h} \), which can be viewed as:

\[
\pi_{t+h} = F_{h}(X_{t}, \theta) + \epsilon_{t+h}
\]

where \( h = 1, \ldots, H \) is the forecast horizon. \( F(\cdot) \) represents the relationship between inflation and its predictors, which could be either linear or non-linear, depending on the model being used. \( \theta \) represents both the parameters and hyperparameters of the ML models, and \( \epsilon_{t+h} \) is the forecast error.

#### 3.1.1 Forecasting Procedure

To perform the analysis of the predictive power of the different ML models, we first standardize the data to ensure all features are on the same scale. We divide our data into two consecutive sub-samples: training and testing. In the training sample, we will both fit the model and calibrate the hyperparameters. To do that, we implement a time series cross-validation, which, unlike other forms of cross-validation, considers the structure of the time series. The data is split into \( k \)-folds. Each iteration, the \( j \)-fold is used as the validation set, and the remaining \( j-1 \) folds are used as the training set. The optimization parameters in \( \theta \) are chosen to minimize a metric or loss function in a process called hyperparameter optimization. It can be described as follows:

1. Given a set of hyperparameters in the hyperparameter space \( \theta \in \Theta \), we define a grid containing a set of \( l \) hyperparameters to evaluate \( G = \{\theta_{1}, \theta_{2}, \ldots, \theta_{l}\} \).

2. We train the model using the training data and the hyperparameters \( \theta_{i} \) \( M_{i} = M(\theta_{i}, D_{\text{training}}) \).

3. We evaluate the performance of the model with the validation set using the performance metric \( L_{i} = L(M_{i}, D_{\text{training}}, D_{\text{validation}}) \).

4. The hyperparameter is chosen to minimize the performance metric (e.g., MSE) \( \theta^{*} = \arg\min_{\theta_{i} \in G} L(M(\theta_{i}), D_{\text{training}}, D_{\text{validation}}) \text{ subject to } \theta_{i} \in G \text{ for } i \in \{1, 2, \ldots, l\} \).

The process is repeated until all folds have been used to calibrate the tuning parameters. The final hyperparameters are those that, on average, minimized the metrics across the different folds. This means that the ML model is trained a total of \( l \times k \) times. Once the model has been calibrated, we use it to perform out-of-sample forecasting in the testing sample. Ultimately, we rank the models based on their performance in the out-of-sample forecast.



This section provides an overview of the Machine Learning methods employed in the research, including their implementation and evaluation strategies. We are considering three econometric models (RW, VAR, ARIMA) and four machine learning models (LASSO, Ridge, EN and RF). Regarding the comparison methods, we are using the root mean square error (RMSE) and mean absolute percentage error (MAPE) of the preditec values against the real values.

### 3.1 Linear ML Models
The linear machine learning models are LASSO and Ridge Regression

Given the following linear regression model

$$y_t = \beta x_t + \epsilon_t$$
where $y$ is an Nx1 vector of dependant variables, X is an NxK matrix of explanatory variables, $\beta=(\beta_1 \ , ... \ , \beta_k)$ is a vector regression coefficients and $\epsilon$ is a vector of errors. It is possible that $K$ is relatively large compared to $N$. In those cases, the LASSO estimates are chosen to minimize

$$LASSO = \min_{\beta}(\sum_{i=1}^N (y_i - \sum_{j=1}^K \beta_j x_ij )^2 + \lambda \sum_{i=1}^K|\beta_i|)$$

where the term $\lambda \sum_{i=1}^K\beta_i^2$ is a regularization of type $\ell_2$ and $\lambda$ is the tuning parameter.

while the Ridge estimates minimize

$$Ridge = \min_{\beta} ( RSS + \lambda \sum_{i=1}^K\beta_i^2)$$

where the term $\lambda \sum_{i=1}^K\beta_i^2$ is a regularization of type $\ell_2$ and $\lambda$ is the tuning parameter.

### 3.2 Non-linear ML Model
The non-linear machine learning models used is a Random Forest Regressor. 

Let us assume a regression tree model in the form
$$y = \sum_{m=1}^M c_m \cdot 1_{(x \in R_m)}$$
where $(R_1, . . .,R_M)$ correspond to the partition regions for the observations. To construct the regression tree, a set of possible values $(x_1, ..., x_p)$ is split into $M$ possible non-overlapping regions $(R_1, . . .,R_M)$. Then, for every observation that falls into the region $R_m$, we will make the same prediction, which is the average of the response values for the
training observations in $R_m$.

In the context of RF, each time a split in a tree is done, a random sample of $m$ predictors is chosen as split possible candidates from the full set of $M$ predictors. This split is allowed to use only one of those $m$ predictors. Usually, the number of predictors assessed at each split is approximately the square root of the total number of predictors $M$, meaning $m=\sqrt{M}$, which differentiates RF from bootstrapping, where the split considers the full sample $m=M$ each time. The RF way of splitting predictors will typically be useful when we have a large number of correlated predictors in our dataset, which could be the case of inflation.

### 3.3 Model implementation

All models are implemented using the Scikit-Learn and XGBoost package in Python. Linear models are imported as the Lasso,  Ridge, and ElasticNet functions respectively. Non-linear models are imported from RandomForestRegressor and XGBRegressor. All models are implemented with a random state = 2024. A cross validation followed by a grid-search is implemented using the TimeSeriesSplit and GridSearchCV modules from
Scikit-Learn.

### 3.4 Model comparsion
Given a $X_t$ vector of k explanatory variables for the variable $y_t$ for $(t = 1,...,T)$ we can construct the point forecast of $y_{t+h}$ given the information $I_h$
The forecast in $t+h$ is
$$y_{t+h}=f_{t+h} (X_{t+h}, I_h)$$

And we can define the forecast error as
$$e_{t+h}=y_{t+h}-\hat{y}_{t+h}$$

Therefore, the RMSFE is defined by
$$RMSFE = \sqrt{MSFE} = \sqrt{\frac{1}{T} \sum_{t=1}^N e_{t+h}^2}$$

and the MAPE is defined as

$$MAPE = \frac{1}{T} \sum_{t=1}^T |\frac{e_{t+h}}{y_{t+h}}|$$

## 4. Update
This repository is updated by Esteban Cabrera. Acknowledgement of QLAB-PUCP for the help provided during this investigation.  
