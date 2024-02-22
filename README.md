# The-Endless-Line


Our analysis focuses on the waiting time experienced at the Port Aventura park. 
The project is structured as follows: 

## A. Data files cleaning, merging, & feature engineering :

### I- Cleaning & Merging 
We were provided with several files that gathered information regarding the Weather, Client types & transactions, Attraction attributes,... 
and we merged them to get a comprehensive data repository of all the key information we needed to have. 
We created two main files, of which the only difference is the granularity of the data: 15 min, and 1 day. The latter was computed as a daily average of all relevant attributes. 

### II- Feature Engineering: 
Most notable was the treatment of noise by making the data more robust for further statistical analysis. The strategy is shown in the function 'robustization',
where we simply disregard data that falls 4.5 standard deviations away from the mean. This number may seem arbitrary, but we tested several values (2, 2.5, 3, 3.5, 4, 4.5, and 5)
on our Random Forest model and selected the one with the best result.
We also droped days for which the park was close, and treated any non-stationarity issues for continuous variables with the ADF test
Util functions used to correct stationarity also appear in the ipynb file "Feature Engineering P1". 
Two notable categorical variables were added : SEASONALITY (summer, winter, autumn, and spring) and DAY PERIOD (morning, afternoon, evening) for further usage. 
Regarding categorical variables, we hot-encoded them as dummy variables. 

Because the file is so big (800Mo), we could not upload it in our github's data folder. 

## B. Exploratory Data Analysis 
Unearths key dynamics via various graph plots correlation heatmaps between our main selected variables.
Our key insights can be found in the ipynb file named EDA, which leverages the work done in the previous section to uncover reliable and meaningful statistics. 


## C- Modelling:
  1)   In the file 'Modeling.ipynb' we try to find a model that estimates well the waiting time. 
       Our final selection is a Random Forest for which the error drops below 10%. 
       Other models were tried below the results for Random Forest, but the results were not convincing and thus the model not selected.
  2)   In the file 'Log.ipynb', we study the feature importance for each independent variable (with respect to each & all attractions) to understand which
       factors explain best the waiting time (target variable) and gain insights on their quantitative impact on the latter.
       We use a log-log OLS model to find the best-fitting regression equations that explain the tendency observed in our target variable, 'WAIT TIME'. 
       This part is key as our dashboard exhibits the effect that changing the level of important features has on our target variable.
       
## D- Forecasting:
Use of Prophet for forecasting based on our available historical data. 
Found in the file "Prophet.py"


## E- DASHBOARD: 
To visualize our dashboard, please run the **app.py** and then click on the webapp URL on the terminal.

https://github.com/Behachee/The-Endless-Line/assets/140748662/b9d92c8e-3beb-47d7-b54b-8707aa306bd5




