# Employee Salary Prediction using Machine Learning Regression Algorithms

Created a regression model with **MSE score of 355** on employee salary data given their corresponding qualifications and role descriptions to help employers/recruiters determine suitable salaries for potential candidates. 

# Data & Problem
* **Problem**: The goal of this problem is to accurately predict salaries given known salaries of professions based on job descriptions so that recruiters can hire suitable candidates for job roles. With the help of machine learning regression algorithms implemented in this notebook, the recruitment company will be able to offer candidates appropriate and compettitive salaries while staying within budget. In a statement,
> Given employee job descriptions and qualifications, can recruiters predict salaries and offer competitive salaries for potential job candidates?

* **Data**: The raw dataset for this problem is split into 3 smaller datasets as CSVs. There is 1 training dataset containing the description of the employee as stated above, and their corresponding salary. From this training dataset, 20% was split into a seperate test dataset with corresponding employee salary in order to compute the accuracy and error of the models. In order to test the model, predictions will be made on the third dataset, the testing dataset, which comprises of job descriptions for an employee with no corresponding salaries. This testing dataset will act as a substitute to real-world data so that the model can be used for salary prediction. Data Dictionary - The features (job descriptions for employees) used to predict the target variable (employee salary):

  1. jobType - position held (CEO, CFO, CTO, Vice President, Manager, Janitor, and senior or junior position)

  2. degree - type of degree (Doctoral, Masters, Bachelors, High School, or None)

  3. major - type of major (Biology, Business, Chemistry, Computer Science, Engineering, Literature, Math, Physics, or None)

  4. industry - type of industry (Auto, Education, Finance, Health, Oil, Service, or Web)

  5. yearsExperience - experience as number of years

  6. milesFromMetropolis - distance, in miles, from a metropolis

* **Success Metric**: To reduce the difference between predicted salaries and actual employee salaries through minimizing/lowering the **Mean Squarred Error (MSE)** of our model's salary predictions to **below 360 (MSE < 360)**. 

# Code and Resources Used
* **Python Version**: 3.8
* **Environment**: Miniconda, Jupyter Notebook
* **Packages**: Pandas, Scikit-Learn, NumPy, Matplotlib, Seaborn, Joblib

# EDA & Feature Engineering
After loading the data, inspecting the dataset's features (employee qualifications and role description) and target variable (salary), and merging the employee description (features) dataset with the salary (target) dataset on the 'jobID` feature (common to both), I needed to clean the data and visualize the data to better understand the relationship between the features, and the target. I did the following steps with the datasets:
* Found the number of unique values in the dataset through creating a dictionary.
* Checked for duplicate values in the dataset and found **no duplicate employees** in the dataset.
* Checked for total number of missing values in each column and found that there are no missing values in this dataset.
* Removed salaries that were listed at or below $0 because they were all industry-based advanced roles that pay well. Having employees with these positions and $0 salaries would severely impact our predictions.
* Explored target variable (salary) through assessing normality of the variable and finding outliers.
  - Histogram:
  - 
  ![salaryHist](https://user-images.githubusercontent.com/46492654/190313235-bf1aba3b-493b-45f6-8095-c906409145a8.png)
  
  - Box and Distribution Plots:
  ![salaryBoxDensPlot](https://user-images.githubusercontent.com/46492654/190313285-b257c551-2362-4ba4-993f-70627a870702.png)

* Computed the skew and kurtosis for the target variable. Found that since the skewness was less than 0.5, the distribution of the salary was very symmetric, and the low kurtosis value indicated that the data did not have many outliers.
* Determined that outlier salaries found through IQR inspection were above the upper bound. An overwhelming majority of salaries that exceeded the upper bound were being earned by employees who retained advanced positions. Consequently, their salaries being high made sense, and they were not removed from the dataset. 
* Created univariate data visualizations of all the features through plotting:
  - Bar Plots:
  ![bargraphs](https://user-images.githubusercontent.com/46492654/190313339-7c9ccd85-e3e3-48dc-acd8-3eda1db376a1.png)
  
  - Box and Density Plots:
  ![boxanddensplots](https://user-images.githubusercontent.com/46492654/190313402-4911213c-f01c-4d12-b32a-30def873dd31.png)
  
* Created bivariate data visualizations comparing continuous and categorical features to target variable:

  - Scatter Plots:
  
  ![salaryVsexperience](https://user-images.githubusercontent.com/46492654/190313475-5bd6c12b-0f1a-45b1-b24e-cb86ac0ac4b3.png)
  
  ![salaryVsmetropolis](https://user-images.githubusercontent.com/46492654/190313517-999d9e53-165b-4a4e-a34a-733128802094.png)
  
  
  - Violin Plots:
  
  ![salaryVseducation](https://user-images.githubusercontent.com/46492654/190314032-f7707693-9833-49c2-9e62-f10fb17bb610.png)
  
  ![salaryVsindustry](https://user-images.githubusercontent.com/46492654/190314087-bef3197b-e282-4126-9483-0c5dd4c85cf9.png)
  
  ![salaryVsmajor](https://user-images.githubusercontent.com/46492654/190314124-47c8da0e-89c7-4e2d-b013-68db3af8dfba.png)
  
  ![salaryVsposition](https://user-images.githubusercontent.com/46492654/190314168-d976fc3a-11e2-4d0f-9080-1498c80bfb5d.png)
  
  
  
* Removed useless categorical features, `jobID` and `companyID`, because they have no impact on employee's salary.
* Converted categorical features to numerical data via **OneHotEncoding**, so they can be used for salary prediction.
* Performed correlation analysis through creating a heatmap of all features, and the target:
![heatmap](https://user-images.githubusercontent.com/46492654/190313585-d4089252-9fe7-401b-a954-3918e3e69b16.png)

* Concluded from correlation analysis that no multicolinearity was present amongst the features (predictor variables), so none of the features were dropped. Moreover, the `yearsExperience` feature had the highest correlation with salary followed by `jobTypefeature`. This made sense because an employee's salary is highly dependent upon their experience, and the field they work in.

# Model Building
First, I split the data into X (features) and y (target - salary). Then, I split the X and y data into training and test sets with test size of 20%. As a baseline, I trained a **Linear Regression** model and evaluated the **R^2** and **MSE** of the training and testing data as primary evaluation metrics.

# Model Performance

**Summary of Baseline Model performance**:
* Predictions and MSE of training (**MSE ~383**) and testing data (**MSE ~385**) were **very similar**.
* The MSE was reduced while simultaneously improving R^2 through building upon the baseline model by implementing more regression algorithms and tuning their hyperparameters.
* Furthermore, the data was be standardized to account for the difference in scale between the features.
* Distribution plot of prediction vs. actual salaries for training data as obtained from Linear Regression Model:
![trainingPredsVsactual](https://user-images.githubusercontent.com/46492654/190313634-a1aeaa35-9d36-4fec-9881-e0786f1421df.png)

* Distribution plot of prediction vs. actual salaries for testing data as obtained from Linear Regression Model:
![testingPredsVsactual](https://user-images.githubusercontent.com/46492654/190313659-32dcf597-18f1-44c1-a72a-8034ff3085bb.png)

# Improving Model Performance
To improve the baseline model, the following 4 algorithms were implemented:
1. Lasso Regression 
2. Elastic Net Regression 
3. Ridge Regression
4. Random Forest Regressor
The best performing model was the **Random Forest Ensemble Regressor** which achieved an **MSE under 300 (MSE ~292)** on the training data, and achieved the goal of an **MSE under 380 (MSE ~369)** with the test data. This model was obtained by performing hyperparameter tuning with RandomizedSearchCV

Additionally, the features were also standardized to counter the different scales of the features (example: `milesFroMetropolis` and `jobType_CFO`). However, standardization had no impact on any of the model's MSE values, but the R^2 values were significantly lowered for the linear models. 

The best performing model (most accurate predictions) was the baseline **Random Forest model** with `max_samples = 1000` and `n_jobs = -1` with the non-standardized data. The **MSE scores** for this model were **292 on the training data** and **369 on the testing data** with **R^2 values of ~0.8 and ~0.75** for the training and test data respectively. This model was used to predict employee salaries given their job descriptions and qualifications but no corresponding salaries to verify the accuracy of the predictions.

# Feature Importance
Which features contribute the most towards a model predicting an employee's salary?
* `yearsExperience`, `milesFromMetropolis`, and `jobTypeJanitor` are by far the most important features for determining/predicting an employee salaries. This makes sense because salaries are highly dependent upon the experience an employee has as well as the distance they need to travel to get to work. Additonally, janitors earn very low salaries, and an employee being a janitor or not as a big impact on predicting what their potential salary could be.

* The least important features for predicting an employee's salary were the majors obtained by an employee, and if they held a CFO or CTO position. Typically, the major one pursues in school has little influence on salary because employers seek candidates with the right skills and experience. Additionally, employees that hold higher up positions are bound to earn high salaries, but this does not tell us about employees working in advanced fields with higher levels of education and more experience. These employees also have the potential to earn a lot.
![featureimportance](https://user-images.githubusercontent.com/46492654/190313710-c6d810fe-c9ec-442e-a512-a875981b5183.png)

# Conclusion

The **Random Forest model** with `max_samples = 1000` yielded the most accurate predictios with minimized error. With this model, an **MSE of 369** and **accuracy** of **75%**.

This model can be used by recruiters to determine an appropriate salary for an employee given their education, experience, industry of interest, and distance from a metropolis.

# Further Experimentation

According to [Scikit-Learn's algorithm selection suggestions](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html), given that we are trying to predict a quantity (salary) and have over 100k samples; the documentation suggests implementing the `SGD Regressor` algorithm. Additionally, we will also try using `Gradient Boosting Regressor` ensemble algorithm to try to reduce MSE. If any of the algorithms have promising results, we will try to tune their hyperparameters.

Achieved an even **lower MSE score of 355 (MSE <360)** for both the training and testing data with the **Gradient Boosting Regressor model**. This **model provided better predictions** than all other models used in this analysis and employers can employ this model for employee salary prediction over the finalized Random Forest model that was built earlier.

Extensive hyperparameter tuninng can be done with the ensemble techniques provided a sufficient grid size that covers a range of hyperparameter values, but this would be too computationally expensive and take too much time for the average computer to handle. One solution could be to train the models in batches given that the dataset contains 1000000 samples, but given the goal of getting an MSE under 380, the salary predictions were very accurate.
