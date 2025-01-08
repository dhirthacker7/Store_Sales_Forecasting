# Store Sales Forecasting | Numpy, Pandas, Matplotlib, Seaborn, Scikit-Learn, XGBoost, CatBoost

This project aims at creating a robust forecasting model to predict daily retail sales, enabling retailers to make informed decisions and optimize their operations in an economically volatile environment, primarily affected by fluctuating oil prices.

### Table of Contents
1. [Introduction](#introduction)
2. [Project Motivation](#project-motivation)
3. [Project Goals](#project-goals)
4. [Methodological Approach](#methodological-approach)
5. [Dataset Overview](#dataset-overview)
6. [Setup and Installation](#setup-and-installation)
7. [How to Use](#how-to-use)
8. [Detailed Results and Analysis](#detailed-results-and-analysis)
9. [Concluding Remarks](#concluding-remarks)
10. [References and Additional Resources](#references-and-additional-resources)

---

## Introduction

This project is centered around the challenges of retail management in Ecuador, a country where the economy is highly influenced by oil prices. These economic fluctuations impact consumer purchasing power, creating a volatile market for retailers. Our forecasting models are designed to predict sales with high accuracy, using historical sales data intertwined with promotional activities and economic indicators like oil prices and holidays.

## Project Motivation

The motivation behind this project is to empower retailers with predictive insights that allow them to stay ahead in a rapidly changing market. By integrating advanced data analytics and machine learning techniques, we transform raw data into actionable insights that drive strategic business decisions.

## Project Goals

The goal is to develop a sophisticated predictive tool that not only adapts to but also anticipates market changes, providing clear, actionable daily sales predictions. This tool is expected to enhance operational efficiency and profitability across various retail outlets.

## Methodological Approach

### Data Preparation
- **Data Cleaning:** Initial steps include loading the data into a suitable format, removing duplicates, and correcting inconsistencies.
- **Handling Missing Data:** Techniques such as interpolation for continuous variables and imputation for categorical variables are used to address gaps in the dataset.

### Exploratory Data Analysis (EDA)
- **Visual Analytics:** Utilizing plots like histograms, scatter plots, and box plots to understand distributions and relationships.
- **Statistical Analysis:** Conducting correlation analysis and hypothesis testing to validate assumptions and insights.

### Feature Engineering
- **Temporal Features:** Incorporation of time-based features to capture seasonal and cyclical patterns.
- **Advanced Features:** Developing lag features and rolling averages to better capture trends and smooth out noise in the data.

### Model Development and Evaluation
- **Model Selection:** Employing various statistical and machine learning models including Linear Regression, Random Forest, and XGBoost.
- **Performance Metrics:** Using RMSE and MAE for quantitative model evaluation, alongside visual comparisons of predicted vs. actual sales data.

## Dataset Overview

We utilize a comprehensive set of data from various sources, structured into several CSV files as described below:

### Data Files
- `train.csv` - Training dataset with historical sales data.
- `test.csv` - Test dataset used for model evaluation.
- `sample_submission.csv` - A template for submitting predictions.
- `stores.csv` - Information about store types and locations.
- `oil.csv` - Daily oil prices as an economic indicator.
- `holidays_events.csv` - Details on holidays and events affecting sales.

## Result and Analysis 
## Linear Regression with Time Series

Linear Regression was applied to capture linear trends over time, treating the 'time' variable as a direct predictor for sales. This approach is based on the assumption that sales exhibit a linear progression as time advances. Additional features like lag variables were incorporated to utilize historical sales data as a basis for predicting future values, an essential aspect of time series forecasting due to the chronological nature of sales patterns.

### Time Plot of Total Store Sales

![Time Plot of Total Store Sales](https://github.com/dhirthacker7/Store_Sales_Forecasting/blob/main/images/LR_TimeSeries.png)

The graph titled "Time Plot of Total Store Sales" displays two sets of time-series data: the actual sales (in grey) and the predicted sales (in blue), over a span from January to April 2021. Each point on the time series represents a daily sales figure, with the actual sales depicted as a lighter grey line with a dot marker for each day, and the predicted sales shown as a solid, darker blue line.

The fluctuations in the grey line show the variability in actual daily sales figures, ranging from as low as 100 to as high as 500. The blue line, representing the predicted sales, also varies each day but is plotted with a thicker line, making it visually distinct from the actual sales.

### Importance of Lag Features and Rolling Averages

Lag features and rolling averages were critically engineered to capture temporal trends and smooth out volatile sales data. Lag features provide the model with access to previous sales points, vital for recognizing sales continuity or change, while rolling averages dampen the noise and reveal more stable trends within the fluctuating sales figures.

Ridge regression was employed to address multicollinearity among features and prevent overfitting, due to its regularization capabilities. The model's alpha parameter, which governs the regularization strength, was optimized using GridSearchCV to ensure the most effective shrinkage of the coefficients, enhancing the model's prediction accuracy on unseen data. This approach was particularly beneficial given the large number of features generated from the extensive feature engineering process. The entire process is encapsulated within a pipeline for standardized scaling and efficient hyperparameter tuning.

### Comparison of Models

The plot below compares actual sales with the sales predicted by two different models over time. Actual sales are represented by a semi-transparent pink line, while the predictions from Model 1 and Model 2 are shown as a dashed red line and a dotted green line, respectively. This visual comparison allows for the assessment of each model's accuracy, with the proximity of the predicted lines to the actual sales indicating the level of precision. Deviations between the predictions and actual sales provide insights into the models' performance, highlighting areas for potential improvement. The plot is a visual tool for evaluating and contrasting the forecasting capabilities of the two predictive models.

![Lag Features vs Rolling Averages](https://github.com/dhirthacker7/Store_Sales_Forecasting/blob/main/images/LR_Actual_vs_pred.png)


## XGBoost Model Optimization

XGBoost was employed for its exceptional ability to model complex, nonlinear relationships inherent in the data. The optimized parameters for XGBoost, determined through a rigorous hyperparameter tuning process, included a colsample_bytree of 0.9, a learning_rate of 0.2, a max_depth of 5, and n_estimators of 300, with a subsample rate of 0.9. These settings were instrumental in enhancing the model's accuracy and its ability to generalize beyond the training dataset.

**Before Hyperparameter Tuning:**
- RMSE: 569.158529113529

**After Hyperparameter Tuning:**
- RMSE: 546.5577987964145

## Ensemble Techniques: Random Forest and Decision Trees

### Random Forest

The Random Forest model, an ensemble of decision trees, was utilized for its robust performance in high-dimensional spaces. Its structure is ideal for capturing complex interactions between variables. The model was fine-tuned using cross-validation techniques to identify optimal hyperparameters, ensuring a balance between the model's bias and variance, thereby enhancing its predictive performance on unseen data.

**Hyperparameter Tuned Performance:**
- RMSE: 555.1503312622295

### Decision Trees

Decision Trees served as a straightforward, interpretable model for sales forecasting. The best parameters for the Decision Trees were established through GridSearchCV, which conducted exhaustive searches across a predefined parameter grid to optimize model performance. This process ensured that the model was neither overfitted nor underfitted, providing reliable and understandable predictions.

**Hyperparameter Tuned Performance:**
- RMSE: 586.6535011376407

## Conclusion

Based on the RMSE scores obtained from the cross-validation process, we can conclude that the XGBoost model outperforms the other three models with the lowest RMSE mean of 538.97, indicating that it is the most accurate in terms of prediction among the tested models for this dataset. The ensemble methods like XGBoost and Random Forest are more effective for this dataset compared to a single decision tree or a linear approach.

![Comparison plot of ML Models](https://github.com/dhirthacker7/Store_Sales_Forecasting/blob/main/images/Comaprison_models.png)

``

## Setup and Installation

To set up the project environment

```bash
# Clone the repository
git clone <repository-url>
# Navigate to the project directory
cd <repository-name>
# Install required Python packages
pip install -r requirements.txt
```
