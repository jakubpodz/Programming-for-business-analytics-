import numpy as np
import pandas as pd
from FREDMD_Tools import *

backtest_dates = pd.date_range(start='2005-1-1', end='2021-3-1', freq='M').to_period('M')

tf_data = pd.read_csv('2023-09-TF.csv', index_col=0, parse_dates=True)
nber_dates = pd.read_csv('NBER DATES.csv', index_col=0, parse_dates=True)

tf_data.index = pd.to_datetime(tf_data.index).to_period('M')
nber_dates.index = pd.to_datetime(nber_dates.index).to_period('M')

prob_results = pd.DataFrame(index=backtest_dates, columns=['LogisticRegression', 'SVC'])


# Since we are predicting 6 months ahead, we first “shift” the recession indicator data
# backward by 6 months so that for any given date t the indicator now corresponds to t+6.
nber_dates_shifted = nber_dates.copy()
nber_dates_shifted.index = nber_dates_shifted.index - 6


for t in backtest_dates:

    available_data = tf_data.loc[:t]

    #Remove series with less than 36 non-missing observations.
    valid_cols = available_data.columns[available_data.notnull().sum() >= 36]
    available_data = available_data[valid_cols]

    #Standardise each series: subtract its mean and divide by its standard deviation.
    #Fill any remaining missing values with zeros.
    standardized_data = (available_data - available_data.mean()) / available_data.std()
    standardized_data = standardized_data.fillna(0)

    
    #Compute 8 Principal Components (PCs) from the standardized data.
    pcs = pca_function(standardized_data, 8)
    
    #Align the PCs with the (shifted) recession indicator.
    # Only use dates that are in both the PC data and in nber_dates_shifted.
    common_dates = pcs.index.intersection(nber_dates_shifted.index)
    if len(common_dates) == 0:
        continue

    X = pcs.loc[common_dates]
    y = nber_dates_shifted.loc[common_dates]

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    #Fit the classification models using the supplied function.
    #The function fit_class_models() fits both models on the entire historical sample
    #(X and y) and then returns the predicted probabilities (using the last observation).

    lr_predict, svc_predict, lr_model, svc_model = fit_class_models(X, y)

    #Store the probability of recession.
    prob_results.loc[t, 'LogisticRegression'] = lr_predict[0, 1]
    prob_results.loc[t, 'SVC'] = svc_predict[0, 1]

print(prob_results)


#Extract the actual classification for each backtest date.
actual = nber_dates_shifted.iloc[:, 0].reindex(prob_results.index)

mapping = {"Expansion": 0, "Recession": 1}
actual = actual.map(mapping)

if actual.isnull().any():
    raise ValueError("Some values in the actual classification could not be mapped. Please check your NBER data.")

# Calculate the Brier Score for each model.
# Brier Score = average of (forecast probability - actual outcome)^2
brier_lr  = ((prob_results['LogisticRegression'] - actual) ** 2).mean()
brier_svc = ((prob_results['SVC'] - actual) ** 2).mean()

print("Brier Score for Logistic Regression:", brier_lr)
print("Brier Score for SVC:", brier_svc)

# The Combination model is defined as the average of the probabilities from both models.
prob_results['Comb'] = (prob_results['LogisticRegression'] + prob_results['SVC']) / 2.0

# Calculate the Brier Score for the Combination model.
brier_comb = ((prob_results['Comb'] - actual) ** 2).mean()

print("Brier Score for Combination model:", brier_comb)


# Compare the Brier Scores to determine the best model (lower = better).
brier_scores = {
    'LogisticRegression': brier_lr,
    'SVC': brier_svc,
    'Comb': brier_comb
}

best_model = min(brier_scores, key=brier_scores.get)
print("Best model based on Brier Score:", best_model)

# Extract the forecasts from the best model.
best_forecasts = prob_results[[best_model]]

best_forecasts.to_csv('RecessionIndicator.csv')

