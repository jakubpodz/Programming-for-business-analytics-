import numpy as np
import pandas as pd
from FREDMD_Tools import *

backtest_dates = pd.date_range(start='2005-1-1',end='2021-3-1',freq='M').to_period('M')
horizons=[3,6,9,12]


def load_data(csv_file):
    """
    Load FRED-MD data from a CSV file, 
    remove the transformation row
    and format the index as a monthly Period Index.
    """

    fred_data = pd.read_csv(csv_file, skiprows=[0])

    fred_data.set_index(fred_data.columns[0], inplace=True)

    fred_data.index = pd.to_datetime(fred_data.index).to_period('M')

    return fred_data


FRED_DATA = load_data('2023-09.csv')


models = {}  
forecasts = {h: {} for h in horizons}  
naive_forecasts = {h: {} for h in horizons} 
forecast_errors = {h: {} for h in horizons}  
naive_errors = {h: {} for h in horizons}  
r2_scores = {}  


for date in backtest_dates:
    models[date] = {}  
    for h in horizons:
        forecasts[h][date] = {}  
        naive_forecasts[h][date] = {}  
        forecast_errors[h][date] = {}  
        naive_errors[h][date] = {}  

    for srs_name in FRED_DATA.columns:
        series_data = FRED_DATA[srs_name]

        # Select training data before the backtest date
        training_data = series_data.loc[:date]
        training_data = select_continuous(training_data) 

        # Fit ETS models and generate forecasts for h = 1 to 12 months
        mdl, forecast_values = fit_models(training_data, 12)

        models[date][srs_name] = mdl  

        # Store forecasts for required horizons (3, 6, 9, 12 months)
        for h in horizons:
            forecasts[h][date][srs_name] = forecast_values[h-1]  # Adjust index (0-based)

        # Compute naive forecast 
        naive_value = training_data.iloc[-1]
        for h in horizons:
            naive_forecasts[h][date][srs_name] = naive_value

        # Compute forecast errors 
        for h in horizons:
            target_date = date + h  
            if target_date in series_data.index:
                actual_value = series_data.loc[target_date]

                # Forecast error for ETS models
                if srs_name in forecasts[h][date]:  
                    forecast_errors[h][date][srs_name] = actual_value - forecasts[h][date][srs_name]

                # Forecast error for naïve model
                if srs_name in naive_forecasts[h][date]:  
                    naive_errors[h][date][srs_name] = actual_value - naive_forecasts[h][date][srs_name]

forecast_error_dfs = {h: pd.DataFrame.from_dict(forecast_errors[h], orient="index") for h in horizons}
naive_error_dfs = {h: pd.DataFrame.from_dict(naive_errors[h], orient="index") for h in horizons}

# Compute Out-of-Sample R² for each horizon
for h in horizons:
    numerator = forecast_error_dfs[h].pow(2).sum(axis=0)  # ∑ (y_t - ŷ_t)^2
    denominator = naive_error_dfs[h].pow(2).sum(axis=0)  # ∑ (y_t - ỹ_t)^2
    r2_scores[h] = 1 - (numerator / denominator)  # Compute R²

# Convert R² scores to DataFrames
r2_dfs = {h: pd.DataFrame(r2_scores[h], columns=[f'R2_{h}M']) for h in horizons}

# Store latest selected models and 3-month R² in a DataFrame
final_results = pd.DataFrame({
    'model': {srs: models[backtest_dates[-1]][srs] for srs in FRED_DATA.columns if srs in models[backtest_dates[-1]]},
    'R2': r2_dfs[3].iloc[:, 0]  # Select 3-month R² values
})

final_results.to_csv('fcast_res.csv')
