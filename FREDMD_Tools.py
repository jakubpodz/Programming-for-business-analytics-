import pandas as pd
import numpy as np
import statsmodels.api as sma
import statsmodels.tsa.api as smt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def select_continuous(srs):
    """Given a Pandas Series, select continous, non-missing data,
     finishing at the end of the supplied series, and return"""
    if srs.isna().any():
        test_vals = srs.isna().cumsum()
        test_idx = test_vals[test_vals==test_vals[-1]].index
        return srs.loc[test_idx][1:]
    else:
        return srs

def fit_models(srs,h):
    """Fit three ETS Models:
    level (simple exponential smoothing)
    trend
    damped trend
    to data in srs.
    Inputs : srs - a pandas Series with appropriate date index
           : h forecast horizon
    Outputs : mdl, a code for the selected model
            : forecasts - h setep ahead forecasts
    """
    model_args = {'A': {}, 'AA': {'trend': 'add'}, 'AAd': {'trend': 'add', 'damped_trend': True}}
    fits = {i: '' for i in model_args.keys()}
    AICc = pd.Series(index=model_args.keys(), dtype=np.float64)
    for m in model_args.keys():
        md = smt.ETSModel(srs, **model_args[m])
        fits[m] = md.fit(disp=False)
        AICc[m] = fits[m].aicc
    mdl = AICc.idxmin()
    forecasts = fits[mdl].forecast(h)
    return mdl, forecasts

def pca_function(stdata,n_comps):
    """Returns n_comps principal component of a data set.
    input: stdata - a n x t pandas data frame
    output: n_comps principal components, standardised to s.d = 1 """
    factors = sma.PCA(stdata, n_comps).factors
    factors = (factors - factors.mean(0)) / factors.std(0)
    return factors

def fit_class_models(X,y):
    lr_model = LogisticRegression().fit(X=X, y=y)
    lr_predict = lr_model.predict_proba(X.iloc[[-1]])
    svc_model = SVC(probability=True).fit(X=X, y=y)
    svc_predict = svc_model.predict_proba(X.iloc[[-1]])
    return lr_predict,svc_predict,lr_model,svc_model
