import time
from HousePrices.Dan.LassoRidgeENet import linReg
from HousePrices.Dan.xgboost_ import xgboost


def ensemble_models(models, use_log=True, scale=True, use_dum=False):
    reg = ['None', 'lasso', 'ridge', 'enet']
    booster = ['gblinear', 'gbtree', 'dart']

    optimal_models = []

    for mdl in models:
        for r in reg:
            print(r)
            try:
                start_t = time.time()
                fitted_model = mdl(r, use_log, scale, use_dum)
                end_t = time.time()
                print(f'time : {end_t - start_t}')
                print(f'best parameters {fitted_model.best_params_}')

                optimal_models.append(fitted_model.best_estimator_)
            except:
                break
        for b in booster:
            print(b)
            try:
                start_t = time.time()
                fitted_model = mdl(b, use_log, scale, use_dum)
                end_t = time.time()
                print(f'time : {end_t - start_t}')
                print(f'best parameters {fitted_model.best_params_}')

                optimal_models.append(fitted_model.best_estimator_)
            except:
                break


use_log = True
scale = True
use_dum = False

models = [linReg, xgboost]
ensemble_models(models, use_log, scale, use_dum)
