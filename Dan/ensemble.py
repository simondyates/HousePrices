from HousePrices.Dan.LassoRidgeENet import linReg
from HousePrices.Dan.xgboost_ import xgboost

def ensemble_models(models, use_log = True, scale = True, use_dum = False):
    reg = ['None', 'lasso', 'ridge', 'enet']
    booster = ['gblinear', 'gbtree', 'dart']

    use_log = True
    scale = True
    use_dum = False

    for mdl in models:
        for r in reg:
            try:
                mdl(r, use_log, scale, use_dum)
            except:
                break
        for b in booster:
            try:
                mdl(b, use_log, scale, use_dum)
            except:
                break