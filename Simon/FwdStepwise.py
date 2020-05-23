# Runs a forward stepwise multilinear regression on the entire dataset
# Uses mean value encoding and targets log sale price

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import the data
house = pd.read_csv('./train_meanEnc.csv', index_col='Id')
house['priceSqFt'] = house['SalePrice'] / house['GrLivArea']

# create a Box-Cox transformed y
y = house['SalePrice'].to_numpy().reshape(-1, 1)
y = np.log(y)

# I also ran a regression on price / sq ft using the following:
#y = house['priceSqFt'].to_numpy().reshape(-1, 1)

# drop target columns
house = house.drop(['SalePrice', 'logSalePrice', 'priceSqFt'], axis=1)
cols = house.columns

# split into test / train
Xtrain, Xtest, ytrain, ytest = train_test_split(house, y, test_size=0.2, random_state=1)

# scale the data
ss = StandardScaler(with_mean=True, with_std=True)  # set these to False to turn off scaling
ss.fit(Xtrain)
Xs_train = ss.transform(Xtrain)
Xs_test = ss.transform(Xtest)

def t_stat(reg, X, y):
        sse = np.sum((reg.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        try:
            xTx_inv = np.linalg.inv(X.T @ X)
            if min(np.diag(xTx_inv)) <= 0:
                t = np.full([1, X.shape[1]], np.nan)
            else:
                se = np.sqrt(np.diagonal(sse * xTx_inv))
                t = reg.coef_ / se
        except:
            t = np.full([1, X.shape[1]], np.nan)
        return t[0]

lin = LinearRegression()
kf = KFold(n_splits=5, shuffle=False)

# initialise global result variables
col_picks = []
ts = []
ISR2s = []
OSR2s = []
coefs = []
search = [i for i in range(len(cols)) if i not in col_picks]
while len(search) > 45:
    print(len(search))
    # initialise result variables for this column set
    col_t = []
    col_ISR2 = []
    col_OSR2 = []
    for col in search:
        X = Xs_train[:, col_picks + [col]]
        split_t = []
        split_ISR2 = []
        split_OSR2 = []
        for train_index, test_index in kf.split(Xs_train):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = ytrain[train_index], ytrain[test_index]
            lin.fit(X_train, y_train)
            #Calc in-sample stats
            split_ISR2.append(lin.score(X_train, y_train))
            split_t.append(t_stat(lin, X_train, y_train)[len(col_picks)])
            # Calc OOS-sample stat
            split_OSR2.append(lin.score(X_test, y_test))
        col_t.append(sum(split_t) / len(split_t))
        col_ISR2.append(sum(split_ISR2) / len(split_ISR2))
        col_OSR2.append(sum(split_OSR2) / len(split_OSR2))
    idx = col_OSR2.index(max(col_OSR2))
    col_picks.append(search[idx])
    ts.append(col_t[idx])
    ISR2s.append(col_ISR2[idx])
    # Use the fully OOS score for the current model
    lin.fit(Xs_train[:, col_picks], ytrain)
    OSR2s.append(lin.score(Xs_test[:, col_picks], ytest))
    coefs.append(lin.coef_[0, len(col_picks)-1])
    search = [i for i in range(len(cols)) if i not in col_picks]

results = pd.DataFrame(index=range(len(col_picks)), columns=['col', 'ISR2', 'OSR2', 'coefs', 't_stats'])
results['col'] = cols[col_picks]
results['ISR2'] = ISR2s
results['OSR2'] = OSR2s
results['coefs'] = coefs
results['t_stats'] = ts

filename = 'FwdStepTop30.csv'
#results.to_csv(filename, index=False)  #commented out so as not to keep resaving
#print(f'Results saved as {filename}')
print('Score on completely OOS: {0:.2%}'.format(lin.score(Xs_test[:, col_picks], ytest)))

# Make a beautiful graph of R^2 vs complexity
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=results['col'], y=results['ISR2'], name='In-sample R<sup>2</sup>'))
fig.add_trace(go.Scatter(x=results['col'], y=results['OSR2'], name='Out-sample R<sup>2</sup>'))
fig.add_trace(go.Bar(x=results['col'], y=abs(results['t_stats']), name='t stats', opacity=.6), secondary_y=True)
fig.update_layout(
    title='R<sup>2</sup> and t stats for Forward Multilinear Regression of log(SalePrice)',
    xaxis_tickangle=45,
    xaxis_title='',
    legend = dict(x = .81, y = .85),
    shapes=[
        dict(
            type='line',
            yref='y2', y0=2, y1=2,
            xref='paper', x0=0, x1=.94,
            line=dict(
                color="red",
                width=2,
                dash="dot",
            )
        )],
    annotations=[
        dict(
            x=results.iloc[28, 0],
            y= 2.8,
            xref="x",
            yref="y2",
            text="|t|=2",
            font=dict(
                color="red",
                size=14
            ),
            showarrow=False
            )]
)
fig.update_yaxes(title='', tickformat='.0%', secondary_y=False)
fig.update_yaxes(title='|t stat|', tickformat='.0f', showgrid = False, secondary_y=True)
fig.show()