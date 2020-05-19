# Use the MeanValue encoding and run univariate linear regressions on all variables

import pandas as pd
import numpy as np
import numpy.linalg as l
import plotly.graph_objects as go

house = pd.read_csv('./train_meanEnc.csv', index_col='Id')

# Split target from attributes and normalise attribs
Y = house['SalePrice'].to_numpy().reshape(-1, 1)
X = house.drop('SalePrice', axis=1)
X = (X - X.mean(axis=0)) / X.std(axis=0)

results = pd.DataFrame(index=X.columns, columns=['Beta', 'Coef t', 'R2'])
MTE = l.norm(Y - Y.mean())**2 / (len(Y) - 1)

# Run univariate linear regression on each feature
for col in X.columns:
    x = X[[col]].copy()
    x.insert(0, 'ones', 1)
    x = x.to_numpy()
    xTx_inv = l.inv(x.T @ x)
    beta = xTx_inv @ x.T @ Y
    y_hat = x @ beta
    MSE = l.norm(Y - y_hat)**2 / (len(Y) - 1)
    xTx_diag = np.reshape(np.diag(xTx_inv), [-1, 1])
    se_beta = np.sqrt(xTx_diag *  MSE)
    t_stats = abs(beta / se_beta)
    results.loc[col] = [beta[1, 0], t_stats[1, 0], 1 - MSE/MTE]

results = results.sort_values('R2', ascending = False)
results.to_csv('./univariate_meanEcoding.csv')
top20 = results.iloc[0:20]
bottom20 = results.iloc[-1:-21:-1]

# Plot the results
fig1=go.Figure()
fig1.add_trace(go.Bar(x=top20.index, y=top20['R2'], name='R Squared'))
fig1.update_layout(
    title='R^2 of Top 20 Features',
    xaxis_title='',
    yaxis_title='R^2'
)
fig1.show()

fig2=go.Figure()
fig2.add_trace(go.Bar(x=bottom20.index, y=bottom20['R2'], name='R Squared'))
fig2.update_layout(
    title='R^2 of Bottom 20 Features',
    xaxis_title='',
    yaxis_title='R^2'
)
fig2.show()