import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
df = pd.read_csv('Fraud.csv')
df = df.rename(columns={'oldbalanceOrg' :'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest' : 'newBalanceDest'})
df.info()
df.head(200)
df.tail(300)
df.isnull().values.any()
print('fraudulent transaction are {}'.format(list(df.loc[df.isFraud == 1].type.drop_duplicates().values)))
dfFraudTransfer=df.loc[(df.isFraud == 1)&(df.type == 'TRANSFER')]
dfFraudCashout=df.loc[(df.isFraud == 1)&(df.type == 'CASH_OUT')]
print('\nnumber of fraudulent TRANSFERs={}'.format(len(dfFraudTransfer)))
print('\nnumber of fraudulent CASH_OUT={}'.format(len(dfFraudCashout)))
CountisFlaggedFraud = df.loc[(df.isFlaggedFraud == 1)]
CountisFlaggedFraudWithTransfer = df.loc[(df.isFlaggedFraud == 1) & (df.type == 'TRANSFER')]
print(len(CountisFlaggedFraud))
print(len(CountisFlaggedFraudWithTransfer))
print(df.shape)
print(CountisFlaggedFraudWithTransfer)
print('\nThe type of transactions in which isFlaggedFraud is set: \{}'.format(list(df.loc[df.isFlaggedFraud == 1].type.drop_duplicates())))
dfTransfer=df.loc[df.type == 'TRANSFER']
dfFlagged=df.loc[df.isFlaggedFraud == 1]
dfNotFlagged=df.loc[df.isFlaggedFraud == 0]
print('\nminimum amount transacted when isFlaggedFraud is set={}'.format(dfFlagged.amount.min()))
print('\nmax amount transacted when isFlaggedFraud is set={}'.format(dfFlagged.amount.max()))
print('\nmax amount is Transfered when isFlaggedFraud is NOT set={}'.format(dfNotFlagged.amount.max()))
print('\n number of Transfers where isFlaggedFraud = 0, yet oldBalanceDest = 0 and newBalanceDest = 0: {}'.format(len(dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & (dfTransfer.oldBalanceDest == 0) & (dfTransfer.newBalanceDest == 0)]))) 
print('\n Min, Max of OldBalanceOrig for isFlaggedFraud = 1 Transfers are : {}'.format([round(dfFlagged.oldBalanceOrig.min()), round(dfFlagged.oldBalanceOrig.max())]))
print('\n Min, Max of oldBalanceOrig of isFlaggedFraud = 0 Transfers where oldBalanceOrig = newBalanceOrig : {}'.format([dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0)  & (dfTransfer.oldBalanceOrig == dfTransfer.newBalanceOrig)].oldBalanceOrig.min(),round(dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & (dfTransfer.oldBalanceOrig == dfTransfer.newBalanceOrig)].oldBalanceOrig.max())]))
print('\nHave Originators of transactions flagged as fraud transacted more than once? {}'.format((dfFlagged.nameOrig.isin(pd.concat([dfNotFlagged.nameOrig,dfNotFlagged.nameDest]))).any()))
print('\nHave destinations for transactions flagged as fraud initiated other transactions? {}'.format((dfFlagged.nameDest.isin(dfNotFlagged.nameOrig)).any()))
print('\nHow many destination accounts of transactions as flagged as fraud have been destination accounts more than once?: {}'.format(sum(dfFlagged.nameDest.isin(dfNotFlagged.nameDest))))
print('\nAre there any merchants among originators accounts for CASH_IN transactions? {}'.format((df.loc[df.type == 'CASH_IN'].nameOrig.str.contains('M')).any()))
print('\nAre there any merchants among destination accounts for CASH_OUT transactions? {}'.format((df.loc[df.type == 'CASH_OUT'].nameDest.str.contains('M')).any()))
print('\nAre there are any merchants accounts among any originator accounts?\n{}'.format(df.nameOrig.str.contains('M').any()))
print('\nAre there any transactions having merchants among destination accounts other than the PAYMENT type?\n{}'.format((df.loc[df.nameDest.str.contains('M')].type != 'PAYMENT').any()))
print('\nWithin fraudlent transactions, are there destinations for TRANSFERs that are also originators for CASH_OUTs?\n{}'.format((dfFraudTransfer.nameDest.isin(dfFraudCashout.nameOrig)).any()))
dfNotFraud = df.loc[df.isFraud == 0]
print('\nFraudlent TRANSFERs whose destination accounts are originators of genuine CASH_OUTs:\n\n{}'.format(dfFraudTransfer.loc[dfFraudTransfer.nameDest.isin(dfNotFraud.loc[dfNotFraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())]))
X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
randomState = 5
np.random.seed(randomState)
Y = X['isFraud']
del X['isFraud']
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)
Xfraud = X.loc[Y == 1]
XnonFraud = X.loc[Y == 0]
print('\nThe fraction of fraudulent transactions with \'oldBalanceDest\' = \\'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.\
format(len(Xfraud.loc[(Xfraud.oldBalanceDest == 0) & \
(Xfraud.newBalanceDest == 0) & (Xfraud.amount)]) / (1.0 * len(Xfraud))))
print('\nThe fraction of genuine transactions with \'oldBalanceDest\' = \newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.\
format(len(XnonFraud.loc[(XnonFraud.oldBalanceDest == 0) & \
(XnonFraud.newBalanceDest == 0) & (XnonFraud.amount)]) / (1.0 * len(XnonFraud))))
X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), ['oldBalanceDest', 'newBalanceDest']] = - 1
X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
limit = len(X)

def plotStrip(x, y, hue, figsize = (14, 9)):
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y, 
             hue = hue, jitter = 0.4, marker = '.', 
             size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1), 
               loc=2, borderaxespad=0, fontsize = 16);
    return ax
import matplotlib.pyplot as plt
import seaborn as sns

def plotStrip(x, y, hue, figsize=(14, 9)):
    plt.figure(figsize=figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x=x, y=y, hue=hue, jitter=0.4, marker='.', size=4, palette=colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size=16)
        return ax
ax = plotStrip(X.step[:limit], Y[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size=16)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size=20)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plotStrip(x, y, hue, figsize=(14, 9)):
    plt.figure(figsize=figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x=x, y=y, hue=hue, jitter=0.4, marker='.', size=4, palette=colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size=16)
        return ax
limit = len(X)
ax = plotStrip(Y[:limit], X.amount[:limit], X['type'][:limit], figsize=(14, 9))
ax.set_ylabel('amount', size=16)
ax.set_title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size=18)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plotStrip(x, y, hue, figsize=(14, 9)):
    plt.figure(figsize=figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x=x, y=y, hue=hue, jitter=0.4, marker='.', size=4, palette=colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size=16)
        return ax
limit = len(X)
ax = plotStrip(Y[:limit], -X.errorBalanceDest[:limit], X['type'][:limit], figsize=(14, 9))
ax.set_ylabel('- errorBalanceDest', size=16)
ax.set_title('Opposite polarity fingerprints over the error in destination account balances', size=18)
plt.show()
#long computatuiton time (copied code)
x = 'errorBalanceDest'
y = 'step'
z = 'errorBalanceOrig'
zOffset = 0.02
limit = len(X)
sns.reset_orig() 
fig = plt.figure(figsize = (10, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.loc[Y == 0, x][:limit], X.loc[Y == 0, y][:limit], \
  -np.log10(X.loc[Y == 0, z][:limit] + zOffset), c = 'g', marker = '.', \
  s = 1, label = 'genuine')
    ax.scatter(X.loc[Y == 1, x][:limit], X.loc[Y == 1, y][:limit], \
  -np.log10(X.loc[Y == 1, z][:limit] + zOffset), c = 'r', marker = '.', \
  s = 1, label = 'fraudulent')
ax.set_xlabel(x, size = 16); 
ax.set_ylabel(y + ' [hour]', size = 16); 
ax.set_zlabel('- log$_{10}$ (' + z + ')', size = 16)
ax.set_title('Error-based features separate out genuine and fraudulent \
transactions', size = 20)
plt.axis('tight')
ax.grid(1)
noFraudMarker = mlines.Line2D([], [], linewidth = 0, color='g', marker='.',markersize = 10, label='genuine')
fraudMarker = mlines.Line2D([], [], linewidth = 0, color='r', marker='.',markersize = 10, label='fraudulent')
plt.legend(handles = [noFraudMarker, fraudMarker], \bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={'size': 16});
print('skew = {}'.format( len(Xfraud) / float(len(X)) ))
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2,random_state = randomState)
pip install xgboost
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
clf = XGBClassifier(max_depth=3, scale_pos_weight=weights, n_jobs=4)
probabilities = clf.fit(trainX, trainY).predict_proba(testX)
auprc = average_precision_score(testY, probabilities[:, 1])
print('AUPRC = {}'.format(auprc))
from xgboost import plot_importance
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)
colours = plt.cm.Set1(np.linspace(0, 1, 9))
plot_importance(clf, height=1, color=colours, grid=False,show_values=False, importance_type='cover', ax=ax)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
        ax.set_xlabel('importance score', size=16)
ax.set_ylabel('features', size=16)
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_title('Ordering of features by importance to the model learnt', size=20)

plt.show()
trainSizes, trainScores, crossValScores = learning_curve(\XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4), trainX,\trainY, scoring = 'average_precision')
import numpy as np
import matplotlib.pyplot as plt
trainScoresMean = np.mean(trainScores, axis=1)
trainScoresStd = np.std(trainScores, axis=1)
crossValScoresMean = np.mean(crossValScores, axis=1)
crossValScoresStd = np.std(crossValScores, axis=1)
colours = plt.cm.tab10(np.linspace(0, 1, 9))
fig, ax = plt.subplots(figsize=(14, 9))
ax.fill_between(trainSizes, trainScoresMean - trainScoresStd,
                trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
ax.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,
                crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
ax.plot(trainSizes, trainScoresMean, 'o-', label='train', color=colours[0])
ax.plot(trainSizes, crossValScoresMean, 'o-', label='cross-val', color=colours[1])
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.legend(['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15), loc=2, borderaxespad=0, fontsize=16)
ax.set_xlabel('Training Set Size', size=16)
ax.set_ylabel('AUPRC', size=16)
ax.set_title('Learning Curves: Model Performance vs. Training Set Size', size=20)
plt.show()
