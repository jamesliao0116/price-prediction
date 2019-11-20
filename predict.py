import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from talib import abstract
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

MODEL_TYPE = 'LF' #RF for random forest, LF for Logistic Regression

# 1. Dealing with the date format of the raw data
# output: weekday and next week date
raw = pd.read_csv('SPY.csv')
raw['Date'] = pd.to_datetime(raw['Date'])
raw['Weekday'] = raw['Date'].dt.dayofweek + 1
raw['Date'] = raw['Date'].dt.date
raw['Future_date'] = np.where((raw.Date + timedelta(days=7)).isin(raw.Date), 
                               raw.Date + timedelta(days=7), None)

# 2. Fetch detailed price data from Yahoo Finance and caculate relevant technical indicator fetures
# output: technical indicator features
stk = yf.Ticker('SPY')
data = stk.history(start = '1993-01-29', end = '2018-06-30')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.columns = ['open','high','low','close','volume'] # change to reconizable term for ta-lib
ta_list = ['MACD','RSI','MOM','STOCH']
for x in ta_list:
    output = eval('abstract.'+x+'(data)')
    output.name = x.lower() if type(output) == pd.core.series.Series else None
    data = pd.merge(data, pd.DataFrame(output), left_on = data.index, right_on = output.index)
    data = data.set_index('key_0')

# 3. Merge raw data and downloaded data based on date and remove incomplete (Monday disapper issue) or irrelevant data
# output: clean dataset 
data.reset_index(inplace=True)
data = pd.merge(data,raw, left_index=True, right_index=True, how='outer')
data = pd.merge(data, data, left_on='Future_date', right_on='Date', how='left')
data = data[['Date_x', 'Weekday_x', 'SPY_x', 'Future_date_x', 'SPY_y', 'macd_y', 'macdsignal_y', 'macdhist_y', 'rsi_y', 'mom_y', 'slowk_y', 'slowd_y']]
data.columns = ['Date', 'Weekday', 'Price', 'Future_date', 'Future_price', 'macdsignal', 'macd', 'macdhist', 'rsi', 'mom', 'slowk', 'slowd']
data['Label'] = np.where(data.Future_price >= data.Price, 1, 0)
data['IsMon'] = np.where(data.Weekday == 1, 1, None)
data = data.dropna()
data = data[['Date', 'macd', 'macdsignal', 'macdhist', 'rsi', 'mom', 'slowk', 'slowd', 'Label']]

# 4. Prepare training set (60%), validation set (27%), and testing set (13%)
# output: split datasets

# As required by the spec, testing set should be set after 2015/01/01
split_point1, split_point2 = int(len(data)*0.5), int(len(data)- len(data[data.Date > datetime.date(2015, 1, 1)]))+1
data = data.drop('Date', axis = 1)
train = data.iloc[:split_point1,:].copy()
validate = data.iloc[split_point1:split_point2,:].copy()
test = data.iloc[split_point2:,:].copy()

# seperate label from the datasets
train_X = train.drop('Label', axis = 1)
train_y = train.Label
validate_X = validate.drop('Label', axis = 1)
validate_y = validate.Label
test_X = test.drop('Label', axis = 1)
test_y = test.Label

# 5. Check model type and return corresponding hyperparameters
if MODEL_TYPE == 'RF':
    print('Model: Random Forest')
    hyperparameters = np.arange(5, 16)
elif MODEL_TYPE == 'LF':
    print('Mode: Logistic Regression')
    hyperparameters = np.arange(0.5, 1.5, 0.1)
else:
    print('Model type not correct')
    exit()

# 6. Train the model with different hyperparameters, save the performance and model in record
# output: record that contains peformance and the trainined models, decscently sorted by the performance
train_auc= []
validate_auc = []
records = []

for parameter in hyperparameters:
    if MODEL_TYPE == 'RF':
        temp_model = RandomForestClassifier(n_estimators = parameter)
    else:
        temp_model = LogisticRegression(C = parameter)
    temp_model.fit(train_X, train_y)
    train_predictions = temp_model.predict(train_X)
    validate_predictions = temp_model.predict(validate_X)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_predictions)
    auc_area = auc(false_positive_rate, true_positive_rate)
    train_auc.append(auc_area)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(validate_y, validate_predictions)
    auc_area = auc(false_positive_rate, true_positive_rate)
    validate_auc.append(auc_area)
    records.append((auc_area, parameter, temp_model))

records = sorted(records, key=lambda x: x[0], reverse=True)
print(f'Best hyperparamter: {records[0][1]}')

# 7. Display the performance of each models and the final perofmance 
plt.figure(figsize = (14,10))
plt.plot(hyperparameters, train_auc, 'b', label = 'Train AUC')
plt.plot(hyperparameters, validate_auc, 'r', label = 'Validate AUC')
plt.ylabel('AUC')
if MODEL_TYPE == 'RF':
    plt.xlabel('number of decision tree')
else:
    plt.xlabel('C')
plt.legend(loc='upper right')
plt.show()

model = records[0][2]
prediction = model.predict(test_X)
print('Confusion matrix:')
print(confusion_matrix(test_y, prediction))
print(f'Accuracy: {model.score(test_X, test_y)}')
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, prediction)
print(f'AUC: {auc(false_positive_rate, true_positive_rate)}')