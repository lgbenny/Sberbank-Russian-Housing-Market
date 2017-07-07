import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()

get_ipython().magic(u'matplotlib inline')

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)

train_df = pd.read_csv("F://train.csv", parse_dates=['timestamp'])
train_df.shape

#data visualization

for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

f, ax = plt.subplots(figsize=(10, 7))
plt.scatter(x=train_df['full_sq'], y=train_df['price_doc'], c='r')
ax.set(xlabel='full_sq', ylabel='price_doc')

plt.figure(figsize=(12,8))
sns.countplot(x="floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

f, ax = plt.subplots(figsize=(12, 8))
plt.xticks(rotation='90')
by_df = train_df.sort_values(by=['build_year'])
sns.countplot(x=by_df['build_year'])
ax.set(title='Distribution of build year')

internal_chars = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'kitch_sq', 'state', 'price_doc']
corrmat = train_df[internal_chars].corr()
f, ax = plt.subplots(figsize=(10, 7))
plt.xticks(rotation='90')
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)

# First Model

train = pd.read_csv('F://train.csv', parse_dates=['timestamp'])
test = pd.read_csv('F://test.csv', parse_dates=['timestamp'])
id_test = test.id

#clean data
bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[train.full_sq > 400].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq > 400].index
test.loc[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 300].index
test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.build_year > 2017].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year > 2017].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index
test.loc[bad_index, "num_room"] = np.NaN
bad_index = train[train.num_room >= 10].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[train.num_room >= 10].index
test.loc[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.loc[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.loc[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = np.NaN
test.state.value_counts()

# feature engineering
# Timestamp

#year and month
train["yearmonth"] = train["timestamp"].dt.year*100 + train["timestamp"].dt.month
test["yearmonth"] = test["timestamp"].dt.year*100 + test["timestamp"].dt.month

# year and week #
train["yearweek"] = train["timestamp"].dt.year*100 + train["timestamp"].dt.weekofyear
test["yearweek"] = test["timestamp"].dt.year*100 + test["timestamp"].dt.weekofyear

# year #
train["year"] = train["timestamp"].dt.year
test["year"] = test["timestamp"].dt.year

# month of year #
train["month_of_year"] = train["timestamp"].dt.month
test["month_of_year"] = test["timestamp"].dt.month

# week of year #
train["week_of_year"] = train["timestamp"].dt.weekofyear
test["week_of_year"] = test["timestamp"].dt.weekofyear

# day of week #
train["day_of_week"] = train["timestamp"].dt.weekday
test["day_of_week"] = test["timestamp"].dt.weekday

#Other feature engineering
train["ratio_life_sq_full_sq"] = train["life_sq"] / np.maximum(train["full_sq"].astype("float"),1)
test["ratio_life_sq_full_sq"] = test["life_sq"] / np.maximum(test["full_sq"].astype("float"),1)

train["ratio_kitch_sq_life_sq"] = train["kitch_sq"] / np.maximum(train["life_sq"].astype("float"),1)
test["ratio_kitch_sq_life_sq"] = test["kitch_sq"] / np.maximum(test["life_sq"].astype("float"),1)

train["ratio_kitch_sq_full_sq"] = train["kitch_sq"] / np.maximum(train["full_sq"].astype("float"),1)
test["ratio_kitch_sq_full_sq"] = test["kitch_sq"] / np.maximum(test["full_sq"].astype("float"),1)

train['room_size'] = train['life_sq'] / np.maximum(train['num_room'].astype("float"), 1)
test['room_size'] = test['life_sq'] / np.maximum(test['num_room'].astype("float"), 1)

train["ratio_floor_max_floor"] = train["floor"] / train["max_floor"].astype("float")
test["ratio_floor_max_floor"] = test["floor"] / test["max_floor"].astype("float")

train["floor_from_top"] = train["max_floor"] - train["floor"]
test["floor_from_top"] = test["max_floor"] - test["floor"]

train["age_of_building"] = train["year"] - train["build_year"]
test["age_of_building"] = test["year"] - test["build_year"]

mult = 1.054880504
train['price_doc'] = train['price_doc'] * mult
y_train = train["price_doc"]

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))

x_train = x_all[:num_train]
x_test = x_all[num_train:]

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 422
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_predict = model.predict(dtest)
output1 = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


# Second Model

train = pd.read_csv('F://train.csv')
test = pd.read_csv('F://test.csv')
id_test = test.id 


mult = .969

y_train = train["price_doc"] * mult + 10
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 385  # This was the CV output, as earlier version shows
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output2 = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

# Combine Model

result = output1.merge(output2, on="id", suffixes=['_1','_2'])
result["price_doc"] = np.exp( .22*np.log(result.price_doc_1) +
                                    .78*np.log(result.price_doc_2) ) 
result["price_doc"] =result["price_doc"] *0.9910     
result.drop(["price_doc_1","price_doc_2"],axis=1,inplace=True)

result.to_csv("result.csv", index=False)



