# Auto-ML
NO code/ Low code
Ingest
import numpy
from numpy import arange
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
from pandas import read_csv
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
boston_housing = "https://raw.githubusercontent.com/noahgift/boston_housing_pickle/master/housing.csv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = read_csv(boston_housing, delim_whitespace=True, names=names)
df.head()
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296.0	15.3	396.90	4.98	24.0
1	0.02731	0.0	7.07	0	0.469	6.421	78.9	4.9671	2	242.0	17.8	396.90	9.14	21.6
2	0.02729	0.0	7.07	0	0.469	7.185	61.1	4.9671	2	242.0	17.8	392.83	4.03	34.7
3	0.03237	0.0	2.18	0	0.458	6.998	45.8	6.0622	3	222.0	18.7	394.63	2.94	33.4
4	0.06905	0.0	2.18	0	0.458	7.147	54.2	6.0622	3	222.0	18.7	396.90	5.33	36.2
EDA
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
RM - average number of rooms per dwelling
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
Bk is the proportion of blacks by town

LSTAT - % lower status of the population

MEDV - Median value of owner-occupied homes in $1000's
prices = df['MEDV']
df = df.drop(['CRIM','ZN','INDUS','NOX','AGE','DIS','RAD'], axis = 1)
features = df.drop('MEDV', axis = 1)
df.head()
CHAS	RM	TAX	PTRATIO	B	LSTAT	MEDV
0	0	6.575	296.0	15.3	396.90	4.98	24.0
1	0	6.421	242.0	17.8	396.90	9.14	21.6
2	0	7.185	242.0	17.8	392.83	4.03	34.7
3	0	6.998	222.0	18.7	394.63	2.94	33.4
4	0	7.147	222.0	18.7	396.90	5.33	36.2
Modeling
Split Data
# Split-out validation dataset
array = df.values
X = array[:,0:6]
Y = array[:,6]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
for sample in list(X_validation)[0:2]:
    print(f"X_validation {sample}")
X_validation [  1.      6.395 666.     20.2   391.34   13.27 ]
X_validation [  0.      5.895 224.     20.2   394.81   10.56 ]
## Tune
# Test options and evaluation metric using Root Mean Square error method
num_folds = 10
seed = 7
RMS = 'neg_mean_squared_error'
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
Best: -11.793750 using {'n_estimators': 200}
-12.438375 (6.314128) with: {'n_estimators': 50}
-12.063336 (6.409448) with: {'n_estimators': 100}
-11.804976 (6.598562) with: {'n_estimators': 150}
-11.793750 (6.529366) with: {'n_estimators': 200}
-11.844873 (6.485019) with: {'n_estimators': 250}
-11.860770 (6.461158) with: {'n_estimators': 300}
-11.973209 (6.441418) with: {'n_estimators': 350}
-12.018959 (6.427801) with: {'n_estimators': 400}
Fit Model
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print("Mean Squared Error: \n")
print(mean_squared_error(Y_validation, predictions))
Mean Squared Error: 

26.086121710797425
Evaluate
predictions=predictions.astype(int)
evaluate = pd.DataFrame({
        "Org House Price": Y_validation,
        "Pred House Price": predictions
    })
evaluate["difference"] = evaluate["Org House Price"]-evaluate["Pred House Price"]
evaluate.head()
Org House Price	Pred House Price	difference
0	21.7	21	0.7
1	18.5	19	-0.5
2	22.2	20	2.2
3	20.4	19	1.4
4	8.8	9	-0.2
evaluate.describe()
Org House Price	Pred House Price	difference
count	102.000000	102.000000	102.000000
mean	22.573529	22.117647	0.455882
std	9.033622	8.758921	5.154438
min	6.300000	8.000000	-34.100000
25%	17.350000	17.000000	-0.800000
50%	21.800000	20.500000	0.600000
75%	24.800000	25.000000	2.200000
max	50.000000	56.000000	22.000000
Adhoc Predict
actual_sample = df.head(1)
actual_sample
CHAS	RM	TAX	PTRATIO	B	LSTAT	MEDV
0	0	6.575	296.0	15.3	396.9	4.98	24.0
adhoc_predict = actual_sample[["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]]
adhoc_predict.head()
CHAS	RM	TAX	PTRATIO	B	LSTAT
0	0	6.575	296.0	15.3	396.9	4.98
json_payload = adhoc_predict.to_json()
json_payload
'{"CHAS":{"0":0},"RM":{"0":6.575},"TAX":{"0":296.0},"PTRATIO":{"0":15.3},"B":{"0":396.9},"LSTAT":{"0":4.98}}'
scaler = StandardScaler().fit(adhoc_predict)
scaled_adhoc_predict = scaler.transform(adhoc_predict)
scaled_adhoc_predict
array([[0., 0., 0., 0., 0., 0.]])
list(model.predict(scaled_adhoc_predict))
[20.353731771344123]
Pickel the model
from sklearn.externals import joblib
joblib.dump(model, 'boston_housing_prediction.joblib')
['boston_housing_prediction.joblib']
Unpickel and predict
clf = joblib.load('boston_housing_prediction.joblib')
actual_sample2 = df.head(5)
actual_sample2
CHAS	RM	TAX	PTRATIO	B	LSTAT	MEDV
0	0	6.575	296.0	15.3	396.90	4.98	24.0
1	0	6.421	242.0	17.8	396.90	9.14	21.6
2	0	7.185	242.0	17.8	392.83	4.03	34.7
3	0	6.998	222.0	18.7	394.63	2.94	33.4
4	0	7.147	222.0	18.7	396.90	5.33	36.2
adhoc_predict2 = actual_sample[["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]]
adhoc_predict2.head()
CHAS	RM	TAX	PTRATIO	B	LSTAT
0	0	6.575	296.0	15.3	396.9	4.98
scale input
scaler = StandardScaler().fit(adhoc_predict2)
scaled_adhoc_predict2 = scaler.transform(adhoc_predict2)
scaled_adhoc_predict2
array([[0., 0., 0., 0., 0., 0.]])
# Use pickle loaded model
list(clf.predict(scaled_adhoc_predict2))
[20.353731771344123]
