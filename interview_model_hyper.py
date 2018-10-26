import pandas as pd
import numpy as np

#----Model selection--------------------------------
from sklearn.model_selection import train_test_split
from sklearn import linear_model
#----------------------------------------------------

#----Models--------------------------------
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
#----Tree visualization--------------------------------
from sklearn.tree import export_graphviz
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
#from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

# def show_tree(tree):
#     # export to graph description language
#     export_graphviz(tree, "tree.dot")
#
#     # Execute bash command to convert from .dot format to .png format
#     !dot -Tpng tree.dot -o tree.png
#
#     # iPython shows the image!
#     return Image("tree.png")

# df = pd.read_csv('data/millie_cols.csv')
# pd.options.display.max_columns = 200
#
# df2 = pd.read_csv('data/emily_cols.csv')
# df3 = pd.read_csv('data/jane_cols.csv')
# df4 = pd.read_csv('data/ryan.csv')
#
# result = pd.concat([df, df2, df3, df4], axis=1, join_axes=[df.index])

result = pd.read_csv('data/clean_data.csv')
result.drop('Unnamed: 0', axis = 1, inplace = True)

result.to_csv('data/clean_data.csv')

result.dropna(inplace = True)
y = result.pop('observed_attendance')
X = result.values

n_estimators_lst = [10, 30, 50, 100, 150, 200, 500]
max_depth_lst = [1, 3, 5, 10, 15, 20, 40]

X_train, X_test, y_train, y_test = train_test_split(X, y)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)

zipped = (zip(result.columns, rfc.feature_importances_))
importance = sorted(zipped, key = lambda t: t[1])
#importance = [item for item in zipped]
#print (importance)

gdbc = GradientBoostingClassifier(learning_rate=0.1,
                                  n_estimators=100,
                                  random_state=1)

abc = AdaBoostClassifier(DecisionTreeClassifier(),
                         learning_rate=0.1,
                         n_estimators=100,
                         random_state=1)


rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True)
rf_oob.fit(X_train, y_train)
print("RFC OOB Score: ", rf_oob.oob_score_)

print("RFC Score:", rfc.score(X_test, y_test))
abc.fit(X_train, y_train)
y_predict = abc.predict(X_test)
print("ABC Test Score:", abc.score(X_test, y_test))

gdbc.fit(X_train, y_train)
y_predict = gdbc.predict(X_test)
print("GDBC Test Score:", gdbc.score(X_test, y_test))

#other

n_estimators_lst = [1, 3, 5, 10, 50, 100, 200]
max_depth_lst = [1, 3, 5, 10, 15, 20, 40]

#gridsearchcv
# rand_f = RandomForestClassifier()
# params = {"max_depth": [3, None],
#               "max_features": [1, 3, 10, 20, 30],
#               "min_samples_split": [1, 3, 10, 20, 30],
#               "min_samples_leaf": [1, 3, 10, 20, 30],
#               # "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}
# grid_s = GridSearchCV(rand_f, params, cv=5, verbose=0)
# best_model = grid_s.fit(X, y)

#mse vs number of estimators

train_errors_rf = []
test_errors_rf = []

for num_est in n_estimators_lst:
    rf = RandomForestRegressor(n_estimators = num_est, n_jobs=2)
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)
    y_pred_train = rf.predict(X_train)

    train_errors_rf.append(mean_squared_error(y_pred_train, y_train))
    test_errors_rf.append(mean_squared_error(y_pred_test, y_test))

plt.figure(figsize=(10,5))
plt.plot(n_estimators_lst, train_errors_rf, label='Training MSE')
plt.plot(n_estimators_lst, test_errors_rf, label='Test MSE')
plt.xlabel('Number of Estimators')
plt.ylabel('MSE')
plt.xscale('log')
plt.title('Random Forest MSE vs. Num Estimators')
plt.legend()
plt.show()
