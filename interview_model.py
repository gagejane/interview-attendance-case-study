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
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
#----Tree visualization--------------------------------
from sklearn.tree import export_graphviz
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
#from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer

from sklearn.tree import export_graphviz


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

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)

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
print("RFC Test Score:", rfc.score(X_test, y_test))
print("RFC precision score:", precision_score(y_test, y_predict))
print("RFC recall score:", recall_score(y_test, y_predict))


abc.fit(X_train, y_train)
y2_predict = abc.predict(X_test)
print("ABC Test Score:", abc.score(X_test, y_test))
print("ABC precision score:", precision_score(y_test, y2_predict))
print("ABC recall score:", recall_score(y_test, y2_predict))

gdbc.fit(X_train, y_train)
y3_predict = gdbc.predict(X_test)
print("Gradient Boosting Regressor Test Score:", gdbc.score(X_test, y_test))
print("Gradient Boosting Regressor precision score:", precision_score(y_test, y3_predict))
print("Gradient Boosting Regressor recall score:", recall_score(y_test, y_predict))

# tree = DecisionTreeClassifier()
# tree.fit(X_train, y_train)
# export_graphviz(tree, "tree.dot")
#
# !dot -Tpng tree.dot -o tree.png
#
# Image("tree.png")


# logr = linear_model.LogisticRegression()
# logr.fit(X_train, y_train)
# y4_predict = logr.predict(X_test)
# print("Logistic accuracy score:", logr.score(X_test, y_test))
# print("Logistic precision score:", precision_score(y_test, y4_predict))



# num_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# accuracy_results = []
# for num in num_features:
#     rf = RandomForestClassifier(max_features = num)
#     rf.fit(X_train, y_train)
#     y_predict = rf.predict(X_test)
#     accuracy = rf.score(X_test, y_test)
#     accuracy_results.append(accuracy)
#
# #print("Accuracy Results: ", accuracy_results)
#
# plt.figure(figsize=(10,5))
# plt.plot(num_features, accuracy_results, label='Accuracy')
# plt.xlabel('Number of Estimators')
# plt.ylabel('Accuracy')
#
#
# plt.title('Accuracy vs. Num Estimators')
# plt.legend()
# plt.show()
