import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Task 1
data = pd.read_csv("cc_approvals.data", header=None)
cc_apps = pd.DataFrame(data)
print(cc_apps.head(5))
print("\n")

# Task 2
# cc_apps description
cc_apps_description = cc_apps.describe()
print("cc_apps description")
print(cc_apps_description)
print("\n")

# cc_apps info
cc_apps_info = cc_apps.info()
print("cc_apps info")
print(cc_apps_info)
print("\n")

print(cc_apps.tail(17))
print("\n")

# Task 3
print(cc_apps.tail(17))
print("\n")
cc_apps = cc_apps.replace(["?"], np.NaN)
print(cc_apps.tail(17))
print("\n")

# Task 4
cc_apps.fillna(cc_apps.mean(), inplace=True)
# Checks our dataset array for missing values, and returns true or false. In this case we print the sum
print(pd.isna(cc_apps).sum())
print("\n")

# Task 5
for col in cc_apps:
    if cc_apps[col].dtype == "object":
        # Not sure why this works?
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])
print(cc_apps.isna().sum())
print("\n")

# Task 6
le = LabelEncoder()
for col in cc_apps:
    if cc_apps[col].dtype == 'object':
        cc_apps[col] = le.fit_transform(cc_apps[col])

# Task 7
cc_apps = cc_apps.drop([cc_apps.columns[11], cc_apps.columns[13]], axis=1)
cc_apps = cc_apps.values
X, y = cc_apps[:, 0:13], cc_apps[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# kept getting a error, but I think I found the problem
# In the code above I wrote X_train, y_train, X_test, y_test, but that is wrong and splits the data wrong
# Changed it to X_train, X_test, y_train, y_test, and now I don't get any errors

# Task 8
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# Task 9
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)

# Task 10
y_pred = logreg.predict(rescaledX_test)
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))
confusion_matrix(y_test, y_pred)

# Task 11
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol=tol, max_iter=max_iter)

# Task 12
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
