import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.simplefilter('ignore')

titanic_train_data = pd.read_csv('E:\practice\AIApplications\PandasPractice\dataset\Titanic\\train.csv')
titanic_test_data = pd.read_csv('E:\practice\AIApplications\PandasPractice\dataset\Titanic\\test.csv')

#print(titanic_train_data.head())
#print(titanic_train_data)
print(titanic_test_data.head())
#print(titanic_test_data)

from sklearn.ensemble import RandomForestClassifier

y = titanic_train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(titanic_train_data[features])
X_test = pd.get_dummies(titanic_test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)


#print(titanic_test_data.columns.tolist())

output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")