import pandas as pd
from numpy.f2py.crackfortran import previous_context
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)

data = pd.read_csv("dataset\\Student_Performance.csv")

#print(data.head()), print(40 * '-')

hours_studied_imputer = SimpleImputer(strategy='mean')
data['Hours Studied'] = hours_studied_imputer.fit_transform(data[['Hours Studied']])

previous_scores_imputer = SimpleImputer(strategy="constant", fill_value=0)
data['Previous Scores'] = previous_scores_imputer.fit_transform(data[['Previous Scores']])

sleep_hours_imputer = SimpleImputer(strategy='most_frequent')
data['Sleep Hours'] = sleep_hours_imputer.fit_transform(data[['Sleep Hours']])

sample_questions_imputer = SimpleImputer(strategy='mean')
data['Sample Question Papers Practiced'] = sample_questions_imputer.fit_transform(data[['Sample Question Papers Practiced']])

performace_imputer = SimpleImputer(strategy='mean')
data['Performance Index'] = performace_imputer.fit_transform(data[['Performance Index']])

ohe_extracurricular = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
ohe_extracurricular.fit(data[['Extracurricular Activities']])
encoded_extracurricular = ohe_extracurricular.transform(data[['Extracurricular Activities']])

data = data.drop('Extracurricular Activities', axis=1)
data = pd.concat([data, pd.DataFrame(encoded_extracurricular, columns=ohe_extracurricular.get_feature_names_out(['Extracurricular Activities']))], axis=1)

target = data['Performance Index']
data = data.drop('Performance Index', axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
#print(data.head()), print(40 * '-')

linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / len(y_true)

mse_linear_reg = mean_squared_error(y_test, linear_regression.predict(x_test))

print(f"MSE: {mse_linear_reg}")

dec_tree_reg = DecisionTreeRegressor(random_state=42)
dec_tree_reg.fit(x_train, y_train)
mse_dec_tree_reg = mean_squared_error(y_test, dec_tree_reg.predict(x_test))

print(f"MSE: {mse_dec_tree_reg}")