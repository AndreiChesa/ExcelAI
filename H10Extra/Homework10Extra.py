import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

os.system('cls' if os.name == 'nt' else 'clear')

train = pd.read_csv("H10Extra/train.csv")

train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])


#print(train.describe()), print(40 * '-')

#print(train.isnull().sum()), print(40 * '-')

age_imputer = SimpleImputer(strategy='mean')
age_imputer.fit(train[['Age']])
train['Age'] = age_imputer.transform(train[['Age']])

embarked_imputer = SimpleImputer(strategy='most_frequent')
embarked_imputer.fit(train[['Embarked']])
train['Embarked'] = embarked_imputer.transform(train[['Embarked']]).ravel()

#print(train.isnull().sum()), print(40 * '-')

ohe_pclass = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)

ohe_pclass.fit(train[['Pclass']])
encoded_pclass = ohe_pclass.transform(train[['Pclass']])

encoded_train = pd.DataFrame(encoded_pclass, columns=ohe_pclass.get_feature_names_out(['Pclass'])) 
train = pd.concat([train,encoded_train],axis=1)
train = train.drop('Pclass',axis=1)

#print(train.head()), print(40 * '-')

ohe_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
ohe_sex.fit(train[['Sex']])
encoded_sex = ohe_sex.transform(train[['Sex']])

encoded_train = pd.DataFrame(encoded_sex, columns=ohe_sex.get_feature_names_out(['Sex']))
train  = pd.concat([train, encoded_train], axis=1)
train = train.drop('Sex', axis=1)

#print(train.head()), print(40 * '-')

ohe_embarked = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
ohe_embarked.fit(train[['Embarked']])
encoded_embarked = ohe_embarked.transform(train[['Embarked']])

encoded_train = pd.DataFrame(encoded_embarked, columns=ohe_embarked.get_feature_names_out(['Embarked']))
train  = pd.concat([train, encoded_train], axis=1)
train = train.drop('Embarked', axis=1)

#print(train.head()), print(40 * '-')

X = train.drop('Survived',axis=1)
y = train['Survived']

#print(X.shape, y.shape), print(40 * '-')

model = DecisionTreeClassifier(random_state=42)

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

#print(f"Mean CV Accuracy: {stratified_scores.mean():.4f}")
#print(f"Standard Deviation: {stratified_scores.std():.4f}")
#print("-" * 40)

param_grid = {
    'max_depth': [3, 5, 7, 11, 13, 17,None],
    'min_samples_split': [2, 3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
)

grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", f"{grid_search.best_score_:.4f}")
print("Best estimator:", grid_search.best_estimator_)
print("-" * 40)

true_model = grid_search.best_estimator_

test = pd.read_csv('H10Extra/test.csv')
passengers_ids = test['PassengerId']
test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

#print(test.head()), print(40 * '-')
#print(test.describe()), print(40 * '-')

fare_imputer = SimpleImputer(strategy='mean')
fare_imputer.fit(train[['Fare']])
test['Fare'] = fare_imputer.transform(test[['Fare']])

test['Age'] = age_imputer.transform(test[['Age']])

#print(test.describe()), print(40 * '-')

encoded_pclass = ohe_pclass.transform(test[['Pclass']])

encoded_train = pd.DataFrame(encoded_pclass, columns=ohe_pclass.get_feature_names_out(['Pclass']))
test = pd.concat([test,encoded_train], axis=1)
test = test.drop('Pclass', axis=1)

#print(test.head()), print(40 * '-')

encoded_sex = ohe_sex.transform(test[['Sex']])

encoded_train = pd.DataFrame(encoded_sex,columns=ohe_sex.get_feature_names_out(['Sex']))
test = pd.concat([test,encoded_train], axis=1)
test = test.drop('Sex', axis=1)

#print(test.head()), print(40 * '-')

encoded_embarked = ohe_embarked.transform(test[['Embarked']])

encoded_train = pd.DataFrame(encoded_embarked,columns=ohe_embarked.get_feature_names_out(['Embarked']))
test = pd.concat([test,encoded_train], axis=1)
test = test.drop('Embarked', axis=1)

print(test.head()), print(40 * '-')

test_predictions = true_model.predict(test)

output = pd.DataFrame({'PassengerId': passengers_ids, 'Survived': test_predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


