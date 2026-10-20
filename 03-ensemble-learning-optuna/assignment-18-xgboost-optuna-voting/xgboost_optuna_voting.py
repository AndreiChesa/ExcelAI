import pandas as pd
import lightgbm as lgb
import optuna # type: ignore
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("H18/train.csv")
test_data = pd.read_csv("H18/test.csv")

train_target = train_data["FloodProbability"]
train_data = train_data.drop(columns=["FloodProbability"])
train_data = train_data.drop(columns=["id"])
train_data = train_data.drop(columns=["PoliticalFactors"])

x_train, x_val, y_train, y_val = train_test_split(train_data, train_target, test_size=0.2, random_state=42)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
    num_leaves = trial.suggest_int("num_leaves", 20, 50)

    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=learning_rate, num_leaves=num_leaves, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    return r2_score(y_val, predictions)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best parameters for LightGBM:", study.best_params)

def objective_xgb(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
    max_depth = trial.suggest_int("max_depth", 3, 10)

    model = XGBRegressor(n_estimators=2000, learning_rate=learning_rate, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    return r2_score(y_val, predictions)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)

print("Best parameters for XGBoost:", study_xgb.best_params)

model1 = lgb.LGBMRegressor(n_estimators=2000, **study.best_params, random_state=42, n_jobs=-1)
model1.fit(train_data, train_target)
model2 = XGBRegressor(n_estimators=2000, **study_xgb.best_params, random_state=42, n_jobs=-1)
model2.fit(train_data, train_target)

model3 = VotingRegressor(estimators=[("lgb", model1), ("xgb", model2)])
model3.fit(train_data, train_target)
predictions = model3.predict(test_data.drop(columns=["id", "PoliticalFactors"]))

submission = pd.DataFrame({"id": test_data["id"], "FloodProbability": predictions})
submission.to_csv("H18/submission.csv", index=False)

print("Done")