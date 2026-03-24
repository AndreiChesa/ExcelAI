import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("H19/train.csv")
test_data = pd.read_csv("H19/test.csv")

target_col = "NObeyesdad"
feature_drop_cols = [target_col, "id"]

x_full = train_data.drop(columns=[col for col in feature_drop_cols if col in train_data.columns])
x_test = test_data.drop(columns=["id"]) if "id" in test_data.columns else test_data.copy()

# One-hot encode all categorical columns using train+test to keep columns aligned.
combined = pd.concat([x_full, x_test], axis=0, ignore_index=True)
combined_encoded = pd.get_dummies(combined, drop_first=False)
x_full_encoded = combined_encoded.iloc[: len(x_full)].copy()
x_test_encoded = combined_encoded.iloc[len(x_full) :].copy()

label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(train_data[target_col])

x_train, x_val, y_train, y_val = train_test_split(
    x_full_encoded,
    y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full,
)

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(n_estimators=1000, **params, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model.score(x_val, y_val)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

best_params = study.best_params
print("Best parameters for LightGBM:", best_params)

model1 = lgb.LGBMClassifier(
    n_estimators=1000,
    **best_params,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
model1.fit(x_train, y_train)

def objective2(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }
    model = xgb.XGBClassifier(
        n_estimators=1000,
        **params,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train)
    return model.score(x_val, y_val)

study2 = optuna.create_study(direction="maximize")
study2.optimize(objective2, n_trials=25)

best_params2 = study2.best_params
print("Best parameters for XGBoost:", best_params2)

model2 = xgb.XGBClassifier(
    n_estimators=1000,
    **best_params2,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    eval_metric="mlogloss",
)
model2.fit(x_train, y_train)

model3 = VotingClassifier(estimators=[("lgb", model1), ("xgb", model2)], voting="soft")
model3.fit(x_train, y_train)

predictions_encoded = model3.predict(x_test_encoded)
predictions = label_encoder.inverse_transform(predictions_encoded)

submission_ids = test_data["id"] if "id" in test_data.columns else pd.Series(range(len(test_data)))
submission = pd.DataFrame({"id": submission_ids, "NObeyesdad": predictions})
submission.to_csv("H19/sample_submission.csv", index=False)


