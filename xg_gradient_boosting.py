from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.datasets import make_classification
from xgboost import XGBClassifier , XGBRegressor


X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=42, test_size=0.20)

xg_model = XGBClassifier(
    n_estimators=300,          # number of trees
    max_depth=4,               # tree depth
    learning_rate=0.1,         # eta
    subsample=0.8,             # row sampling
    colsample_bytree=0.8,      # column sampling
    reg_alpha=0.0,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    eval_metric="logloss",     # or "error", "auc", etc.
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)

xg_model.fit(X_train, y_train)

y_pred_1 = xg_model.predict(X_train)
y_pred = xg_model.predict(X_test)
ac = accuracy_score(y_train, y_pred_1)
ac1 = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(ac)
print(ac1)
print(cm)
print(cr)

xg_model_re = XGBRegressor(
    n_estimators=300,          # number of trees
    max_depth=4,               # tree depth
    learning_rate=0.1,         # eta
    subsample=0.8,             # row sampling
    colsample_bytree=0.8,      # column sampling
    reg_alpha=0.0,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    eval_metric="logloss",     # or "error", "auc", etc.
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)

xg_model_re.fit(X_train, y_train)

y_pred_re = xg_model_re.predict(X_test)

mse = mean_squared_error(y_test, y_pred_re)
r_score = r2_score(y_test, y_pred_re)
rmse = root_mean_squared_error(y_test, y_pred_re)
print(mse)
print(r_score)
print(rmse)






