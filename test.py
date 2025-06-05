import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Load and clean data
data = pd.read_csv('air_quality.csv', sep=';', decimal=',', header=0)
data = data.drop(columns=['Unnamed: 15', 'Unnamed: 16']).dropna().replace(-200, np.nan)
data = data.drop(columns=['Date', 'Time', 'NMHC(GT)'])  # Drop column with too many missing values
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

# Prepare features and target variable
# X = data.drop(columns=['CO(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'RH', 'AH'])
X = data.drop(columns=['CO(GT)'])

y = data['CO(GT)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    return mse_train, r2_train, mse_test, r2_test

from sklearn.tree import DecisionTreeRegressor

tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train_scaled, y_train)
y_pred_tree = tree_regressor.predict(X_test_scaled)

mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Decision Tree - MSE: {mse_tree}, R2: {r2_tree}")

models = {
    "Linear Regression": (LinearRegression(), {})
    # "Decision Tree": (DecisionTreeRegressor(), {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20], 'min_samples_leaf': [1, 5, 10], 'max_features': ['auto', 'sqrt', 'log2'], 'criterion': ['squared_error', 'friedman_mse']}),
    # "Ridge": (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    # "Lasso": (Lasso(), {'alpha': [0.01, 0.1, 1.0]}),
    # "Random Forest": (RandomForestRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}),
    # "Gradient Boosting": (GradientBoostingRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01]}),
    # "Multi-Layer Perceptron": (MLPRegressor(), {'hidden_layer_sizes': [(40, 40, 40)], 'activation': ['relu'], 'alpha': [0.01], 'learning_rate': ['constant'], 'solver': ['adam']}),
    # "K-Nearest Neighbors": (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']})
}

results = {}
for model_name, (model, param_grid) in models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    mse_train, r2_train, mse_test, r2_test = evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    overfitting_score = mse_train - mse_test  # Positive value indicates overfitting
    
    results[model_name] = {
        'mse_train': mse_train, 
        'r2_train': r2_train,
        'mse_test': mse_test,
        'r2_test': r2_test,
        'overfitting_score': overfitting_score,
        'best_params': grid_search.best_params_,
        'best_model': best_model
    }

    print(f"{model_name} - Train MSE: {mse_train}, Test MSE: {mse_test}, Overfitting Score: {overfitting_score}")
