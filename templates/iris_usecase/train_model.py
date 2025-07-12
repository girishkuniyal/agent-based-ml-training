import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
import json
import os

def train_pipeline(df, target_col, model_type='classification'):
    # Validate model_type
    if model_type not in ['classification', 'regression', 'auto']:
        raise ValueError("model_type must be one of 'classification', 'regression', or 'auto'.")

    # Check for missing values in the target column
    if df[target_col].isnull().any():
        raise ValueError("Target column contains missing values.")
    
    # Preprocessing
    # Drop identity column if it exists
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing for numerical and categorical features
    numeric_features = X.select_dtypes(include=['float64', 'int']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define models to evaluate
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier()
    }
    
    # Define hyperparameters for tuning
    param_grid = {
        'RandomForest': {'model__n_estimators': [50, 100, 200]},
        'SVC': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
        'LogisticRegression': {'model__C': [0.1, 1, 10]},
        'GradientBoosting': {'model__n_estimators': [50, 100, 200]}
    }
    
    best_model = None
    best_score = 0
    best_model_name = ''
    results = {}
    
    for model_name, model in models.items():
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Evaluate the model
        train_score = grid_search.best_score_
        test_score = grid_search.score(X_test, y_test)
        
        # Save results
        results[model_name] = {
            'train_score': train_score,
            'test_score': test_score,
            'best_params': grid_search.best_params_
        }
        
        # Update best model if current model is better
        if test_score > best_score:
            best_score = test_score
            best_model = grid_search.best_estimator_
            best_model_name = model_name
    
    # Ensure the directory exists
    os.makedirs('usecases/iris_usecase', exist_ok=True)
    
    # Save the best model pipeline
    dump(best_model, 'usecases/iris_usecase/model.joblib')
    
    # Save the results to a JSON file
    with open('usecases/iris_usecase/experiments.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return best_model, best_model_name, train_score, test_score, 'accuracy'

# Load the data
df = pd.read_csv('usecases/iris_usecase/training_data.csv')

# Train the model pipeline
best_model, best_model_name, train_score, test_score, metric_name = train_pipeline(df, 'Species')