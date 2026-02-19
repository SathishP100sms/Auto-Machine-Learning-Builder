import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, mean_absolute_error
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def handle_class_imbalance(X_train, y_train, strategy='smote'):
    """
    Handle class imbalance using SMOTE or hybrid approach
    """
    if strategy == 'smote':
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    elif strategy == 'hybrid':
        smote = SMOTE(random_state=42, k_neighbors=5)
        under_sampler = RandomUnderSampler(random_state=42)
        pipeline = ImbPipeline([
            ('smote', smote),
            ('under', under_sampler)
        ])
        X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    return X_train_balanced, y_train_balanced


def tune_hyperparameters(model_type, X_train, y_train, cv=3):
    """
    Hyperparameter tuning using GridSearchCV
    """
    param_grids = {
        "SVC": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
        "RandomForestClassifier": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
        "XGBoostClassifier": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        "LogisticRegression": {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
        "GradientBoostingClassifier": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        "KNeighborsClassifier": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    }
    
    if model_type not in param_grids:
        return None
    
    model_instance = _create_model_instance(model_type)
    grid_search = GridSearchCV(model_instance, param_grids[model_type], cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def _create_model_instance(model_type):
    """
    Create a base model instance for hyperparameter tuning
    """
    if model_type == "SVC":
        return SVC(probability=True, random_state=42)
    elif model_type == "RandomForestClassifier":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_type == "XGBoostClassifier":
        return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_type == "LogisticRegression":
        return LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "GradientBoostingClassifier":
        return GradientBoostingClassifier(random_state=42)
    elif model_type == "KNeighborsClassifier":
        return KNeighborsClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_feature_importance(model, feature_names, model_type):
    """
    Extract feature importance from tree-based and linear models
    """
    feature_importance = None
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For linear models
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    
    return feature_importance


def plot_feature_importance(feature_importance_df):
    """
    Plot feature importance visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance_df.head(15)
    ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Feature Importances')
    ax.invert_yaxis()
    return fig


def build_model(model_type, X_train, y_train):
    """
    Build and train a classification model
    """
    if model_type == "SVC":
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100, max_depth=None)
    elif model_type == "XGBoostClassifier":
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_type == "LogisticRegression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
    elif model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    training_acc = model.score(X_train, y_train)
    return model, training_acc


def evaluate(model, X_test, y_test):
    """
    Evaluate classification model on test set
    """
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy, y_pred


def build_regressor_model(model_type, X_train, y_train):
    """
    Build and train a regression model
    """
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoostRegressor":
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    elif model_type == "Ridge":
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == "Lasso":
        model = Lasso(alpha=0.1, random_state=42)
    elif model_type == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_type == "KNeighborsRegressor":
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def regressor_evaluate(model, X_test, y_test):
    """
    Evaluate regression model on test set
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r_2 = r2_score(y_test, y_pred)
    return mse, mae, r_2


class plots():
    """
    Visualization class for model evaluation plots
    """
    def heat_map(df):
        """
        Generate correlation heatmap
        """
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8}, ax=ax) 
        return fig

    def conf_matrix(y_test, y_pred):
        """
        Generate confusion matrix plot
        """
        confusion = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d", ax=ax) 
        return fig
