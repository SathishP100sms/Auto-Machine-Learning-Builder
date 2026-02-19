import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def read_data(data_set):
    """
    Read data from CSV or Excel files
    """
    if data_set is not None:
        if data_set.name.endswith('.csv'):
            df = pd.read_csv(data_set)
            return df
        elif data_set.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_set)
            return df
    else:
        return None


def remove_columns(df, list):
    """
    Remove specified columns from dataframe
    """
    try:
        df = df.drop(columns=list, axis=1)
    except Exception as e:
        raise e    
    return df


def remove_outliers(data, numeric_cols):
    """
    Remove outliers using IQR method
    """
    Q1 = data[numeric_cols].quantile(0.25)
    Q3 = data[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (data[numeric_cols] >= lower_bound) & (data[numeric_cols] <= upper_bound)
    mask = mask.all(axis=1)
    return data[mask]


def get_data_statistics(df):
    """
    Generate detailed statistics about the dataset
    """
    stats = {
        'Shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Numeric Columns': len(df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns),
        'Categorical Columns': len(df.select_dtypes(include=['object']).columns)
    }
    return stats


def data_preprocessing(df, target_column, scaler_option, test_size, remove_outliers_flag=False, polynomial_degree=1):
    """
    Preprocess data: handle missing values, encode categorical variables, scale features
    """
    data = df.copy()

    if data[target_column].dtype == 'object':
        labelencoder = LabelEncoder()
        data[target_column] = labelencoder.fit_transform(data[target_column]) 

    numeric_cols = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Remove outliers if requested
    if remove_outliers_flag:
        data = remove_outliers(data, numeric_cols)
    
    imputer_num = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])

    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

    if len(categorical_cols) > 0:
        onehotencoder = OneHotEncoder(sparse_output=False, drop='first')
        data_encoded = onehotencoder.fit_transform(data[categorical_cols])
        data_encoded_df = pd.DataFrame(data_encoded, columns=onehotencoder.get_feature_names_out(categorical_cols))
        data = pd.concat([data.drop(columns=categorical_cols).reset_index(drop=True), data_encoded_df.reset_index(drop=True)], axis=1)

    data = data.dropna()
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]
    
    # Apply polynomial features if degree > 1
    if polynomial_degree > 1:
        poly_features = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        X = poly_features.fit_transform(X)
    
    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif scaler_option == "RobustScaler":
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    else:
        pass        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    return X_train, X_test, y_train, y_test
