import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from data_utils import (read_data,
                        remove_columns,
                        data_preprocessing,
                        remove_outliers,
                        get_data_statistics)

from model_utils import (build_model,
                         build_regressor_model,
                         evaluate,
                         regressor_evaluate,
                         plots,
                         handle_class_imbalance,
                         tune_hyperparameters,
                         get_feature_importance,
                         plot_feature_importance)


st.set_page_config(
    page_title="Auto Machine Learning Builder",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Auto Machine Learning Builder")
st.write("Welcome to the Auto Machine Learning Builder! This application helps you build and evaluate machine learning models with ease.")
st.divider()

dataset = st.file_uploader("📊 Upload your dataset", type=["csv","xlsx","xls"])

df = read_data(dataset)

if df is not None:
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
    
    st.divider()
    with st.expander("📊 Show Correlation Heatmap", expanded=False):
        st.write("Correlation matrix of numeric features:")
        heat_map = plots.heat_map(df)
        st.pyplot(heat_map) 

    st.divider()
    with st.expander("� Data Statistics", expanded=False):
        st.write("Key statistics about your dataset:")
        stats = get_data_statistics(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Shape", stats['Shape'])
        with col2:
            st.metric("Missing Values", stats['Missing Values'])
        with col3:
            st.metric("Duplicate Rows", stats['Duplicate Rows'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Memory Usage", stats['Memory Usage'])
        with col2:
            st.metric("Numeric Columns", stats['Numeric Columns'])
        with col3:
            st.metric("Categorical Columns", stats['Categorical Columns'])
        
        # Display numeric column statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns Summary:**")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    
    st.divider()
    with st.expander("🗑️ Remove Columns", expanded=False):
        st.write("Select columns you want to remove from the dataset:")
        columns = st.multiselect("Select Columns to remove", options=df.columns)
        if columns:
            if st.button("Remove Selected Columns", key="remove_btn"):
                df = remove_columns(df, columns)
                st.success(f"✅ {len(columns)} column(s) removed successfully!")
                st.dataframe(df.head(), use_container_width=True)

    st.divider()
    with st.expander("🔧 Advanced Preprocessing Options", expanded=False):
        st.write("Configure preprocessing steps:")
        col1, col2, col3 = st.columns(3)
        with col1:
            remove_outliers_flag = st.checkbox("Remove Outliers (IQR Method)", value=False)
        with col2:
            polynomial_degree = st.selectbox("Polynomial Features Degree", options=[1, 2, 3], index=0)
        with col3:
            handle_imbalance = st.checkbox("Handle Class Imbalance (SMOTE)", value=False, help="Uses SMOTE to balance imbalanced datasets")
    
    
    st.divider()
    problem_type = st.selectbox("🎯 Select the problem type", options=["Classification", "Regression"])    

    if problem_type == "Classification":
        st.subheader("⚙️ Classification Configuration")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            target_column = st.selectbox("Target Column", options=df.columns)
        with col2:
            scaler_option = st.selectbox("Scaler Type", options=["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])        
        with col3:     
            model_type = st.selectbox("Model Type", options=["SVC", "RandomForestClassifier", "XGBoostClassifier", "LogisticRegression", "GradientBoostingClassifier", "KNeighborsClassifier"])
        with col4:
            test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5)
        
        col1, col2 = st.columns(2)
        with col1:
            use_hyperparameter_tuning = st.checkbox("🎯 Enable Hyperparameter Tuning", value=False, help="Use GridSearchCV to find best hyperparameters (slower but more accurate)")

        if st.button("🚀 Build Classification Model", type="primary", use_container_width=True):
            with st.spinner("Building the model..."):
                X_train, X_test, y_train, y_test = data_preprocessing(df, target_column, scaler_option, test_size, remove_outliers_flag, polynomial_degree) 
                
                # Handle class imbalance
                if handle_imbalance:
                    with st.spinner("Applying SMOTE for class imbalance..."):
                        X_train, y_train = handle_class_imbalance(X_train, y_train, strategy='smote')
                        st.info(f"✅ SMOTE Applied: Training set rebalanced")
                
                # Hyperparameter tuning
                train_accuracy = None
                if use_hyperparameter_tuning:
                    with st.spinner("Tuning hyperparameters... (this may take a moment)"):
                        tuning_result = tune_hyperparameters(model_type, X_train, y_train, cv=3)
                        if tuning_result:
                            model, best_params, best_score = tuning_result
                            st.info(f"🎯 Best Parameters: {best_params}")
                            st.info(f"📊 Best CV Score: {best_score:.4f}")
                            train_accuracy = best_score
                        else:
                            model, train_accuracy = build_model(model_type, X_train, y_train)
                else:
                    model, train_accuracy = build_model(model_type, X_train, y_train)
                
                test_accuracy, y_pred = evaluate(model, X_test, y_test)
            
            st.divider()
            st.subheader("📊 Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                if train_accuracy is not None:
                    st.metric("Train Accuracy", f"{train_accuracy*100:.2f}%")
                else:
                    st.metric("Train Accuracy", "Tuned Model")
            with col2:
                st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
            
            st.success("✅ Model building and evaluation completed!")
            st.divider()
            
            st.subheader("🔍 Confusion Matrix")    
            confusion_mat = plots.conf_matrix(y_test, y_pred)
            st.pyplot(confusion_mat)
            
            # Feature Importance
            st.divider()
            st.subheader("⭐ Feature Importance")
            try:
                # Get feature names from X_train (after preprocessing)
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                feature_importance_df = get_feature_importance(model, feature_names, model_type)
                
                if feature_importance_df is not None:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = plot_feature_importance(feature_importance_df)
                        st.pyplot(fig)
                    with col2:
                        st.write("**Top 10 Features:**")
                        st.dataframe(feature_importance_df.head(10), use_container_width=True)
                else:
                    st.info("This model type does not support feature importance extraction.")
            except Exception as e:
                st.warning(f"Could not extract feature importance: {str(e)}")
                            

    elif problem_type == "Regression":
        st.subheader("⚙️ Regression Configuration")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            target_column = st.selectbox("Target Column", options=df.columns, key="reg_target")
        with col2:
            scaler_option = st.selectbox("Scaler Type", options=["StandardScaler", "MinMaxScaler", "RobustScaler", "None"], key="reg_scaler")        
        with col3:     
            model_type = st.selectbox("Model Type", options=["Linear Regression", "RandomForestRegressor", "XGBoostRegressor", "Ridge", "Lasso", "GradientBoostingRegressor", "KNeighborsRegressor"], key="reg_model")
        with col4:
            test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5, key="reg_test")
            
        if st.button("🚀 Build Regression Model", type="primary", use_container_width=True):
            with st.spinner("Building the model..."):
                X_train, X_test, y_train, y_test = data_preprocessing(df, target_column, scaler_option, test_size, remove_outliers_flag, polynomial_degree) 
                model = build_regressor_model(model_type, X_train, y_train)
                mse, mae, r_2 = regressor_evaluate(model, X_test, y_test)
            
            st.divider()
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with col2:
                st.metric("Mean Absolute Error", f"{mae:.4f}")
            with col3:
                st.metric("R² Score", f"{r_2:.4f}")
            st.success("✅ Model building and evaluation completed!")    
    else:
        st.info("⚠️ Please select a problem type to continue")
else:
    st.info("📤 Please upload a dataset to get started")            
              