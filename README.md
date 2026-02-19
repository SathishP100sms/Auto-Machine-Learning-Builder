# 🤖 Auto Machine Learning Builder

A powerful **Streamlit-based web application** that simplifies machine learning model building and evaluation. No coding expertise required! Just upload your dataset and let the app handle data preprocessing, model training, and performance evaluation.

## ✨ Features

### 📊 Data Management
- **File Upload** - Support for CSV and Excel files
- **Data Preview** - Display first few rows with shape and column count
- **Correlation Heatmap** - Visualize relationships between numeric features
- **Data Statistics** - Comprehensive dataset insights including:
  - Dataset dimensions and memory usage
  - Missing values and duplicates count
  - Numeric and categorical column breakdown
  - Detailed statistics for numeric columns (mean, std, min, max, quartiles)
- **Column Removal** - Easily remove unwanted columns

### 🔧 Advanced Preprocessing
- **Outlier Detection** - Remove outliers using IQR (Interquartile Range) method
- **Polynomial Features** - Generate polynomial combinations (degree 1-3)
- **Scaling Options**:
  - StandardScaler (zero mean, unit variance)
  - MinMaxScaler (0-1 range)
  - RobustScaler (resistant to outliers)
  - No scaling
- **Categorical Encoding** - One-Hot Encoding with automatic handling
- **Missing Value Imputation**:
  - Numeric: Mean strategy
  - Categorical: Most frequent strategy

### 🎯 Classification Models
Build and evaluate with 6 different algorithms:
- Support Vector Machine (SVC)
- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression
- Gradient Boosting Classifier
- K-Nearest Neighbors (KNN)

### 📈 Regression Models
Build and evaluate with 7 different algorithms:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor

### 🎯 Hyperparameter Tuning
- **GridSearchCV Integration** - Automatically find optimal hyperparameters
- **Cross-Validation** - 3-fold CV by default (customizable)
- **Best Parameters Display** - Shows optimal configuration and CV score

### ⚖️ Class Imbalance Handling
- **SMOTE** - Synthetic Minority Over-sampling for balanced training
- **Hybrid Strategy** - Combines SMOTE with undersampling
- **Automatic Detection** - Apply to imbalanced datasets

### ⭐ Feature Importance
- **Automatic Extraction** - Supports tree-based and linear models
- **Top 15 Visualization** - Bar chart of most important features
- **Detailed Rankings** - Top 10 features table with importance scores

### 📊 Model Evaluation
- **Confusion Matrix** - Visual classification performance breakdown
- **Accuracy Metrics** - Train and test accuracy comparison
- **Regression Metrics**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AutoMLPrediction.git
   cd AutoMLPrediction
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Run the Application
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

### Workflow

1. **Upload Dataset** - Click the file uploader and select your CSV or Excel file
2. **Explore Data** - View statistics, correlation heatmap, and data info
3. **Clean Data** - Remove columns, handle outliers, manage missing values
4. **Configure Model** - Select:
   - Problem type (Classification or Regression)
   - Target column
   - Model algorithm
   - Preprocessing options (scaling, polynomial features)
   - Test size
5. **Advanced Options** (Optional)
   - Enable hyperparameter tuning
   - Handle class imbalance
6. **Build & Evaluate** - Click the build button and get instant results
7. **Analyze Results** - View performance metrics, confusion matrix, feature importance

## 📁 Project Structure

```
AutoMLPredection/
├── main.py                 # Main Streamlit application
├── data_utils.py          # Data loading and preprocessing functions
├── model_utils.py         # Model building and evaluation functions
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 📦 Dependencies

- **streamlit** - Web app framework
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **xgboost** - Gradient boosting
- **imbalanced-learn** - Class imbalance handling
- **seaborn** - Statistical visualizations
- **matplotlib** - Plotting library
- **openpyxl** - Excel file support

## 🎯 Features Breakdown

### Data Utilities (`data_utils.py`)
- `read_data()` - Load CSV/Excel files
- `remove_columns()` - Remove unwanted columns
- `remove_outliers()` - IQR-based outlier removal
- `get_data_statistics()` - Generate dataset insights
- `data_preprocessing()` - Complete preprocessing pipeline

### Model Utilities (`model_utils.py`)
- `build_model()` - Train classification models
- `evaluate()` - Classify and calculate accuracy
- `build_regressor_model()` - Train regression models
- `regressor_evaluate()` - Evaluate regression performance
- `handle_class_imbalance()` - SMOTE implementation
- `tune_hyperparameters()` - GridSearchCV wrapper
- `get_feature_importance()` - Extract feature importance
- `plot_feature_importance()` - Visualize importance
- `plots` class - Heatmap and confusion matrix visualizations

## 🐛 Error Handling

The application includes robust error handling for:
- Empty categorical columns during imputation
- Unsupported model types
- Missing feature importance support
- Invalid file formats

## 📝 Example Workflow

1. Upload `iris.csv`
2. Select "Classification" and "Species" as target
3. Choose "RandomForestClassifier" model
4. Enable hyperparameter tuning
5. Click "Build Classification Model"
6. View confusion matrix and feature importance
7. Get test accuracy and model insights

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 👨‍💻 Author

Created as an AutoML solution to simplify machine learning for data analysts and scientists.

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Happy Machine Learning! 🚀**
