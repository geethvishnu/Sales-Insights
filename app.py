import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from scipy.stats import randint

# Load the dataset
@st.cache_data
def load_data():
    url = 'https://drive.google.com/uc?id=10sofXyF6NjwN6ngLyFfiPI-CUDpeqaN_'
    df = pd.read_csv(url)
    return df

df = load_data()

# Convert date columns to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

# Handle missing values
df.dropna(subset=['Order Date', 'Ship Date'], inplace=True)
df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
df['Unit Price'] = pd.to_numeric(df['Unit Price'], errors='coerce')
df['Unit Cost'] = pd.to_numeric(df['Unit Cost'], errors='coerce')
df['Total Revenue'] = pd.to_numeric(df['Total Revenue'], errors='coerce')
df['Total Cost'] = pd.to_numeric(df['Total Cost'], errors='coerce')
df['Total Profit'] = pd.to_numeric(df['Total Profit'], errors='coerce')
df.dropna(subset=['Units Sold', 'Unit Price', 'Unit Cost'], inplace=True)

# Calculate additional features
df['Profit Margin'] = (df['Total Profit'] / df['Total Revenue']) * 100
df['Delivery Days'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year
df['Year_Month'] = df['Order Date'].dt.to_period('M').astype(str)

# Encode categorical variables
label_encoders = {}
for column in ['Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and targets
features = df[['Units Sold', 'Unit Price', 'Unit Cost', 'Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority']]
target_profit = df['Total Profit']
target_sales = df['Total Revenue']
df['High Preference'] = (df['Total Revenue'] > df['Total Revenue'].median()).astype(int)
target_classification = df['High Preference']

# Train-test split
X_train, X_test, y_train_profit, y_test_profit = train_test_split(features, target_profit, test_size=0.2, random_state=42)
X_train, X_test, y_train_sales, y_test_sales = train_test_split(features, target_sales, test_size=0.2, random_state=42)
X_train, X_test, y_train_pref, y_test_pref = train_test_split(features, target_classification, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression for Classification with Hyperparameter Tuning
log_reg_params = {'C': [0.1, 1.0, 10.0, 100.0], 'solver': ['liblinear', 'lbfgs']}
log_reg_grid_search = GridSearchCV(LogisticRegression(), log_reg_params, cv=5, scoring='accuracy')
log_reg_grid_search.fit(X_train_scaled, y_train_pref)
log_reg_best_model = log_reg_grid_search.best_estimator_
y_pred_class_log_reg = log_reg_best_model.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test_pref, y_pred_class_log_reg)

# Decision Tree Classifier with Hyperparameter Tuning
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train_scaled, y_train_pref)
dt_best_model = dt_grid_search.best_estimator_
y_pred_class_dt = dt_best_model.predict(X_test_scaled)
dt_classification_report = classification_report(y_test_pref, y_pred_class_dt)

# Support Vector Classifier with RandomizedSearchCV
svc_param_dist = {
    'C': randint(1, 100),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf']
}
svc_random_search = RandomizedSearchCV(SVC(), svc_param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)
svc_random_search.fit(X_train_scaled, y_train_pref)
svc_best_model = svc_random_search.best_estimator_
y_pred_class_svc = svc_best_model.predict(X_test_scaled)
svc_accuracy = accuracy_score(y_test_pref, y_pred_class_svc)

# Stacking Model with Gradient Boosting Classifier
estimators = [
    ('log_reg', LogisticRegression(C=1.0, solver='liblinear')),
    ('svc', SVC(kernel='linear'))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier())
stacking_model.fit(X_train_scaled, y_train_pref)
y_pred_class_stack = stacking_model.predict(X_test_scaled)
stacking_accuracy = accuracy_score(y_test_pref, y_pred_class_stack)

# Cross-Validation Scores
log_reg_cv_scores = cross_val_score(log_reg_best_model, X_train_scaled, y_train_pref, cv=5, scoring='accuracy')
svc_cv_scores = cross_val_score(svc_best_model, X_train_scaled, y_train_pref, cv=5, scoring='accuracy')

# Feature Importance Visualization for Profit Prediction
profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
profit_model.fit(X_train, y_train_profit)
y_pred_profit = profit_model.predict(X_test)
mse_profit = mean_squared_error(y_test_profit, y_pred_profit)

importance_profit = profit_model.feature_importances_
feature_importance_profit = pd.DataFrame({'Feature': features.columns, 'Importance': importance_profit}).sort_values(by='Importance', ascending=False)



# Streamlit app layout
st.title("Sales and Profit Analysis Dashboard")

st.sidebar.header("Select Analysis")

option = st.sidebar.selectbox(
    "Choose Analysis",
    ["Overview", "Classification Models", "Visualizations"]
)

if option == "Overview":
    st.subheader("Dataset Overview")
    st.write("### Data Preview")
    st.write(df.head())
    st.write("### Summary Statistics")
    st.write(df.describe())

elif option == "Classification Models":
    st.subheader("Classification Models")
    st.write(f"### Logistic Regression - Accuracy: {log_reg_accuracy:.2f}")
    st.write(f"### SVC - Accuracy: {svc_accuracy:.2f}")
    st.write(f"### Stacking Model - Accuracy: {stacking_accuracy:.2f}")
  



elif option == "Visualizations":
    st.subheader("Data Visualizations")

    # Monthly sales trend
    st.write("### Monthly Sales Trend")
    monthly_sales = df.groupby('Year_Month').agg({
        'Total Revenue': 'sum',
        'Total Profit': 'sum'
    }).reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x='Year_Month', y='Total Revenue', marker='o', label='Revenue')
    sns.lineplot(data=monthly_sales, x='Year_Month', y='Total Profit', marker='o', label='Profit')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Amount')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Yearly sales trend
    st.write("### Yearly Sales Trend")
    yearly_sales = df.groupby('Year').agg({
        'Total Revenue': 'sum',
        'Total Profit': 'sum'
    }).reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=yearly_sales, x='Year', y='Total Revenue', color='blue', label='Revenue')
    sns.barplot(data=yearly_sales, x='Year', y='Total Profit', color='orange', label='Profit')
    plt.title('Yearly Sales Trend')
    plt.xlabel('Year')
    plt.ylabel('Amount')
    plt.legend()
    st.pyplot(fig)

    # Profit Margin Visualization
    st.write("### Profit Margin Distribution")
    fig = plt.figure(figsize=(12, 6))
    sns.histplot(df['Profit Margin'], bins=30, kde=True)
    plt.title('Distribution of Profit Margin')
    plt.xlabel('Profit Margin')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # Delivery Days Visualization
    st.write("### Delivery Days Distribution")
    fig = plt.figure(figsize=(12, 6))
    sns.histplot(df['Delivery Days'], bins=30, kde=True)
    plt.title('Distribution of Delivery Days')
    plt.xlabel('Delivery Days')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # Regional Performance Visualization
    st.write("### Regional Performance")
    region_performance = df.groupby('Region').agg({
        'Total Revenue': 'sum',
        'Total Profit': 'sum'
    }).reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=region_performance, x='Region', y='Total Revenue', color='blue', label='Revenue')
    sns.barplot(data=region_performance, x='Region', y='Total Profit', color='orange', label='Profit')
    plt.title('Regional Performance')
    plt.xlabel('Region')
    plt.ylabel('Amount')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Product Category Performance
    st.write("### Product Category Performance")
    product_category_performance = df.groupby('Item Type').agg({
        'Total Profit': 'mean',
        'Total Revenue': 'mean'
    }).reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=product_category_performance, x='Item Type', y='Total Profit', color='orange', label='Profit')
    sns.barplot(data=product_category_performance, x='Item Type', y='Total Revenue', color='blue', label='Revenue')
    plt.title('Product Category Performance')
    plt.xlabel('Item Type')
    plt.ylabel('Amount')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Most and Least Sold Items
    st.write("### Most Sold Items")
    most_sold_items = df.groupby('Item Type').agg({
        'Units Sold': 'sum'
    }).reset_index().sort_values(by='Units Sold', ascending=False)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=most_sold_items, x='Item Type', y='Units Sold', color='blue')
    plt.title('Most Sold Items')
    plt.xlabel('Item Type')
    plt.ylabel('Units Sold')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### Least Sold Items")
    least_sold_items = df.groupby('Item Type').agg({
        'Units Sold': 'sum'
    }).reset_index().sort_values(by='Units Sold')
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=least_sold_items, x='Item Type', y='Units Sold', color='red')
    plt.title('Least Sold Items')
    plt.xlabel('Item Type')
    plt.ylabel('Units Sold')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Regional Sales and Profit Performance
    st.write("### Regional Sales and Profit Performance")
    region_sales = df.groupby('Region').agg({
        'Total Revenue': 'sum',
        'Total Profit': 'sum',
        'Units Sold': 'sum'
    }).reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=region_sales, x='Region', y='Total Revenue', color='blue', label='Revenue')
    sns.barplot(data=region_sales, x='Region', y='Total Profit', color='orange', label='Profit')
    plt.title('Regional Sales and Profit Performance')
    plt.xlabel('Region')
    plt.ylabel('Amount')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
