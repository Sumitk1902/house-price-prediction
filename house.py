# House Price Prediction Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
import joblib

# Step 1: Load the dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")
print("\nFirst 5 Rows of Dataset:\n", dataset.head(5))
print("\nDataset Shape:", dataset.shape)

# Step 2: Data Type Analysis
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Step 3: Exploratory Data Analysis (EDA)
numerical_dataset = dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")  # Save instead of showing in .py script

# Categorical Bar Plot
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10, 6))
plt.title('No. of Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)
plt.tight_layout()
plt.savefig("categorical_unique_counts.png")

# Step 4: Data Cleaning
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
dataset = dataset.dropna()  # drop remaining nulls

# Step 5: OneHotEncoding for Categorical Columns
s = (dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("\nOneHotEncoding the following categorical columns:\n", object_cols)

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()

df_final = dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Step 6: Split into train and test sets
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Step 7: Train Models

# 1. Support Vector Regressor
model_SVR = SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_svr = model_SVR.predict(X_valid)
mape_svr = mean_absolute_percentage_error(Y_valid, Y_pred_svr)

# 2. Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred_rfr = model_RFR.predict(X_valid)
mape_rfr = mean_absolute_percentage_error(Y_valid, Y_pred_rfr)

# 3. Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_lr = model_LR.predict(X_valid)
mape_lr = mean_absolute_percentage_error(Y_valid, Y_pred_lr)

# Step 8: Show Results
print("\n--- Model Performance (MAPE - Lower is Better) ---")
print(f"Support Vector Regressor: {mape_svr:.4f}")
print(f"Random Forest Regressor : {mape_rfr:.4f}")
print(f"Linear Regressor        : {mape_lr:.4f}")

# Step 9: Visualize Predictions
plt.figure(figsize=(8, 6))
plt.scatter(Y_valid, Y_pred_svr, alpha=0.5, label="SVR")
plt.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'k--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (SVR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_svr.png")

# Step 10: Save best model
joblib.dump(model_SVR, "best_house_price_model.pkl")
print("\n✅ Best model (SVR) saved as 'best_house_price_model.pkl'")
