# FIXED Complete code for Project 1 - Customer Churn Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Step 1: Data Ingestion
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Dataset shape:", df.shape)

# Step 2: Data Cleaning and Preprocessing
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
print("Cleaned shape:", df.shape)

# Label Encoding
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        df[col] = le.fit_transform(df[col].astype(str))
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Step 3: Feature Engineering
X = df.drop('Churn', axis=1)
y = df['Churn']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features shape:", X.shape)
print("Churn distribution:\n", df['Churn'].value_counts())

# FIXED Visualizations
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')

plt.subplot(1, 3, 2)
# FIXED: Create proper DataFrame for correlation plot
corr_data = pd.DataFrame(X_scaled).corrwith(y).sort_values(ascending=False)
corr_df = pd.DataFrame({'correlation': corr_data.values[:10]}, index=corr_data.index[:10])
sns.barplot(data=corr_df, x='correlation', y=corr_df.index)
plt.title('Top 10 Churn Factors')

plt.subplot(1, 3, 3)
sns.boxplot(x='Churn', y='TotalCharges', data=df)
plt.title('TotalCharges vs Churn')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\n=== MODEL PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(1, 2, 2)
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Top 10 Feature Importance for Churn')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Step 6: Business Insights
print("\n=== BUSINESS INSIGHTS ===")
print("Top 5 Churn Drivers:")
for i, (feature, imp) in enumerate(feat_imp.head().items(), 1):
    print(f"{i}. {feature}: {imp:.3f}")

# Step 7: Prediction Function
def predict_churn(new_customer_features):
    """Predict churn for new customer (must match training features order)"""
    new_scaled = scaler.transform([new_customer_features])
    prob = model.predict_proba(new_scaled)[0][1]
    prediction = "High Risk" if prob > 0.5 else "Low Risk"
    return prediction, prob

# Example prediction
sample_features = X.iloc[0].values  # First row as example
risk, prob = predict_churn(sample_features)
print(f"\nSample Prediction: {risk} (Probability: {prob:.2%})")

print("\n✅ Code executed successfully!")
