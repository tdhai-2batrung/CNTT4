#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns


# In[8]:


df = pd.read_csv(r'C:\Users\Admin\Downloads\KiemTra_K234\CSDL_TieuDuong.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())


# In[9]:


#du lieu thieu 
print(df.isnull().sum())


# In[10]:


#it du lieu thieu nen lua chon xoa
df = df.dropna()


# In[11]:


#xu ly du lieu o cot gioi tinh
df["GioiTinh"] = (
    df["GioiTinh"]
    .astype("string")
    .str.strip()
    .str.lower()
)

mapping = {
    "nam": "Nam",
    "male": "Nam",
    "m": "Nam",
    "nữ": "Nữ",
    "nu": "Nữ",
    "female": "Nữ",
    "f": "Nữ"
}

df["GioiTinh"] = df["GioiTinh"].map(mapping)


# In[12]:


print(df["GioiTinh"].value_counts(dropna=False))


# In[13]:


#dong bo du lieu cac cot 
numeric_cols = [
    "Tuoi",
    "Glucose",
    "HbA1C",
    "Cholesterol",
    "HDL",
    "LDL",
    "Triglycerid",
    "Creatinine",
    "Ure",
    "AcidUric",
    "AST",
    "ALT"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# In[14]:


print(df.dtypes)


# In[15]:


#kiem tra gia tri am#ki
for col in numeric_cols:
    negative_count = (df[col] < 0).sum()

    print(col, negative_count)

    


# In[16]:


#kiem tra du lieu trung lap
print(df.duplicated().sum())


# In[17]:


#kiem tra nhan du doan
print(
    df["NguyCoTieuDuong"].value_counts(
        dropna=False
    )
)


# In[20]:


#Xu ly nhan
df["NguyCoTieuDuong"] = (
    df["NguyCoTieuDuong"]
    .astype(str)
    .str.strip()
    .str.upper()
)

mapping = {
    "TRUE": 1,
    "FALSE": 0,
    "1": 1,
    "0": 0,
    "KHÔNG CÓ NGUY CƠ": 0
}

df["NguyCoTieuDuong"] = df["NguyCoTieuDuong"].map(mapping)


# In[21]:


print(
    df["NguyCoTieuDuong"].value_counts(
        dropna=False
    )
)


# In[22]:


#kiem tra outlier

"""plt.figure(figsize=(8, 4))

sns.boxplot(x=df["Glucose"])

plt.title("Glucose Outliers")

plt.show()
"""

# In[23]:


#phan chia du lieu
from sklearn.model_selection import train_test_split

X = df.drop(columns=["NguyCoTieuDuong"])
y = df["NguyCoTieuDuong"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# In[25]:


#pipeline tien xu ly 
numeric_cols = [
    "Tuoi", "Glucose", "HbA1C",
    "Cholesterol", "HDL", "LDL",
    "Triglycerid", "Creatinine",
    "Ure", "AcidUric", "AST", "ALT"
]

categorical_cols = ["GioiTinh"]
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# In[26]:


#Hoi quy logistic
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_val)

y_prob_lr = logistic_model.predict_proba(X_val)[:, 1]


# In[27]:


#danh gia 
print("Hoi quy logistic")
print(classification_report(y_val, y_pred_lr))

print("ROC-AUC:", roc_auc_score(y_val, y_prob_lr))


# In[28]:


#Rung cay ngau nhien 
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)

y_prob_rf = rf_model.predict_proba(X_val)[:, 1]


# In[29]:


#danh gia
print("Rung ngau nhien")
print(classification_report(y_val, y_pred_rf))

print("ROC-AUC:", roc_auc_score(y_val, y_prob_rf))


# In[30]:


#so sanh 2 mo hinh 
def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_val,
        y_pred_lr,
        y_prob_lr
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_val,
        y_pred_rf,
        y_prob_rf
    )
)

results_df = pd.DataFrame(results)

print(results_df.round(4))


# In[31]:


#quyet dinh chon mo hinh rung ngau nhien de su dung tap test
rff_model = rf_model

y_test_pred = rff_model.predict(X_test)
y_test_prob = rff_model.predict_proba(X_test)[:, 1]

print("Mo hinh tot nhat : Rung ngau nhien")
print(classification_report(y_test, y_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_test_prob))

import joblib
best_model = rff_model
joblib.dump(best_model, "diabetes_model.joblib")
# In[ ]:




