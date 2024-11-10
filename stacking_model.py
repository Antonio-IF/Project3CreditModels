import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar y preparar los datos (similar al archivo anterior)
df = pd.read_excel('Project3CreditModels/Data/creditcards_default.xls')
df = df.iloc[1:].reset_index(drop=True)
df = df.drop(['Unnamed: 0'], axis=1)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

X = df.drop(['Y'], axis=1)
y = df['Y']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE para balance de clases
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Definir modelos base con hiperparámetros más robustos
model1 = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    early_stopping_rounds=50,
    eval_metric='auc',
    random_state=42
)

model2 = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)

model3 = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Modelo meta
meta_model = LogisticRegression(random_state=42)

# División adicional para entrenamiento y validación
X_train_meta, X_val, y_train_meta, y_val = train_test_split(
    X_train_scaled, y_train_balanced, 
    test_size=0.2, 
    random_state=42
)

# Entrenamiento de modelos base
print("Entrenando modelos base...")
model1.fit(
    X_train_meta, y_train_meta,
    eval_set=[(X_val, y_val)],
    verbose=False
)

model2.fit(
    X_train_meta, y_train_meta,
    eval_set=[(X_val, y_val)]
)

model3.fit(X_train_meta, y_train_meta)

# Generar predicciones de nivel base
meta_features_train = np.column_stack((
    model1.predict_proba(X_train_scaled)[:, 1],
    model2.predict_proba(X_train_scaled)[:, 1],
    model3.predict_proba(X_train_scaled)[:, 1]
))

meta_features_test = np.column_stack((
    model1.predict_proba(X_test_scaled)[:, 1],
    model2.predict_proba(X_test_scaled)[:, 1],
    model3.predict_proba(X_test_scaled)[:, 1]
))

# Entrenar modelo meta
print("Entrenando modelo meta...")
meta_model.fit(meta_features_train, y_train_balanced)

# Predicciones finales
y_pred_proba = meta_model.predict_proba(meta_features_test)[:, 1]

# Calcular y mostrar curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Stacking (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo Stacking')
plt.legend(loc="lower right")
plt.show()

# Imprimir AUC-ROC de modelos individuales
print("\nAUC-ROC por modelo:")
print(f"XGBoost: {roc_auc_score(y_test, model1.predict_proba(X_test_scaled)[:, 1]):.4f}")
print(f"LightGBM: {roc_auc_score(y_test, model2.predict_proba(X_test_scaled)[:, 1]):.4f}")
print(f"Random Forest: {roc_auc_score(y_test, model3.predict_proba(X_test_scaled)[:, 1]):.4f}")
print(f"Stacking: {roc_auc:.4f}")

# Matriz de confusión
y_pred = meta_model.predict(meta_features_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Modelo Stacking')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación - Modelo Stacking:")
print(classification_report(y_test, y_pred)) 