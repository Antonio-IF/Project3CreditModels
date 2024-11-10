import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar y preparar los datos
df = pd.read_excel('Project3CreditModels/Data/creditcards_default.xls')
df = df.iloc[1:].reset_index(drop=True)

# Convertir columnas a tipo numérico
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar columnas innecesarias
df = df.drop(['Unnamed: 0'], axis=1)

# Preparar X e y
X = df.drop(['Y'], axis=1)
y = df['Y']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Crear pipeline con hiperparámetros fijos
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        n_estimators=80,        # Número de árboles en el bosque
        max_depth=7,            # Profundidad máxima de cada árbol
        learning_rate=0.001,     # Tasa de aprendizaje (paso de actualización)
        subsample=0.8,          # Fracción de muestras usadas para entrenar cada árbol
        colsample_bytree=0.8,   # Fracción de características usadas para cada árbol
        min_child_weight=3,     # Suma mínima de peso necesaria en un nodo hijo
        random_state=42         # Semilla para reproducibilidad
    ))
])

print("Entrenando modelo XGBoost...")
pipeline.fit(X_train_balanced, y_train_balanced)

# Obtener modelo
model = pipeline
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Métricas y visualizaciones (igual que en logistic_regression.py)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - XGBoost')
plt.legend(loc="lower right")
plt.show()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - XGBoost')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.named_steps['classifier'].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Variables más Importantes - XGBoost')
plt.tight_layout()
plt.show()

# Métricas finales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== MÉTRICAS FINALES DEL MODELO XGBOOST ===")
print(f"Accuracy (Exactitud): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")

# Validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\n=== VALIDACIÓN CRUZADA ===")
print(f"Accuracy promedio en CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})") 