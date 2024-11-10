import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
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

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Crear pipeline con escalado y modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Definir parámetros para búsqueda
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__class_weight': ['balanced', None]
}

# Realizar búsqueda de hiperparámetros
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Entrenar modelo con los mejores parámetros
print("Entrenando modelo...")
grid_search.fit(X_train_balanced, y_train_balanced)

# Obtener mejor modelo
best_model = grid_search.best_estimator_
print("\nMejores parámetros:", grid_search.best_params_)

# Evaluar modelo
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Imprimir métricas
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Calcular y mostrar curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Matriz de confusión con porcentajes
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues')
plt.title('Matriz de Confusión (Porcentajes)')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()

# Importancia de variables
nombres_variables = ['Límite de Crédito', 'Sexo', 'Educación', 'Estado Civil', 'Edad',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

coef = best_model.named_steps['classifier'].coef_[0]
importancia = pd.DataFrame({
    'Variable': nombres_variables,
    'Importancia': np.abs(coef)
})
importancia = importancia.sort_values('Importancia', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importancia.head(10), x='Importancia', y='Variable')
plt.title('Top 10 Variables más Importantes')
plt.tight_layout()
plt.show()

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== MÉTRICAS FINALES DEL MODELO ===")
print(f"Accuracy (Exactitud): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")

# Calcular métricas específicas para cada clase
print("\n=== PREDICCIONES POR CLASE ===")
print("Clase 0 (No Default):")
print(f"Accuracy: {accuracy_score(y_test[y_test==0], y_pred[y_test==0]):.4f}")
print("Clase 1 (Default):")
print(f"Accuracy: {accuracy_score(y_test[y_test==1], y_pred[y_test==1]):.4f}")

# Validación cruzada para ver la estabilidad del modelo
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("\n=== VALIDACIÓN CRUZADA ===")
print(f"Accuracy promedio en CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
