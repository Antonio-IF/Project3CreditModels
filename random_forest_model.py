import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar y preparar los datos (igual que en xgboost_model.py)
df = pd.read_excel('Data/creditcards_default.xls')
df = df.iloc[1:].reset_index(drop=True)

# Convertir columnas a tipo numérico
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.drop(['Unnamed: 0'], axis=1)
X = df.drop(['Y'], axis=1)
y = df['Y']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Crear pipeline base
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Grid de hiperparámetros más ligero y eficiente
param_grid = {
    'classifier__n_estimators': [25, 30, 60],      # Reducido a valores más bajos
    'classifier__max_depth': [4, 8, 10],           # Reducido el rango
    'classifier__min_samples_split': [2, 4, 6],      # Un solo valor
    'classifier__min_samples_leaf': [2, 4, 6],        # Un solo valor
    'classifier__max_features': ['sqrt'],       # Mantenemos solo 'sqrt'
    'classifier__class_weight': ['balanced']    # Mantenemos solo 'balanced'
}

# GridSearchCV con menos folds
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,                    # Reducido de 5 a 3 folds
    scoring='accuracy', # ['f1', 'precision', 'recall']
    #refit='accuracy', # or put 'recall', 'precision' or 'accuracy'
    n_jobs=-1,
    verbose=1
)

# Entrenar modelo con grid search
print("Realizando búsqueda de hiperparámetros...")
grid_search.fit(X_train_balanced, y_train_balanced)

# Obtener mejor modelo
best_model = grid_search.best_estimator_
print("\nMejores parámetros:", grid_search.best_params_)

# Función para curvas de aprendizaje
def plot_learning_curves(estimator, X, y, title="Curvas de Aprendizaje"):
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Score de entrenamiento', color='blue', marker='o')
    plt.plot(train_sizes, valid_mean, label='Score de validación', color='red', marker='o')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, 
                     alpha=0.15, color='red')
    
    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Evaluar mejor modelo
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Comparar métricas de entrenamiento y prueba para detectar overfitting
print("\nMétricas de entrenamiento vs prueba:")
print(f"Accuracy entrenamiento: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Accuracy prueba: {accuracy_score(y_test, y_test_pred):.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_test_pred))

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
plt.title('Curva ROC - Random Forest')
plt.legend(loc="lower right")
plt.show()

# Matriz de confusión
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.named_steps['classifier'].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Variables más Importantes - Random Forest')
plt.tight_layout()
plt.show()

# Añadir curvas de aprendizaje
print("\nGenerando curvas de aprendizaje...")
plot_learning_curves(best_model, X, y, "Curvas de Aprendizaje - Random Forest")

# Validación cruzada con el mejor modelo
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("\n=== VALIDACIÓN CRUZADA CON MEJOR MODELO ===")
print(f"Accuracy promedio en CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Métricas finales con el mejor modelo
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("\n=== MÉTRICAS FINALES DEL MEJOR MODELO RANDOM FOREST ===")
print(f"Accuracy (Exactitud): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)") 