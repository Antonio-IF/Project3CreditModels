import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

# Definir la función antes de su uso
def evaluar_modelo_mejorado(X, y, nombre_dataset):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Crear y entrenar el modelo Random Forest
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(X_train_balanced, y_train_balanced)
    
    # Evaluar el modelo en el conjunto de prueba
    y_pred_test = modelo.predict(X_test)
    y_pred_proba_test = modelo.predict_proba(X_test)[:, 1]
    
    print(f"\nReporte de Clasificación - {nombre_dataset} (Test)")
    print(classification_report(y_test, y_pred_test))
    
    # Curva ROC para el conjunto de prueba
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {nombre_dataset} (Test)')
    plt.legend(loc="lower right")

    # Matriz de Confusión para el conjunto de prueba
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre_dataset} (Test)')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    return modelo

# Leer y preparar el dataset de Taiwan
df_taiwan = pd.read_excel('Data/creditcards_default.xls')
df_taiwan = df_taiwan.iloc[1:].reset_index(drop=True)
df_taiwan = df_taiwan.drop(['Unnamed: 0'], axis=1)

# Convertir columnas a tipo numérico
for col in df_taiwan.columns:
    df_taiwan[col] = pd.to_numeric(df_taiwan[col], errors='coerce')

# Dividir los datos en entrenamiento y holdout (para evaluar en datos no vistos)
X_taiwan = df_taiwan.drop('Y', axis=1)
y_taiwan = df_taiwan['Y']
X_taiwan_train, X_taiwan_holdout, y_taiwan_train, y_taiwan_holdout = train_test_split(
    X_taiwan, y_taiwan, test_size=0.2, random_state=42
)

print("Tamaño del dataset de entrenamiento:", len(X_taiwan_train))
print("Tamaño del dataset holdout:", len(X_taiwan_holdout))
print("\nDistribución de clases en entrenamiento:", y_taiwan_train.value_counts(normalize=True))
print("Distribución de clases en holdout:", y_taiwan_holdout.value_counts(normalize=True))

# Evaluar modelo con datos de entrenamiento de Taiwan
modelo_taiwan = evaluar_modelo_mejorado(X_taiwan_train, y_taiwan_train, "Dataset Taiwan Entrenamiento")

# Evaluar el mismo modelo con datos holdout de Taiwan
print("\nEvaluación con datos no vistos (Holdout - Taiwan):")
y_pred_holdout = modelo_taiwan.predict(X_taiwan_holdout)
y_pred_proba_holdout = modelo_taiwan.predict_proba(X_taiwan_holdout)[:, 1]

# Calcular métricas para holdout
print("\nReporte de Clasificación - Dataset Taiwan Holdout")
print(classification_report(y_taiwan_holdout, y_pred_holdout))

# Graficar resultados del holdout
plt.figure(figsize=(15, 5))

# AUC-ROC para holdout
fpr, tpr, _ = roc_curve(y_taiwan_holdout, y_pred_proba_holdout)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Dataset Taiwan Holdout')
plt.legend(loc="lower right")

# Matriz de Confusión para holdout
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_taiwan_holdout, y_pred_holdout)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Dataset Taiwan Holdout')
plt.ylabel('Real')
plt.xlabel('Predicción')

plt.tight_layout()
plt.show()
