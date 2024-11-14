import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar y preparar los datos
df = pd.read_excel('Data/creditcards_default.xls')
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

# Crear pipeline con la red neuronal optimizada
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),  # Arquitectura más profunda
        activation='relu',                # Función de activación ELU para mejor gradiente
        solver='adam',                  
        max_iter=500,                   # Más iteraciones para mejor convergencia
        random_state=42,
        learning_rate_init=0.0005,      # Learning rate más pequeño para mejor precisión
        learning_rate='adaptive',       # Tasa de aprendizaje adaptativa
        batch_size=128,                 # Batch size optimizado para M1
        early_stopping=True,            
        validation_fraction=0.15,       # Más datos para validación
        n_iter_no_change=15,           # Más paciencia para encontrar mejor solución
        alpha=0.0005,                  # Regularización L2 ajustada
        momentum=0.9,                  # Momentum para mejor convergencia
        nesterovs_momentum=True,       # Usar momentum de Nesterov
        power_t=0.5,                   # Tasa de decaimiento para learning rate
        tol=1e-5,                      # Tolerancia más estricta
    ))
])

print("Entrenando modelo de Red Neuronal...")
pipeline.fit(X_train_balanced, y_train_balanced)

# Obtener modelo
model = pipeline
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Métricas y visualizaciones
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
plt.title('Curva ROC - Red Neuronal')
plt.legend(loc="lower right")
plt.show()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Red Neuronal')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()

# Métricas finales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== MÉTRICAS FINALES DEL MODELO DE RED NEURONAL ===")
print(f"Accuracy (Exactitud): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")

# Validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\n=== VALIDACIÓN CRUZADA ===")
print(f"Accuracy promedio en CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})") 