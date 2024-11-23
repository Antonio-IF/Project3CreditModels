import joblib
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, f1_score, precision_recall_curve, auc, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


class CreditModel:
    def __init__(self, data_path, model_type):
        self.data = pd.read_excel(data_path)
        self.data = self.data[self.data['PRODUCTO'] == 'BK']
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def eda(self):
        print(self.data.describe())
        for column in ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento']:
            self.data[column].hist()
            plt.title(f'Histogram of {column}')
            plt.show()

    def dqr(self):
        print("Missing values per column:")
        print(self.data.isnull().sum())
        print("\nData types and non-null counts:")
        print(self.data.info())

    def feature_engineering(self):
        self.data['Ratio_Ingreso_Egreso'] = self.data['Ingreso_Estimado'] / self.data['Egresos']
        print("Feature Engineering Done: New features added.")

    def prepare_data(self):
        #features = ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento', 'Ratio_Ingreso_Egreso']
        features = ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento']
        X = self.data[features]
        y = self.data['ESTATUS'].map({'APROBADO': 1, 'RECHAZADO': 0})
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_model(self):
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(tree_method='hist', device='gpu')
            #self.model = XGBClassifier(tree_method='hist')
            #self.model = XGBClassifier(tree_method='gpu_hist')

        cross_val_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cross_val_scores.mean()}")

        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        print("\nEvaluating model...")
        predictions = self.model.predict(self.X_test)
        probas = self.model.predict_proba(self.X_test)[:, 1]
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))
        print("Matthews Correlation Coefficient:", matthews_corrcoef(self.y_test, predictions))
        print("F1-Score:", f1_score(self.y_test, predictions))
        print("ROC AUC score:", roc_auc_score(self.y_test, probas))

        fpr, tpr, thresholds = roc_curve(self.y_test, probas)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        cm = confusion_matrix(self.y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self):
        filename = f'models/{self.model_type}_model.pkl'
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")

if __name__ == "__main__":
    pass


from tensorflow.keras.callbacks import EarlyStopping

class NeuralNetworkCreditModel(CreditModel):
    def __init__(self, data_path):
        super().__init__(data_path, model_type='neural_network')
        self.model = None

    def build_model(self, input_dim):
        """
        Define la arquitectura de la red neuronal.
        """
        self.model = Sequential([
            Dense(32, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')  # Salida binaria
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.00001),
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'AUC'])
        print("Red Neuronal Construida:")
        self.model.summary()

    def train_model(self, epochs=35, batch_size=32):
        """
        Entrenar la red neuronal con EarlyStopping.
        """
        # Crear un callback para detener el entrenamiento temprano si no hay mejora
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitorea la pérdida en el conjunto de validación
            patience=10,         # Permite hasta 10 épocas sin mejora antes de detener
            restore_best_weights=True  # Restaura los pesos con mejor rendimiento
        )

        self.build_model(input_dim=self.X_train.shape[1])
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],  # Agregar EarlyStopping
            verbose=1
        )
        print("Entrenamiento Completado.")
        return history

    def evaluate_model(self):
        """
        Evaluar la red neuronal y mostrar métricas.
        """
        print("\nEvaluando la Red Neuronal...")
        predictions = (self.model.predict(self.X_test) > 0.5).astype(int).flatten()
        probas = self.model.predict(self.X_test).flatten()

        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))
        print("Matthews Correlation Coefficient:", matthews_corrcoef(self.y_test, predictions))
        print("F1-Score:", f1_score(self.y_test, predictions))
        print("ROC AUC score:", roc_auc_score(self.y_test, probas))

        fpr, tpr, thresholds = roc_curve(self.y_test, probas)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        cm = confusion_matrix(self.y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self):
        """
        Guardar la red neuronal.
        """
        filename = f'models/{self.model_type}_model.h5'
        self.model.save(filename)
        print(f"Modelo guardado como {filename}")

