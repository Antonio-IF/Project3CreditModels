import os
import optuna
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, f1_score, precision_recall_curve, auc, roc_curve, confusion_matrix

class CreditModel:
    def __init__(self, data_path, model_type):
        print(f"\n{'='*50}\nInitializing 'CreditModel' ({model_type})...\n{'='*50}")
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
        print(f"\n{'='*50}\nPerforming Exploratory Data Analysis (EDA)...\n{'='*50}")
        print(self.data.describe())
        for column in ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento']:
            self.data[column].hist()
            plt.title(f'Histogram of {column}')
            plt.show()

    def dqr(self):
        print(f"\n{'='*50}\nPerforming Data Quality Review (DQR)...\n{'='*50}")
        print("Missing values per column:")
        print(self.data.isnull().sum())
        print("\nData types and non-null counts:")
        print(self.data.info())

    def feature_engineering(self):
        print(f"\n{'='*50}\nConducting Feature Engineering...\n{'='*50}")
        self.data['Ratio_Ingreso_Egreso'] = self.data['Ingreso_Estimado'] / self.data['Egresos']
        print("Feature Engineering Done: New features added.")

    def prepare_data(self):
        print(f"\n{'='*50}\nPreparing data for training...\n{'='*50}")
        #features = ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento', 'Ratio_Ingreso_Egreso']
        features = ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento']
        X = self.data[features]
        y = self.data['ESTATUS'].map({'APROBADO': 1, 'RECHAZADO': 0})
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def build_model(self):
        print(f"\n{'='*50}\nBuilding model...\n{'='*50}")
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(tree_method='hist', device='cpu')
        elif self.model_type == 'neural_network':
            input_layer = Input(shape=(self.X_train.shape[1],))
            x = Dense(32, activation='relu')(input_layer)
            x = Dropout(0.2)(x)
            x = Dense(16, activation='relu')(x)
            x = Dropout(0.1)(x)
            output_layer = Dense(1, activation='sigmoid')(x)
            self.model = Model(inputs=input_layer, outputs=output_layer)
            self.model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

    def train_model(self, epochs=None, batch_size=None):
        print(f"\n{'='*50}\nTraining model...\n{'='*50}")
        if self.model_type == 'neural_network':
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            return self.model.fit(self.X_train, self.y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
        else:
            self.model.fit(self.X_train, self.y_train)
            return None

    def evaluate_model(self):
        print(f"\n{'='*50}\nEvaluating model...\n{'='*50}")
        predictions = (self.model.predict(self.X_test) > 0.5).astype(int).flatten() if self.model_type == 'neural_network' else self.model.predict(self.X_test)
        probas = self.model.predict_proba(self.X_test)[:, 1] if not self.model_type == 'neural_network' else self.model.predict(self.X_test).flatten()
        print("\n* Classification Report:\n")
        print(classification_report(self.y_test, predictions))
        print(f"{'-'*50}\n\n* Matthews Correlation Coefficient:", np.round(matthews_corrcoef(self.y_test, predictions), 6))
        print("* F1-Score:", np.round(f1_score(self.y_test, predictions), 6))
        print("* ROC AUC score:", np.round(roc_auc_score(self.y_test, probas), 6))
        self.plot_metrics(predictions, probas)

    def plot_metrics(self, predictions, probas):
        print(f"\n{'='*50}\nPlotting evaluation metrics...\n{'='*50}")
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

    def save_model(self, model_directory='models'):
        print(f"\n{'='*50}\nSaving model...\n{'='*50}")
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
            print(f"\n¡Created directory: '{model_directory}'!\n\n{'-'*50}")

        if self.model_type == 'neural_network':
            filename = f'{model_directory}/{self.model_type}_model.keras'
            self.model.save(filename)
        else:
            filename = f'{model_directory}/{self.model_type}_model.pkl'
            joblib.dump(self.model, filename)
        print(f"\n¡Model saved: '{filename}'!")
