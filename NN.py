import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class NeuralNetworkCreditModel(CreditModel):
    def __init__(self, data_path):
        super().__init__(data_path, model_type='neural_network')
        self.model = None

    def build_model(self, input_dim):
        """
        Define la arquitectura de la red neuronal.
        """
        self.model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Salida binaria
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        print("Red Neuronal Construida:")
        self.model.summary()

    def train_model(self, epochs=50, batch_size=32):
        """
        Entrenar la red neuronal.
        """
        self.build_model(input_dim=self.X_train.shape[1])
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
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
