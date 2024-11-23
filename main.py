from model import CreditModel
from model import NeuralNetworkCreditModel


#def main():
#    model_path = 'data/data_10.xlsx'
    # model_types = ['logistic_regression']
    # model_types = ['random_forest']
    # model_types = ['svm']
#    model_types = ['neural_network']
#
#    for model_type in model_types:
#        print(f"\n{'='*50}\nTraining and evaluating model type: {model_type}\n{'='*50}")
#        credit_model = CreditModel(model_path, model_type)
#
#        #credit_model.eda()
#        #credit_model.dqr()
#        #credit_model.feature_engineering()
#        credit_model.prepare_data()
#        credit_model.train_model()
#        credit_model.evaluate_model()
#        credit_model.save_model()

#if __name__ == "__main__":
#    main()

if __name__ == "__main__":
    # Ruta al archivo de datos
    data_path = 'data/data_10.xlsx'
    
    # Crear instancia del modelo de red neuronal
    nn_model = NeuralNetworkCreditModel(data_path)
    
    # Realizar EDA y DQR
    #nn_model.eda()
    #nn_model.dqr()
    
    # Feature Engineering y preparación de datos
    nn_model.feature_engineering()
    nn_model.prepare_data()
    
    # Entrenar y evaluar el modelo
    history = nn_model.train_model(epochs=30, batch_size=32)
    nn_model.evaluate_model()
    
    # Guardar el modelo
    nn_model.save_model()
