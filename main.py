from model import CreditModel

if __name__ == "__main__":
    data_path = 'data/data.xlsx'
    # Model options: 'logistic_regression', 'random_forest', 'svm', 'xgboost', 'neural_network'
    model_type = 'neural_network'
    credit_model = CreditModel(data_path, model_type)
    #credit_model.eda()
    #credit_model.dqr()
    #credit_model.feature_engineering()
    credit_model.prepare_data()
    credit_model.build_model()

    if model_type == 'neural_network':
        history = credit_model.train_model(epochs=30, batch_size=32)
    else:
        credit_model.train_model()

    credit_model.evaluate_model()
    credit_model.save_model()
