from model import CreditModel

def main():
    model_path = 'data/data_10.xlsx'
    # model_types = ['logistic_regression']
    # model_types = ['random_forest']
    # model_types = ['svm']
    model_types = ['xgboost']

    for model_type in model_types:
        print(f"\n{'='*50}\nTraining and evaluating model type: {model_type}\n{'='*50}")
        credit_model = CreditModel(model_path, model_type)

        #credit_model.eda()
        #credit_model.dqr()
        #credit_model.feature_engineering()
        credit_model.prepare_data()
        credit_model.train_model()
        credit_model.evaluate_model()
        credit_model.save_model()

if __name__ == "__main__":
    main()
