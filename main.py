"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: main.py : python script with main functionality                                             -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/Antonio-IF/Project3CreditModels                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from model import CreditModel

# --------------------------------------------------

if __name__ == "__main__":
    data_path = 'data/data.xlsx'
    selected_columns = ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento', 'ESTATUS']
    # Model options: 'logistic_regression', 'random_forest', 'svm', 'xgboost', 'neural_network'
    model_type = 'random_forest'
    credit_model = CreditModel(data_path, model_type, selected_columns)
    credit_model.eda(data_path)
    credit_model.prepare_data()
    credit_model.build_model()

    if model_type == 'neural_network':
        history = credit_model.train_model(epochs=30, batch_size=32)
    else:
        credit_model.train_model()

    credit_model.evaluate_model()
    credit_model.save_model()