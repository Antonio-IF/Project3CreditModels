# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: main.py : python script with main functionality                                             -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/Antonio-IF/Project3CreditModels                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #

from model import CreditModelWithStacking

# --------------------------------------------------

if __name__ == "__main__":
    data_path = 'data/data.xlsx'
    selected_columns = ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento', 'ESTATUS']
    
    # Initialize stacking model
    credit_model = CreditModelWithStacking(data_path, model_type='stacking', selected_columns=selected_columns)
    
    # Perform Exploratory Data Analysis (EDA)
    credit_model.eda(data_path)
    
    # Prepare data
    credit_model.prepare_data()
    
    # Build stacking model
    credit_model.build_model()
    
    # Perform hyperparameter tuning for the stacking model
    param_grid = {
        'stack_method': ['auto', 'predict_proba'],  # Stacking method options
        'final_estimator__C': [0.01, 0.1, 1, 10],  # Regularization strength for Logistic Regression
        'xgboost__n_estimators': [50, 100, 200],  # Number of trees for XGBoost
        'xgboost__max_depth': [3, 5, 7],  # Depth of trees for XGBoost
    }
    credit_model.hyperparameter_tuning(param_grid=param_grid)
    
    # Train stacking model
    credit_model.train_model()
    
    # Evaluate stacking model
    credit_model.evaluate_model()
    
    # Save the trained stacking model
    credit_model.save_model(model_directory='stacking_models')
