"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: dqr.py : python script with dqr functionality                                               -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/Antonio-IF/Project3CreditModels                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------

class DQR:

    # ------------------------------

    def __init__(self, data):

        self.validate_data(data)
        self.data = data

    # ------------------------------

    def validate_data(self, data):

        if data.empty:
            raise ValueError("DataFrame must be not empty.")

    # ------------------------------

    def preprocess_data(self):

        try:
            self.data = self.data.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan'])

        except KeyError as e:
            print(f"Error dropping columns: {e}")

    # ------------------------------

    def clean_incoherent_values(self):

        try:
            replacements = {
                'Occupation': {'_______': np.nan},
                'Credit_Mix': {'_': np.nan},
                'Payment_Behaviour': {'!@9#%8': np.nan}
            }
            self.data.replace(replacements, inplace=True)

            for key in replacements.keys():
                self.data[key] = self.data[key].fillna(self.data[key].mode().iloc[0])

        except Exception as e:
            print(f"Error cleaning incoherent values: {e}")

    # ------------------------------

    def convert_time_to_numeric(self):

        try:
            if self.data['Credit_History_Age'].dtype != 'object':
                self.data['Credit_History_Age'] = self.data['Credit_History_Age'].astype(str)    

            years = self.data['Credit_History_Age'].str.extract(r'(\d+)\sYears').astype(float)
            months = self.data['Credit_History_Age'].str.extract(r'(\d+)\sMonths').astype(float)
            
            self.data['Credit_History_Age'] = years + (months / 12)

        except Exception as e:
            print(f"Error converting time to numeric: {e}")

    # ------------------------------

    def clean_data_types(self):

        try:
            object_to_numeric_columns = ['Month', 'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']

            non_numeric_columns = self.data.columns[~self.data.columns.isin(object_to_numeric_columns)]

            for column in non_numeric_columns:
                if self.data[column].dtype == 'object':
                    self.data[column] = self.data[column].str.replace('_', '')
                    self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

        except Exception as e:
            print(f"Error cleaning data types: {e}")

    # ------------------------------

    def handle_outliers(self):

        try:
            for col in self.data.select_dtypes(include=['number']).columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)

        except Exception as e:
            print(f"Error handling outliers: {e}")

    # ------------------------------

    def handle_missing_values(self):

        try:
            for column in self.data.columns:
                if self.data[column].dtype in ['float64', 'int64']:
                    self.data[column] = self.data[column].fillna(self.data[column].median())

                elif self.data[column].dtype == 'object':
                    self.data[column] = self.data[column].fillna(self.data[column].mode().iloc[0])

        except Exception as e:
            print(f"Error handling missing values: {e}")
            
    # ------------------------------

    def encode_categorical(self):

        try:
            encoder = LabelEncoder()
            for column in self.data.select_dtypes(include=['object']).columns:
                self.data[column] = encoder.fit_transform(self.data[column].astype(str))

        except Exception as e:
            print(f"Error encoding categorical variables: {e}")

    # ------------------------------

    def replace_negatives(self):

        columns_to_check = [
            "Age",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Num_of_Loan",
            "Delayd_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Monthly_Balance"
        ]

        try:
            for col in columns_to_check:
                if col in self.data.columns:
                    median_value = self.data[col].median()
                    self.data[col] = self.data[col].where(self.data[col] >= 0, median_value)

        except Exception as e:
            print(f"Error replacing negative values: {e}")

    # ------------------------------

    def postprocess_data(self):

        try:
            int_columns = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
                        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']
            self.data[int_columns] = self.data[int_columns].astype('int')

            self.data = self.data.drop_duplicates()

        except Exception as e:
            print(f"Error during postprocessing: {e}")

    # ------------------------------

    def perform_clean(self):

        self.preprocess_data()
        self.clean_incoherent_values()
        self.convert_time_to_numeric()
        self.clean_data_types()
        self.handle_outliers()
        self.handle_missing_values()
        self.encode_categorical()
        self.replace_negatives()
        self.postprocess_data()

        output_path = 'data/clean_data.xlsx'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.data.to_excel(output_path, index=False)

        print(f"Cleaned data saved to {output_path}")

        return self.data
