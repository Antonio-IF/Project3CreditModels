import pandas as pd
import numpy as np
from typing import Tuple, List

class DataPreprocessor:
    @staticmethod
    def calculate_debt_to_income(income: float, debt: float) -> float:
        """Calculate debt to income ratio"""
        return debt / income if income != 0 else float('inf')
    
    @staticmethod
    def encode_categorical_features(df: pd.DataFrame, 
                                 categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables using one-hot encoding"""
        return pd.get_dummies(df, columns=categorical_columns)
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, 
                       columns: List[str], 
                       method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers using IQR or z-score method"""
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean = df_clean[
                    (df_clean[col] >= Q1 - 1.5 * IQR) & 
                    (df_clean[col] <= Q3 + 1.5 * IQR)
                ]
        
        return df_clean 