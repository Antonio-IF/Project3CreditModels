from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

class CreditModel:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, file_path: str) -> None:
        """Load data from CSV file"""
        self.data = pd.read_csv(file_path)
        
    def preprocess_data(self, target_column: str, features: list) -> None:
        """
        Preprocess data including:
        - Handle missing values
        - Scale numerical features
        - Encode categorical variables
        """
        X = self.data[features]
        y = self.data[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale numerical features
        X = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        ) 