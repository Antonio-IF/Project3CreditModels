from models.credit_model import CreditModel
from utils.data_preprocessing import DataPreprocessor
from config.settings import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the credit model"""
    try:
        # Initialize model
        credit_model = CreditModel()
        
        # Load and preprocess data
        logger.info("Loading data...")
        credit_model.load_data(DATA_PATH)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Handle outliers in numerical features
        logger.info("Handling outliers...")
        credit_model.data = preprocessor.handle_outliers(
            credit_model.data, 
            NUMERICAL_FEATURES
        )
        
        # Encode categorical features
        logger.info("Encoding categorical features...")
        credit_model.data = preprocessor.encode_categorical_features(
            credit_model.data, 
            CATEGORICAL_FEATURES
        )
        
        # Preprocess data
        logger.info("Preprocessing data...")
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        credit_model.preprocess_data(TARGET_COLUMN, all_features)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()