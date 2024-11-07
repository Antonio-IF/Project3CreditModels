# Model configuration
MODEL_PARAMS = {
    'random_state': 42,
    'test_size': 0.2
}

# Feature columns
NUMERICAL_FEATURES = [
    'monthly_income',
    'age',
    'years_employed',
    'debt_to_income_ratio'
]

CATEGORICAL_FEATURES = [
    'education_level',
    'employment_sector',
    'housing_type'
]

# Target column
TARGET_COLUMN = 'credit_approved'

# File paths
DATA_PATH = 'data/credit_data.csv'
MODEL_SAVE_PATH = 'models/saved_models/' 