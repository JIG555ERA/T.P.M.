import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of records
n = 6000

# Generate data
data = {
    'TSH': np.round(np.random.uniform(0.1, 100, n), 2),
    'T3': np.round(np.random.uniform(0.5, 10, n), 2),
    'T4': np.round(np.random.uniform(2, 15, n), 2),
    'AGE': np.random.randint(1, 100, n),
    'GENDER': np.random.choice(['Male', 'Female'], n),
    'ANTITHYROID_ANTIBODIES': np.random.choice(['Yes', 'No'], n),
    'HEART_RATE': np.random.randint(50, 150, n),
    'CHOLESTEROL': np.random.randint(100, 300, n),
    'BMI': np.round(np.random.uniform(10, 50, n), 1),
    'BP': np.random.randint(80, 180, n),
    'FAMILY_HISTORY': np.random.choice(['Yes', 'No'], n),
    'GOITER': np.random.choice(['Yes', 'No'], n),
    'GLUCOSE': np.random.randint(60, 200, n),
    'VIT_D': np.random.randint(10, 100, n),
    'SMOKING': np.random.choice(['Yes', 'No'], n),
    'THYROID_TYPE': np.random.choice(['Hypothyroid', 'Hyperthyroid', 'Normal'], n)
}

df = pd.DataFrame(data)

# Add MENSTRUAL column only for females, 'None' for males
df['MENSTRUAL'] = np.where(df['GENDER'] == 'Female',
                           np.random.choice(['Normal', 'Irregular', 'Menopause'], n),
                           'None')

print(df)
