import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
from credit_risk_model import CreditRiskModel
from fraud_detection_model import FraudDetectionModel
from config.config import Config
import pandas as pd
os.system("python config/config.py")

os.system("""python src/data_preprocessing/data_generator.py \
    --customers 5000 \
    --credit-apps 10000 \
    --transactions 100000 \
    --output-dir ./data/synthetic""")

# Train credit risk model
# Load data
credit_data = pd.read_csv('./data/synthetic/credit_applications.csv')
X = credit_data.drop(['is_default', 'customer_id'], axis=1)
y = credit_data['is_default']

# Train model
model = CreditRiskModel(Config.MODEL_CONFIG['credit_risk'])
results = model.train(X, y)
model.save_model('./src/models/credit_risk_model.pkl')
print(f'Credit risk model trained with AUC: {results["auc_score"]:.3f}')

# Train fraud detection model
# Load data
transaction_data = pd.read_csv('./data/synthetic/transactions.csv')
X = transaction_data.drop(['is_fraud', 'transaction_id', 'merchant_city', 'merchant_state'], axis=1)

# Train model
model = FraudDetectionModel(Config.MODEL_CONFIG['fraud_detection'])
results = model.train(X)
model.save_model('./src/models/fraud_detection_model.pkl')
print('Fraud detection model trained successfully')
