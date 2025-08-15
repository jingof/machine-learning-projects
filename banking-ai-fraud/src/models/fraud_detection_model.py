# src/models/fraud_detection_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow import keras
import joblib
import logging
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AutoEncoder:
    """Deep Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.threshold = None
        
    def build_model(self):
        """Build the autoencoder architecture"""
        
        # Encoder
        input_layer = keras.layers.Input(shape=(self.input_dim,))
        encoded = keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = keras.layers.BatchNormalization()(encoded)
        encoded = keras.layers.Dropout(0.2)(encoded)
        
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        encoded = keras.layers.BatchNormalization()(encoded)
        encoded = keras.layers.Dropout(0.2)(encoded)
        
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(64, activation='relu')(encoded)
        decoded = keras.layers.BatchNormalization()(decoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        
        decoded = keras.layers.Dense(128, activation='relu')(decoded)
        decoded = keras.layers.BatchNormalization()(decoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        
        decoded = keras.layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create model
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse')
        
    def fit(self, X_train: np.ndarray, validation_split: float = 0.2):
        """Train the autoencoder"""
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, X_train,
            epochs=100,
            batch_size=256,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        # Calculate reconstruction errors for threshold
        reconstructions = self.model.predict(X_train)
        reconstruction_errors = np.mean(np.square(X_train - reconstructions), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies based on reconstruction error"""
        
        reconstructions = self.model.predict(X)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Anomaly score (higher = more anomalous)
        anomaly_scores = reconstruction_errors / self.threshold
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions, anomaly_scores

class FraudDetectionModel:
    """
    Advanced Real-Time Fraud Detection System
    Uses ensemble of anomaly detection algorithms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.threshold = config.get('threshold', 0.1)
        
    def create_fraud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for fraud detection"""
        
        # Transaction amount features
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        df['is_large_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_small_amount'] = (df['amount'] < df['amount'].quantile(0.05)).astype(int)
        
        # Time-based features
        df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek.isin([5, 6]).astype(int)
        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
        
        # Velocity features (transactions per time window)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        def calc_last_hour_features(g):
            # For each timestamp in this group
            tx_counts = []
            amt_sums = []
            for t in g['timestamp']:
                mask = (t - g['timestamp']) <= pd.Timedelta(hours=1)
                tx_counts.append(mask.sum())
                amt_sums.append(g.loc[mask, 'amount'].sum())
            
            g['transactions_last_hour'] = tx_counts
            g['amount_last_hour'] = amt_sums
            return g

        df = df.groupby('customer_id', group_keys=False).apply(calc_last_hour_features)

        # df['transactions_last_hour'] = df.groupby('customer_id').apply(
        #     lambda g: g['timestamp'].apply(
        #         lambda t: ((t - g['timestamp']) <= pd.Timedelta(hours=1)).sum()
        #     )
        # ).reset_index(level=0, drop=True)

        # df['amount_last_hour'] = df.groupby('customer_id').apply(
        #     lambda g: g['timestamp'].apply(
        #         lambda t: g.loc[(t - g['timestamp']) <= pd.Timedelta(hours=1), 'amount'].sum()
        #     )
        # ).reset_index(level=0, drop=True)


        # Location-based features
        if 'merchant_category' in df.columns:
            df['merchant_risk_score'] = df['merchant_category'].map(
                df.groupby('merchant_category')['amount'].mean()
            )
        
        # Account age and behavior
        df['account_age_days'] = (pd.to_datetime(df['timestamp']) - 
                                 pd.to_datetime(df['account_created'])).dt.days
        df['is_new_account'] = (df['account_age_days'] < 30).astype(int)
        
        # Payment method risk
        if 'payment_method' in df.columns:
            payment_risk = {'credit_card': 0.3, 'debit_card': 0.1, 
                          'bank_transfer': 0.05, 'digital_wallet': 0.2}
            df['payment_risk'] = df['payment_method'].map(payment_risk).fillna(0.15)
        
        # Customer behavior patterns
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'max'],
            'transaction_hour': 'std'
        }).fillna(0)
        
        customer_stats.columns = ['avg_amount', 'amount_std', 'max_amount', 'hour_std']
        df = df.merge(customer_stats, left_on='customer_id', right_index=True, how='left')
        
        # Deviation from normal behavior
        df['amount_deviation'] = np.abs(df['amount'] - df['avg_amount']) / (df['amount_std'] + 1)
        df['is_amount_outlier'] = (df['amount_deviation'] > 3).astype(int)
        
        # merchant category
        # ['gas', 'healthcare', 'utilities', 'restaurant', 'grocery', 'other', 'online', 'retail', 'travel', 'entertainment']
        df['merchant_category_gas'] = df['merchant_category'].str.contains('gas').astype(int)
        df['merchant_category_healthcare'] = df['merchant_category'].str.contains('healthcare').astype(int)
        df['merchant_category_utilities'] = df['merchant_category'].str.contains('utilities').astype(int)
        df['merchant_category_restaurant'] = df['merchant_category'].str.contains('restaurant').astype(int)
        df['merchant_category_grocery'] = df['merchant_category'].str.contains('grocery').astype(int)
        df['merchant_category_online'] = df['merchant_category'].str.contains('online').astype(int)
        df['merchant_category_retail'] = df['merchant_category'].str.contains('retail').astype(int)
        df['merchant_category_travel'] = df['merchant_category'].str.contains('travel').astype(int)
        df['merchant_category_entertainment'] = df['merchant_category'].str.contains('entertainment').astype(int)
        df.drop('merchant_category', axis=1, inplace=True)

        # payment method
        # ['credit_card', 'bank_transfer', 'debit_card', 'digital_wallet']
        df['payment_method_credit_card'] = df['payment_method'].str.contains('credit_card').astype(int)
        df['payment_method_bank_transfer'] = df['payment_method'].str.contains('bank_transfer').astype(int)
        df['payment_method_debit_card'] = df['payment_method'].str.contains('debit_card').astype(int)
        df.drop('payment_method', axis=1, inplace=True)

        # account created
        df['account_age'] = (pd.to_datetime('now') - pd.to_datetime(df['account_created'], format = '%Y-%m-%d')).dt.days
        df.drop('account_created', axis=1, inplace=True)
        df.drop('customer_id', axis=1, inplace=True)


        # timestamp date features
        df['timestamp'] = pd.to_datetime(df['timestamp'], format = '%Y-%m-%d')
        df['timestamp_year'] = df['timestamp'].dt.year
        df['timestamp_month'] = df['timestamp'].dt.month
        df['timestamp_day'] = df['timestamp'].dt.day
        df['timestamp_hour'] = df['timestamp'].dt.hour
        df.drop('timestamp', axis=1, inplace=True)
        
        
        return df
    
    def build_isolation_forest(self, X_train: np.ndarray) -> IsolationForest:
        """Build Isolation Forest model"""
        
        model = IsolationForest(
            contamination=self.threshold,
            random_state=42,
            n_jobs=-1,
            n_estimators=100
        )
        
        model.fit(X_train)
        return model
    
    def build_one_class_svm(self, X_train: np.ndarray) -> OneClassSVM:
        """Build One-Class SVM model"""
        
        model = OneClassSVM(
            nu=self.threshold,
            kernel='rbf',
            gamma='scale'
        )
        
        model.fit(X_train)
        return model
    
    def build_autoencoder(self, X_train: np.ndarray) -> AutoEncoder:
        """Build Autoencoder model"""
        
        input_dim = X_train.shape[1]
        encoding_dim = max(8, input_dim // 4)
        
        model = AutoEncoder(input_dim, encoding_dim)
        model.fit(X_train)
        
        return model
    
    def train(self, X_train: pd.DataFrame) -> Dict[str, Any]:
        """Train the fraud detection ensemble"""
        
        logger.info("Starting fraud detection model training...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Create fraud-specific features
        X_train_processed = self.create_fraud_features(X_train.copy())
        
        # Prepare different scalers for different algorithms
        # StandardScaler for autoencoder and SVM
        self.scalers['standard'] = StandardScaler()
        X_train_standard = self.scalers['standard'].fit_transform(X_train_processed)
        
        # RobustScaler for Isolation Forest
        self.scalers['robust'] = RobustScaler()
        X_train_robust = self.scalers['robust'].fit_transform(X_train_processed)
        
        # Train individual models
        logger.info("Training Isolation Forest...")
        self.models['isolation_forest'] = self.build_isolation_forest(X_train_robust)
        
        logger.info("Training One-Class SVM...")
        self.models['one_class_svm'] = self.build_one_class_svm(X_train_standard)
        
        logger.info("Training Autoencoder...")
        self.models['autoencoder'] = self.build_autoencoder(X_train_standard)
        
        # Evaluate on training data (for baseline)
        predictions, scores = self.predict(X_train_processed)
        
        fraud_rate = np.mean(predictions)
        avg_score = np.mean(scores)
        
        logger.info(f"Fraud detection model training completed.")
        logger.info(f"Detected fraud rate: {fraud_rate:.4f}, Average anomaly score: {avg_score:.4f}")
        
        return {
            "fraud_rate": fraud_rate,
            "average_score": avg_score,
            "models_trained": list(self.models.keys())
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make fraud predictions using ensemble approach"""
        
        # Process features
        # X_processed = self.create_fraud_features(X.copy())
        X_processed = X.copy()
        
        # Scale data for different models
        X_standard = self.scalers['standard'].transform(X_processed)
        X_robust = self.scalers['robust'].transform(X_processed)
        
        predictions_list = []
        scores_list = []
        
        # Isolation Forest predictions
        if_pred = self.models['isolation_forest'].predict(X_robust)
        if_pred = (if_pred == -1).astype(int)  # Convert to 0/1
        if_scores = self.models['isolation_forest'].score_samples(X_robust)
        if_scores = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        
        predictions_list.append(if_pred)
        scores_list.append(if_scores)
        
        # One-Class SVM predictions
        svm_pred = self.models['one_class_svm'].predict(X_standard)
        svm_pred = (svm_pred == -1).astype(int)  # Convert to 0/1
        svm_scores = self.models['one_class_svm'].score_samples(X_standard)
        svm_scores = 1 - (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
        
        predictions_list.append(svm_pred)
        scores_list.append(svm_scores)
        
        # Autoencoder predictions
        ae_pred, ae_scores = self.models['autoencoder'].predict(X_standard)
        predictions_list.append(ae_pred)
        scores_list.append(ae_scores)
        
        # Ensemble voting (majority vote for predictions)
        ensemble_pred = np.array(predictions_list).T
        final_predictions = (np.sum(ensemble_pred, axis=1) >= 2).astype(int)
        
        # Average scores
        ensemble_scores = np.array(scores_list).T
        final_scores = np.mean(ensemble_scores, axis=1)
        
        return final_predictions, final_scores
    
    def predict_realtime(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real-time fraud prediction for a single transaction"""
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Make prediction
        prediction, score = self.predict(df)
        
        # Determine risk level
        if score[0] > 0.8:
            risk_level = "HIGH"
        elif score[0] > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        result = {
            "transaction_id": transaction_data.get("transaction_id", "unknown"),
            "is_fraud": bool(prediction[0]),
            "fraud_score": float(score[0]),
            "risk_level": risk_level,
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_version": "v1.0"
        }
        
        return result
    
    def get_feature_contributions(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, float]:
        """Get feature contributions to fraud score"""
        
        X_processed = self.create_fraud_features(X.copy())
        sample = X_processed.iloc[sample_idx:sample_idx+1]
        
        # Get baseline prediction
        _, baseline_score = self.predict(sample)
        
        contributions = {}
        
        # Calculate feature importance by perturbation
        for feature in X_processed.columns:
            # Create perturbed sample (set feature to mean)
            perturbed_sample = sample.copy()
            perturbed_sample[feature] = X_processed[feature].mean()
            
            _, perturbed_score = self.predict(perturbed_sample)
            
            # Contribution is the difference in scores
            contributions[feature] = float(baseline_score[0] - perturbed_score[0])
        
        return contributions
    
    def save_model(self, path: str):
        """Save the trained fraud detection model"""
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Fraud detection model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained fraud detection model"""
        
        model_data = joblib.load(path)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
        self.config = model_data['config']
        
        logger.info(f"Fraud detection model loaded from {path}")
    
    def update_model_realtime(self, new_data: pd.DataFrame, feedback: np.ndarray):
        """Update model with new data and feedback (online learning simulation)"""
        
        logger.info("Updating fraud detection model with new data...")
        
        # For Isolation Forest, retrain with new data
        X_processed = self.create_fraud_features(new_data.copy())
        X_robust = self.scalers['robust'].transform(X_processed)
        
        # Simple retraining approach (in production, use incremental learning)
        self.models['isolation_forest'] = self.build_isolation_forest(X_robust)
        
        logger.info("Model updated successfully")

class FraudMonitoring:
    """Real-time monitoring and alerting for fraud detection"""
    
    def __init__(self, model: FraudDetectionModel):
        self.model = model
        self.alerts = []
        self.statistics = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'high_risk_transactions': 0
        }
    
    def process_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transaction and generate alerts if needed"""
        
        result = self.model.predict_realtime(transaction_data)
        
        # Update statistics
        self.statistics['total_transactions'] += 1
        if result['is_fraud']:
            self.statistics['fraud_detected'] += 1
        if result['risk_level'] == 'HIGH':
            self.statistics['high_risk_transactions'] += 1
        
        # Generate alerts for high-risk transactions
        if result['fraud_score'] > 0.8:
            alert = {
                'timestamp': result['timestamp'],
                'transaction_id': result['transaction_id'],
                'alert_type': 'HIGH_FRAUD_RISK',
                'fraud_score': result['fraud_score'],
                'customer_id': transaction_data.get('customer_id'),
                'amount': transaction_data.get('amount'),
                'message': f"High fraud risk detected (score: {result['fraud_score']:.3f})"
            }
            self.alerts.append(alert)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current fraud detection statistics"""
        
        stats = self.statistics.copy()
        if stats['total_transactions'] > 0:
            stats['fraud_rate'] = stats['fraud_detected'] / stats['total_transactions']
            stats['high_risk_rate'] = stats['high_risk_transactions'] / stats['total_transactions']
        else:
            stats['fraud_rate'] = 0
            stats['high_risk_rate'] = 0
        
        return stats
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent fraud alerts"""
        
        return self.alerts[-limit:] if len(self.alerts) >= limit else self.alerts

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the model
    pass
