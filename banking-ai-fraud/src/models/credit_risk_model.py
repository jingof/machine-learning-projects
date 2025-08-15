# src/models/credit_risk_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
import shap
import joblib
import logging
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CreditRiskModel:
    """
    Advanced Credit Risk Assessment Model using Ensemble Methods
    Combines XGBoost, LightGBM, and Neural Network for superior performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.meta_model = None
        self.feature_importance = {}
        self.explainer = None
        self.scaler = None
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for credit risk assessment"""
        
        # Create debt-to-income ratios
        df['debt_to_income'] = df['total_debt'] / (df['annual_income'] + 1)
        df['credit_utilization'] = df['total_credit_used'] / (df['total_credit_limit'] + 1)
        
        # Employment stability features
        df['employment_stability'] = (df['employment_years'] > 2).astype(int)
        df['income_stability'] = (df['annual_income'] > df['annual_income'].median()).astype(int)
        
        # Payment history features
        df['late_payments_ratio'] = df['late_payments'] / (df['total_payments'] + 1)
        df['avg_payment_amount'] = df['total_payments_amount'] / (df['total_payments'] + 1)
        
        # Credit history features
        df['credit_age_years'] = df['credit_history_months'] / 12
        df['accounts_per_year'] = df['number_of_accounts'] / (df['credit_age_years'] + 1)
        
        # Behavioral features
        df['recent_inquiries_risk'] = (df['recent_credit_inquiries'] > 3).astype(int)
        df['high_utilization_risk'] = (df['credit_utilization'] > 0.8).astype(int)
        
        # Interaction features
        df['income_debt_interaction'] = df['annual_income'] * df['debt_to_income']
        df['age_income_interaction'] = df['age'] * np.log1p(df['annual_income'])
        
        # Application date features
        df['application_date'] = pd.to_datetime(df['application_date'], format = '%Y-%m-%d')
        df['application_date_year'] = df['application_date'].dt.year
        df['application_date_month'] = df['application_date'].dt.month
        df['application_date_day'] = df['application_date'].dt.day
        df.drop('application_date', axis=1, inplace=True)

        # Education level features
        # ['phd', 'masters', 'bachelors', 'high_school']
        df['education_level_phd'] = df['education_level'].str.contains('phd').astype(int)
        df['education_level_masters'] = df['education_level'].str.contains('masters').astype(int)
        df['education_level_bachelors'] = df['education_level'].str.contains('bachelors').astype(int)
        df.drop('education_level', axis=1, inplace=True)

        # Marital status features
        # ['single', 'married', 'divorced']
        df['marital_status_married'] = df['marital_status'].str.contains('married').astype(int)
        df['marital_status_divorced'] = df['marital_status'].str.contains('divorced').astype(int)
        df.drop('marital_status', axis=1, inplace=True)

        # Home ownership features
        # ['rent', 'mortgage', 'own']
        df['home_ownership_mortgage'] = df['home_ownership'].str.contains('mortgage').astype(int)
        df['home_ownership_own'] = df['home_ownership'].str.contains('own').astype(int)
        df.drop('home_ownership', axis=1, inplace=True)

        # Loan purpose features 
        # ['debt_consolidation', 'home_improvement', 'home_purchase', 'refinance', 'other']
        df['loan_purpose_debt_consolidation'] = df['loan_purpose'].str.contains('debt_consolidation').astype(int)
        df['loan_purpose_home_improvement'] = df['loan_purpose'].str.contains('home_improvement').astype(int)
        df['loan_purpose_home_purchase'] = df['loan_purpose'].str.contains('home_purchase').astype(int)
        df['loan_purpose_refinance'] = df['loan_purpose'].str.contains('refinance').astype(int)
        df.drop('loan_purpose', axis=1, inplace=True)
        return df
    

    def build_xgboost_model(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Build and train XGBoost model"""
        
        xgb_params = {
            'n_estimators': self.config['hyperparameters']['xgb_n_estimators'],
            'learning_rate': self.config['hyperparameters']['xgb_learning_rate'],
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        
        return model
    
    def build_lightgbm_model(self, X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
        """Build and train LightGBM model"""
        
        lgb_params = {
            'n_estimators': self.config['hyperparameters']['lgb_n_estimators'],
            'learning_rate': self.config['hyperparameters']['lgb_learning_rate'],
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)
        
        return model
    
    def build_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
        """Build and train Neural Network model"""
        
        input_dim = X_train.shape[1]
        hidden_layers = self.config['hyperparameters']['nn_hidden_layers']
        dropout_rate = self.config['hyperparameters']['nn_dropout']
        
        model = keras.Sequential([
            keras.layers.Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            
            keras.layers.Dense(hidden_layers[1], activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            
            keras.layers.Dense(hidden_layers[2], activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.Precision(),
                     keras.metrics.Recall() ]
        )
        
        # Train with early stopping and learning rate reduction
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=256,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the ensemble credit risk model"""
        
        logger.info("Starting credit risk model training...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Prepare features
        X_train_processed = self.prepare_features(X_train.copy())
        
        # Scale features for neural network
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        
        # Train individual models
        logger.info("Training XGBoost model...")
        self.models['xgboost'] = self.build_xgboost_model(X_train_scaled, y_train)
        
        logger.info("Training LightGBM model...")
        self.models['lightgbm'] = self.build_lightgbm_model(X_train_scaled, y_train)
        
        logger.info("Training Neural Network...")
        self.models['neural_network'] = self.build_neural_network(X_train_scaled, y_train)
        
        # Create ensemble predictions for meta-model
        train_predictions = self._get_ensemble_predictions(X_train_scaled)
        
        # Check for NaNs using np.isnan()
        if np.isnan(train_predictions).any():
            print("NaN values found in ensemble predictions. Imputing with mean...")
            
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            train_predictions = imputer.fit_transform(train_predictions)

        # Train meta-model (Logistic Regression)
        from sklearn.linear_model import LogisticRegression
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(train_predictions, y_train)
        
        # Calculate ensemble performance
        ensemble_pred = self.meta_model.predict_proba(train_predictions)[:, 1]
        auc_score = roc_auc_score(y_train, ensemble_pred)
        
        # Setup SHAP explainer
        self.explainer = shap.TreeExplainer(self.models['xgboost'])
        
        logger.info(f"Credit risk model training completed. AUC Score: {auc_score:.4f}")
        
        return {"auc_score": auc_score}
    
    def _get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all models for ensemble"""
        
        predictions = []
        
        # XGBoost predictions
        xgb_pred = self.models['xgboost'].predict_proba(X)[:, 1]
        predictions.append(xgb_pred)
        
        # LightGBM predictions
        lgb_pred = self.models['lightgbm'].predict_proba(X)[:, 1]
        predictions.append(lgb_pred)
        
        # Neural Network predictions
        nn_pred = self.models['neural_network'].predict(X).flatten()
        predictions.append(nn_pred)
        
        return np.column_stack(predictions)
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the ensemble model"""
        
        # Prepare features
        X_processed = self.prepare_features(X.copy())
        X_scaled = self.scaler.transform(X_processed)
        
        # Get ensemble predictions
        ensemble_pred = self._get_ensemble_predictions(X_scaled)
        
        # Meta-model predictions
        probabilities = self.meta_model.predict_proba(ensemble_pred)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain individual prediction using SHAP"""
        
        X_processed = self.prepare_features(X.copy())
        X_scaled = self.scaler.transform(X_processed)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled[sample_idx:sample_idx+1])
        
        # Get prediction
        _, prob = self.predict(X.iloc[sample_idx:sample_idx+1])
        
        explanation = {
            "prediction_probability": prob[0],
            "shap_values": shap_values[0].tolist(),
            "feature_names": self.feature_names,
            "base_value": self.explainer.expected_value
        }
        
        return explanation
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from ensemble"""
        
        if not self.feature_importance:
            # XGBoost importance
            xgb_importance = dict(zip(self.feature_names, 
                                    self.models['xgboost'].feature_importances_))
            
            # LightGBM importance
            lgb_importance = dict(zip(self.feature_names, 
                                    self.models['lightgbm'].feature_importances_))
            
            # Average importance
            all_features = set(xgb_importance.keys()) | set(lgb_importance.keys())
            self.feature_importance = {
                feature: (xgb_importance.get(feature, 0) + 
                         lgb_importance.get(feature, 0)) / 2
                for feature in all_features
            }
        
        return self.feature_importance
    
    def save_model(self, path: str):
        """Save the trained model"""
        
        model_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Credit risk model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        
        model_data = joblib.load(path)
        
        self.models = model_data['models']
        self.meta_model = model_data['meta_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        
        # Recreate SHAP explainer
        if 'xgboost' in self.models:
            self.explainer = shap.TreeExplainer(self.models['xgboost'])
        
        logger.info(f"Credit risk model loaded from {path}")

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the model
    pass
