# src/data_preprocessing/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import logging
from faker import Faker

logger = logging.getLogger(__name__)
fake = Faker()

class BankingDataGenerator:
    """Generate realistic synthetic banking data for model training and testing"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        Faker.seed(seed)
        self.fake = Faker()
    
    def generate_credit_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic credit application data"""
        
        logger.info(f"Generating {n_samples} credit application records...")
        
        # Base demographic and financial data
        data = {
            'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
            'application_date': [self.fake.date_between(start_date='-2y', end_date='today') 
                               for _ in range(n_samples)],
            'age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
            'annual_income': np.random.lognormal(10.8, 0.6, n_samples),  # Realistic income distribution
            'employment_years': np.random.exponential(7, n_samples).clip(0, 40),
            'education_level': np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], 
                                              n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'marital_status': np.random.choice(['single', 'married', 'divorced'], 
                                             n_samples, p=[0.35, 0.5, 0.15]),
            'dependents': np.random.poisson(1.2, n_samples).clip(0, 6),
            'home_ownership': np.random.choice(['own', 'rent', 'mortgage'], 
                                             n_samples, p=[0.3, 0.25, 0.45])
        }
        
        df = pd.DataFrame(data)
        
        # Credit-related features
        df['credit_score'] = np.random.normal(700, 100, n_samples).clip(300, 850).astype(int)
        
        # Loan details
        df['loan_amount'] = np.random.lognormal(11.5, 0.8, n_samples)
        df['loan_purpose'] = np.random.choice(
            ['home_purchase', 'refinance', 'debt_consolidation', 'home_improvement', 'other'],
            n_samples, p=[0.35, 0.25, 0.2, 0.1, 0.1]
        )
        df['loan_term'] = np.random.choice([15, 20, 30], n_samples, p=[0.1, 0.2, 0.7])
        
        # Existing debt information
        df['total_debt'] = df['annual_income'] * np.random.beta(1, 4, n_samples) * 0.5
        df['total_credit_limit'] = df['annual_income'] * np.random.gamma(2, 0.3, n_samples)
        df['total_credit_used'] = df['total_credit_limit'] * np.random.beta(2, 3, n_samples)
        
        # Payment history
        df['total_payments'] = np.random.poisson(50, n_samples) + 10
        df['late_payments'] = np.random.poisson(2, n_samples).clip(0, df['total_payments'] * 0.3)
        df['total_payments_amount'] = df['total_payments'] * np.random.lognormal(6, 0.5, n_samples)
        
        # Credit history
        df['credit_history_months'] = np.random.exponential(60, n_samples).clip(6, 360).astype(int)
        df['number_of_accounts'] = np.random.poisson(8, n_samples) + 1
        df['recent_credit_inquiries'] = np.random.poisson(1.5, n_samples).clip(0, 10)
        
        # Property information (for home loans)
        df['property_value'] = np.where(
            df['loan_purpose'].isin(['home_purchase', 'refinance']),
            df['loan_amount'] * np.random.uniform(1.2, 2.0, n_samples),
            np.nan
        )
        df['down_payment'] = np.where(
            df['loan_purpose'] == 'home_purchase',
            df['property_value'] * np.random.uniform(0.05, 0.25, n_samples),
            np.nan
        )
        
        # Create realistic relationships and correlations
        # Higher income -> better credit score
        income_effect = (df['annual_income'] - df['annual_income'].mean()) / df['annual_income'].std()
        df['credit_score'] += (income_effect * 30).clip(-100, 100).astype(int)
        df['credit_score'] = df['credit_score'].clip(300, 850)
        
        # Age effect on credit history
        df['credit_history_months'] = np.minimum(
            df['credit_history_months'], 
            (df['age'] - 18) * 12
        )
        
        # Employment stability effect
        stability_bonus = np.where(df['employment_years'] > 5, 20, 0)
        df['credit_score'] += stability_bonus
        df['credit_score'] = df['credit_score'].clip(300, 850)
        
        # Generate target variable (default probability)
        # Create risk score based on multiple factors
        risk_factors = (
            (df['total_debt'] / df['annual_income']).clip(0, 2) * 0.3 +
            (1 - df['credit_score'] / 850) * 0.25 +
            (df['late_payments'] / df['total_payments']).fillna(0) * 0.2 +
            (df['total_credit_used'] / df['total_credit_limit']).clip(0, 1) * 0.15 +
            np.random.normal(0, 0.1, n_samples) * 0.1  # Random noise
        ).clip(0, 1)
        
        df['default_probability'] = risk_factors
        df['is_default'] = np.random.binomial(1, risk_factors)
        
        logger.info(f"Generated credit data with {df['is_default'].mean():.2%} default rate")
        
        return df
    
    def generate_transaction_data(self, n_samples: int = 100000, 
                                n_customers: int = 5000) -> pd.DataFrame:
        """Generate synthetic transaction data for fraud detection"""
        
        logger.info(f"Generating {n_samples} transaction records for {n_customers} customers...")
        
        # Generate customer base
        customers = [f'CUST_{i:06d}' for i in range(n_customers)]
        
        # Transaction base data
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
            'customer_id': np.random.choice(customers, n_samples),
            'timestamp': [
                self.fake.date_time_between(start_date='-1y', end_date='now')
                for _ in range(n_samples)
            ],
            'amount': np.random.lognormal(4.5, 1.5, n_samples),
            'merchant_category': np.random.choice(
                ['grocery', 'gas', 'restaurant', 'retail', 'online', 'healthcare', 
                 'entertainment', 'travel', 'utilities', 'other'],
                n_samples,
                p=[0.25, 0.15, 0.15, 0.12, 0.1, 0.05, 0.05, 0.03, 0.05, 0.05]
            ),
            'payment_method': np.random.choice(
                ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'],
                n_samples,
                p=[0.45, 0.35, 0.15, 0.05]
            )
        }
        
        df = pd.DataFrame(data)
        
        # Add location data
        df['merchant_city'] = [self.fake.city() for _ in range(n_samples)]
        df['merchant_state'] = [self.fake.state_abbr() for _ in range(n_samples)]
        
        # Customer account information
        df['account_created'] = [
            self.fake.date_between(start_date='-5y', end_date='-1m')
            for _ in range(n_samples)
        ]
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_night'] = (df['hour'] < 6) | (df['hour'] > 22)
        
        # Customer behavior patterns
        customer_profiles = pd.DataFrame({
            'customer_id': customers,
            'avg_transaction_amount': np.random.lognormal(4, 0.8, n_customers),
            'transaction_frequency': np.random.gamma(2, 2, n_customers),
            'preferred_categories': np.random.choice(
                ['grocery', 'gas', 'restaurant', 'retail', 'online'],
                n_customers
            ),
            'risk_profile': np.random.choice(['low', 'medium', 'high'], n_customers, p=[0.8, 0.15, 0.05])
        })
        
        df = df.merge(customer_profiles, on='customer_id', how='left')
        
        # Generate fraud indicators
        fraud_probability = np.zeros(n_samples)
        
        # Risk factors for fraud
        # Large amounts
        fraud_probability += np.where(df['amount'] > df['avg_transaction_amount'] * 5, 0.3, 0)
        
        # Unusual times
        fraud_probability += np.where(df['is_night'], 0.1, 0)
        fraud_probability += np.where(df['is_weekend'], 0.05, 0)
        
        # Online transactions have higher fraud risk
        fraud_probability += np.where(df['merchant_category'] == 'online', 0.15, 0)
        
        # New accounts are riskier
        account_age = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['account_created'])).dt.days
        fraud_probability += np.where(account_age < 30, 0.2, 0)
        
        # High-risk customer profiles
        fraud_probability += np.where(df['risk_profile'] == 'high', 0.25, 0)
        
        # Add some randomness
        fraud_probability += np.random.beta(1, 50, n_samples)
        fraud_probability = np.clip(fraud_probability, 0, 0.95)
        
        # Generate actual fraud labels
        df['is_fraud'] = np.random.binomial(1, fraud_probability)
        df['fraud_score'] = fraud_probability
        
        # For fraudulent transactions, make them more extreme
        fraud_mask = df['is_fraud'] == 1
        
        # Fraudulent transactions tend to be larger
        df.loc[fraud_mask, 'amount'] *= np.random.uniform(2, 10, fraud_mask.sum())
        
        # More likely to be at unusual times
        night_fraud = np.random.choice(fraud_mask.index[fraud_mask], 
                                      size=int(fraud_mask.sum() * 0.4), 
                                      replace=False)
        df.loc[night_fraud, 'hour'] = np.random.choice([1, 2, 3, 23], len(night_fraud))
        
        logger.info(f"Generated transaction data with {df['is_fraud'].mean():.2%} fraud rate")
        
        # Clean up temporary columns
        df = df.drop(['avg_transaction_amount', 'transaction_frequency', 
                     'preferred_categories', 'risk_profile'], axis=1)
        
        return df
    
    def generate_customer_profiles(self, n_customers: int = 5000) -> pd.DataFrame:
        """Generate comprehensive customer profile data"""
        
        logger.info(f"Generating {n_customers} customer profiles...")
        
        data = {
            'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
            'first_name': [self.fake.first_name() for _ in range(n_customers)],
            'last_name': [self.fake.last_name() for _ in range(n_customers)],
            'email': [self.fake.email() for _ in range(n_customers)],
            'phone': [self.fake.phone_number() for _ in range(n_customers)],
            'address': [self.fake.address().replace('\n', ', ') for _ in range(n_customers)],
            'city': [self.fake.city() for _ in range(n_customers)],
            'state': [self.fake.state_abbr() for _ in range(n_customers)],
            'zip_code': [self.fake.zipcode() for _ in range(n_customers)],
            'date_of_birth': [self.fake.date_of_birth(minimum_age=18, maximum_age=80) 
                            for _ in range(n_customers)],
            'registration_date': [self.fake.date_between(start_date='-5y', end_date='today') 
                                for _ in range(n_customers)],
            'customer_segment': np.random.choice(
                ['premium', 'standard', 'basic'], 
                n_customers, 
                p=[0.15, 0.6, 0.25]
            ),
            'preferred_contact': np.random.choice(
                ['email', 'phone', 'sms'], 
                n_customers, 
                p=[0.5, 0.3, 0.2]
            )
        }
        
        df = pd.DataFrame(data)
        df
        
        # Calculate age
        df['age'] = (pd.to_datetime('now') - pd.to_datetime(df['date_of_birth'], format = '%Y-%m-%d')).dt.days // 365
        
        # # Account tenure
        df['account_tenure_days'] = (pd.to_datetime('now') - pd.to_datetime(df['registration_date'], format = '%Y-%m-%d')).dt.days
        
        
        return df
    
    def add_seasonal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic seasonal patterns to transaction data"""
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Holiday spending patterns
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        # Black Friday / Cyber Monday (November)
        black_friday_mask = (df['month'] == 11) & (df['day'].between(23, 30))
        df.loc[black_friday_mask, 'amount'] *= np.random.uniform(1.5, 3.0, black_friday_mask.sum())
        
        # Christmas shopping (December)
        christmas_mask = (df['month'] == 12) & (df['day'] < 25)
        df.loc[christmas_mask, 'amount'] *= np.random.uniform(1.2, 2.0, christmas_mask.sum())
        
        # Back to school (August/September)
        back_to_school_mask = df['month'].isin([8, 9])
        df.loc[back_to_school_mask, 'amount'] *= np.random.uniform(1.1, 1.5, back_to_school_mask.sum())
        
        # Summer vacation (June/July)
        summer_mask = df['month'].isin([6, 7]) & (df['merchant_category'] == 'travel')
        df.loc[summer_mask, 'amount'] *= np.random.uniform(2.0, 4.0, summer_mask.sum())
        
        return df
    
    def create_realistic_correlations(self, credit_df: pd.DataFrame, 
                                    transaction_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create realistic correlations between datasets"""
        
        # Merge datasets on customer_id to create correlations
        common_customers = set(credit_df['customer_id']).intersection(
            set(transaction_df['customer_id'])
        )
        
        logger.info(f"Creating correlations for {len(common_customers)} common customers")
        
        for customer_id in common_customers:
            # Get customer's credit profile
            credit_row = credit_df[credit_df['customer_id'] == customer_id].iloc[0]
            
            # Adjust transaction patterns based on credit profile
            cust_mask = transaction_df['customer_id'] == customer_id
            
            # Higher income customers have larger transactions
            if credit_row['annual_income'] > 100000:
                transaction_df.loc[cust_mask, 'amount'] *= np.random.uniform(1.5, 2.5)
            
            # Customers with defaults have more suspicious transaction patterns
            if credit_row.get('is_default', 0) == 1:
                # Slightly higher fraud probability
                fraud_boost = np.random.uniform(0.1, 0.3, cust_mask.sum())
                transaction_df.loc[cust_mask, 'fraud_score'] += fraud_boost
                transaction_df.loc[cust_mask, 'fraud_score'] = transaction_df.loc[cust_mask, 'fraud_score'].clip(0, 0.95)
        
        return credit_df, transaction_df
    
    def generate_complete_dataset(self, 
                                 n_customers: int = 5000,
                                 n_credit_applications: int = 10000,
                                 n_transactions: int = 100000,
                                 add_seasonal: bool = True,
                                 create_correlations: bool = True) -> Dict[str, pd.DataFrame]:
        """Generate complete banking dataset with all components"""
        
        logger.info("Generating complete banking dataset...")
        
        # Generate all datasets
        customer_profiles = self.generate_customer_profiles(n_customers)
        credit_data = self.generate_credit_data(n_credit_applications)
        transaction_data = self.generate_transaction_data(n_transactions, n_customers)
        
        # Add seasonal patterns
        if add_seasonal:
            transaction_data = self.add_seasonal_patterns(transaction_data)
        
        # Create realistic correlations
        if create_correlations:
            credit_data, transaction_data = self.create_realistic_correlations(
                credit_data, transaction_data
            )
        
        datasets = {
            'customers': customer_profiles,
            'credit_applications': credit_data,
            'transactions': transaction_data
        }
        
        # Generate summary statistics
        summary = {
            'customers_count': len(customer_profiles),
            'credit_applications_count': len(credit_data),
            'transactions_count': len(transaction_data),
            'default_rate': credit_data['is_default'].mean(),
            'fraud_rate': transaction_data['is_fraud'].mean(),
            'date_range': {
                'credit_start': credit_data['application_date'].min(),
                'credit_end': credit_data['application_date'].max(),
                'transaction_start': transaction_data['timestamp'].min(),
                'transaction_end': transaction_data['timestamp'].max()
            }
        }
        
        datasets['summary'] = pd.DataFrame([summary])
        
        logger.info("Dataset generation completed!")
        logger.info(f"Summary: {n_customers} customers, {n_credit_applications} credit apps, "
                   f"{n_transactions} transactions")
        logger.info(f"Default rate: {summary['default_rate']:.2%}, "
                   f"Fraud rate: {summary['fraud_rate']:.2%}")
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str):
        """Save all datasets to CSV files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in datasets.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} dataset to {filepath}")

# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Generate synthetic banking data")
    parser.add_argument("--customers", type=int, default=5000, help="Number of customers")
    parser.add_argument("--credit-apps", type=int, default=10000, help="Number of credit applications")
    parser.add_argument("--transactions", type=int, default=100000, help="Number of transactions")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Generate data
    generator = BankingDataGenerator(seed=args.seed)
    datasets = generator.generate_complete_dataset(
        n_customers=args.customers,
        n_credit_applications=args.credit_apps,
        n_transactions=args.transactions
    )
    
    # Save datasets
    generator.save_datasets(datasets, args.output_dir)
    
    print(f"\n✅ Data generation completed!")
    print(f"📁 Files saved to: {args.output_dir}")
    print(f"📊 Generated {args.customers:,} customers, {args.credit_apps:,} credit applications, "
          f"{args.transactions:,} transactions")
