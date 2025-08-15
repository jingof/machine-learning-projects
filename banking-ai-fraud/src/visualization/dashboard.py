# src/visualization/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Any

# Set page config
st.set_page_config(
    page_title="Banking AI Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 2rem;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class BankingAIDashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:8000/api/v1"
        
    def make_api_call(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make API call with error handling"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return {}
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return {}
    
    def generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample data for demonstration"""
        
        # Credit applications data
        np.random.seed(42)
        n_applications = 1000
        
        credit_data = pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(n_applications)],
            'application_date': pd.date_range('2024-01-01', periods=n_applications, freq='H'),
            'annual_income': np.random.lognormal(10.5, 0.5, n_applications),
            'credit_score': np.random.normal(700, 100, n_applications).astype(int),
            'loan_amount': np.random.lognormal(11, 0.8, n_applications),
            'debt_to_income': np.random.beta(2, 8, n_applications),
            'employment_years': np.random.exponential(5, n_applications),
            'risk_score': np.random.beta(2, 5, n_applications),
            'approved': np.random.choice([True, False], n_applications, p=[0.75, 0.25])
        })
        
        # Transactions data
        n_transactions = 10000
        transaction_data = pd.DataFrame({
            'transaction_id': [f'TXN_{i:08d}' for i in range(n_transactions)],
            'customer_id': [f'CUST_{np.random.randint(0, 1000):06d}' for _ in range(n_transactions)],
            'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='15min'),
            'amount': np.random.lognormal(4, 1.5, n_transactions),
            'fraud_score': np.random.beta(1, 20, n_transactions),
            'is_fraud': np.random.choice([True, False], n_transactions, p=[0.02, 0.98]),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_transactions),
            'payment_method': np.random.choice(['credit_card', 'debit_card', 'bank_transfer'], n_transactions)
        })
        
        return {
            'credit_applications': credit_data,
            'transactions': transaction_data
        }
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🏦 Banking AI System Dashboard")
        st.markdown("---")
        
        # System status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Status", "🟢 Online", "All services operational")
        
        with col2:
            st.metric("Models Loaded", "2/2", "Credit Risk & Fraud Detection")
        
        with col3:
            st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"), "Real-time monitoring")
    
    def render_credit_risk_section(self, data: pd.DataFrame):
        """Render credit risk analysis section"""
        
        st.header("📊 Credit Risk Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_applications = len(data)
        approved_rate = data['approved'].mean()
        avg_risk_score = data['risk_score'].mean()
        high_risk_count = (data['risk_score'] > 0.7).sum()
        
        with col1:
            st.metric("Total Applications", f"{total_applications:,}", "📝")
        
        with col2:
            st.metric("Approval Rate", f"{approved_rate:.1%}", "✅")
        
        with col3:
            st.metric("Avg Risk Score", f"{avg_risk_score:.3f}", "⚠️")
        
        with col4:
            st.metric("High Risk Apps", f"{high_risk_count:,}", "🚨")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            fig_risk = px.histogram(
                data, x='risk_score', nbins=30,
                title="Risk Score Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            fig_risk.update_layout(
                xaxis_title="Risk Score",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            # Approval rate by risk category
            data['risk_category'] = pd.cut(
                data['risk_score'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Low', 'Medium', 'High']
            )
            
            approval_by_risk = data.groupby('risk_category')['approved'].mean().reset_index()
            
            fig_approval = px.bar(
                approval_by_risk, 
                x='risk_category', 
                y='approved',
                title="Approval Rate by Risk Category",
                color='approved',
                color_continuous_scale='RdYlGn'
            )
            fig_approval.update_layout(
                xaxis_title="Risk Category",
                yaxis_title="Approval Rate",
                showlegend=False
            )
            st.plotly_chart(fig_approval, use_container_width=True)
        
        # Time series analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Applications over time
            daily_apps = data.set_index('application_date').resample('D').size()
            
            fig_timeline = px.line(
                x=daily_apps.index, 
                y=daily_apps.values,
                title="Daily Credit Applications",
                labels={'x': 'Date', 'y': 'Applications'}
            )
            fig_timeline.update_traces(line_color='#2E86AB')
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Risk vs Income scatter
            fig_scatter = px.scatter(
                data.sample(500),  # Sample for performance
                x='annual_income',
                y='risk_score',
                color='approved',
                size='loan_amount',
                title="Risk Score vs Annual Income",
                color_discrete_map={True: '#4CAF50', False: '#F44336'}
            )
            fig_scatter.update_layout(
                xaxis_title="Annual Income ($)",
                yaxis_title="Risk Score"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def render_fraud_detection_section(self, data: pd.DataFrame):
        """Render fraud detection analysis section"""
        
        st.header("🛡️ Fraud Detection Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_transactions = len(data)
        fraud_rate = data['is_fraud'].mean()
        avg_fraud_score = data['fraud_score'].mean()
        high_risk_transactions = (data['fraud_score'] > 0.5).sum()
        
        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}", "💳")
        
        with col2:
            st.metric("Fraud Rate", f"{fraud_rate:.2%}", "🚨")
        
        with col3:
            st.metric("Avg Fraud Score", f"{avg_fraud_score:.3f}", "⚡")
        
        with col4:
            st.metric("High Risk Txns", f"{high_risk_transactions:,}", "⚠️")
        
        # Real-time fraud monitoring
        st.subheader("🔄 Real-Time Fraud Monitoring")
        
        # Create placeholder for real-time updates
        fraud_placeholder = st.empty()
        
        # Simulate real-time data
        recent_data = data.tail(100).copy()
        recent_data['timestamp'] = pd.date_range(
            end=datetime.now(), 
            periods=len(recent_data), 
            freq='1min'
        )
        
        with fraud_placeholder.container():
            col1, col2 = st.columns(2)
            
            with col1:
                # Recent transactions timeline
                fig_recent = px.scatter(
                    recent_data,
                    x='timestamp',
                    y='fraud_score',
                    color='is_fraud',
                    size='amount',
                    title="Recent Transaction Risk Scores",
                    color_discrete_map={True: '#FF4444', False: '#44AA44'}
                )
                fig_recent.add_hline(y=0.5, line_dash="dash", line_color="red")
                fig_recent.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Fraud Score"
                )
                st.plotly_chart(fig_recent, use_container_width=True)
            
            with col2:
                # Fraud by merchant category
                fraud_by_merchant = data.groupby('merchant_category').agg({
                    'is_fraud': ['sum', 'count']
                }).round(3)
                fraud_by_merchant.columns = ['fraud_count', 'total_count']
                fraud_by_merchant['fraud_rate'] = (
                    fraud_by_merchant['fraud_count'] / fraud_by_merchant['total_count']
                ).round(4)
                
                fig_merchant = px.bar(
                    fraud_by_merchant.reset_index(),
                    x='merchant_category',
                    y='fraud_rate',
                    title="Fraud Rate by Merchant Category",
                    color='fraud_rate',
                    color_continuous_scale='Reds'
                )
                fig_merchant.update_layout(
                    xaxis_title="Merchant Category",
                    yaxis_title="Fraud Rate"
                )
                st.plotly_chart(fig_merchant, use_container_width=True)
        
        # Fraud alerts
        st.subheader("🚨 Recent Fraud Alerts")
        
        high_risk_txns = data[data['fraud_score'] > 0.7].tail(10)
        
        if not high_risk_txns.empty:
            for _, txn in high_risk_txns.iterrows():
                risk_level = "HIGH" if txn['fraud_score'] > 0.8 else "MEDIUM"
                alert_class = "alert-high" if risk_level == "HIGH" else "alert-medium"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>Alert:</strong> {risk_level} Risk Transaction<br>
                    <strong>ID:</strong> {txn['transaction_id']}<br>
                    <strong>Amount:</strong> ${txn['amount']:.2f}<br>
                    <strong>Fraud Score:</strong> {txn['fraud_score']:.3f}<br>
                    <strong>Time:</strong> {txn['timestamp']}<br>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-risk transactions detected recently.")
    
    def render_model_performance_section(self, credit_data: pd.DataFrame, fraud_data: pd.DataFrame):
        """Render model performance metrics"""
        
        st.header("📈 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Credit Risk Model")
            
            # Simulate model performance metrics
            metrics = {
                'AUC Score': 0.87,
                'Precision': 0.84,
                'Recall': 0.79,
                'F1 Score': 0.81,
                'Accuracy': 0.83
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.3f}")
            
            # Feature importance chart
            features = ['debt_to_income', 'credit_score', 'employment_years', 
                       'annual_income', 'loan_amount']
            importance = np.random.dirichlet([5, 4, 3, 3, 2])
            
            fig_importance = px.bar(
                x=importance,
                y=features,
                orientation='h',
                title="Feature Importance - Credit Risk Model",
                color=importance,
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(
                xaxis_title="Importance",
                yaxis_title="Features",
                showlegend=False
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("Fraud Detection Model")
            
            # Simulate fraud model performance
            fraud_metrics = {
                'Precision': 0.92,
                'Recall': 0.88,
                'F1 Score': 0.90,
                'False Positive Rate': 0.02,
                'Detection Rate': 0.88
            }
            
            for metric, value in fraud_metrics.items():
                st.metric(metric, f"{value:.3f}")
            
            # Model ensemble performance
            models = ['Isolation Forest', 'Autoencoder', 'One-Class SVM']
            performance = [0.85, 0.91, 0.78]
            
            fig_ensemble = px.bar(
                x=models,
                y=performance,
                title="Ensemble Model Performance",
                color=performance,
                color_continuous_scale='Blues'
            )
            fig_ensemble.update_layout(
                xaxis_title="Model",
                yaxis_title="Performance Score",
                showlegend=False
            )
            st.plotly_chart(fig_ensemble, use_container_width=True)
        
        # Model drift monitoring
        st.subheader("📊 Model Drift Monitoring")
        
        # Simulate drift metrics over time
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        drift_scores = np.random.beta(2, 10, len(dates)) * 0.1
        
        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(
            x=dates,
            y=drift_scores,
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='#FF6B6B')
        ))
        fig_drift.add_hline(y=0.05, line_dash="dash", line_color="red", 
                           annotation_text="Alert Threshold")
        
        fig_drift.update_layout(
            title="Model Drift Over Time",
            xaxis_title="Date",
            yaxis_title="Drift Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_drift, use_container_width=True)
    
    def render_interactive_testing(self):
        """Render interactive model testing section"""
        
        st.header("🧪 Interactive Model Testing")
        
        tab1, tab2 = st.tabs(["Credit Risk Assessment", "Fraud Detection"])
        
        with tab1:
            st.subheader("Test Credit Risk Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                customer_id = st.text_input("Customer ID", "CUST_001234")
                annual_income = st.number_input("Annual Income ($)", min_value=0, value=75000)
                employment_years = st.number_input("Employment Years", min_value=0.0, value=5.0)
                credit_score = st.slider("Credit Score", 300, 850, 720)
            
            with col2:
                loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=200000)
                debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
                loan_purpose = st.selectbox("Loan Purpose", 
                                          ["home_purchase", "refinance", "debt_consolidation"])
            
            if st.button("Assess Credit Risk", type="primary"):
                # Simulate API call
                with st.spinner("Processing..."):
                    time.sleep(1)  # Simulate processing time
                    
                    # Generate mock results
                    risk_score = np.random.beta(2, 5)
                    risk_category = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Risk Score", f"{risk_score:.3f}")
                    with col2:
                        st.metric("Risk Category", risk_category)
                    with col3:
                        recommendation = "REJECT" if risk_score > 0.7 else "APPROVE"
                        st.metric("Recommendation", recommendation)
                    
                    # Show explanation
                    st.subheader("Risk Factors")
                    factors = [
                        f"Credit Score: {credit_score} (Weight: 0.25)",
                        f"Debt-to-Income: {debt_to_income:.2f} (Weight: 0.20)",
                        f"Employment Stability: {employment_years} years (Weight: 0.15)",
                        f"Income Level: ${annual_income:,} (Weight: 0.18)",
                        f"Loan Amount: ${loan_amount:,} (Weight: 0.22)"
                    ]
                    
                    for factor in factors:
                        st.write(f"• {factor}")
        
        with tab2:
            st.subheader("Test Fraud Detection Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                txn_id = st.text_input("Transaction ID", "TXN_12345678")
                customer_id_fraud = st.text_input("Customer ID", "CUST_001234", key="fraud_customer")
                amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
                merchant_category = st.selectbox("Merchant Category", 
                                               ["grocery", "gas", "restaurant", "retail", "online"])
            
            with col2:
                payment_method = st.selectbox("Payment Method", 
                                            ["credit_card", "debit_card", "bank_transfer"])
                location = st.text_input("Location", "New York, NY")
                transaction_time = st.time_input("Transaction Time")
            
            if st.button("Detect Fraud", type="primary"):
                with st.spinner("Analyzing transaction..."):
                    time.sleep(1)  # Simulate processing time
                    
                    # Generate mock results
                    fraud_score = np.random.beta(1, 20)
                    is_fraud = fraud_score > 0.5
                    risk_level = "HIGH" if fraud_score > 0.8 else "MEDIUM" if fraud_score > 0.5 else "LOW"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fraud Score", f"{fraud_score:.3f}")
                    with col2:
                        st.metric("Risk Level", risk_level)
                    with col3:
                        status = "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE"
                        st.metric("Status", status)
                    
                    # Show risk factors
                    if fraud_score > 0.3:
                        st.subheader("Risk Indicators")
                        if amount > 500:
                            st.write("• Large transaction amount")
                        if merchant_category == "online":
                            st.write("• Online transaction (higher risk)")
                        if fraud_score > 0.7:
                            st.write("• Unusual transaction pattern detected")
                        if payment_method == "credit_card":
                            st.write("• Credit card transaction")
    
    def run_dashboard(self):
        """Main dashboard runner"""
        
        # Sidebar configuration
        st.sidebar.title("Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
        
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Data filters
        st.sidebar.subheader("Data Filters")
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Generate sample data
        sample_data = self.generate_sample_data()
        
        # Filter data by date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            sample_data['credit_applications'] = sample_data['credit_applications'][
                (sample_data['credit_applications']['application_date'].dt.date >= start_date) &
                (sample_data['credit_applications']['application_date'].dt.date <= end_date)
            ]
            sample_data['transactions'] = sample_data['transactions'][
                (sample_data['transactions']['timestamp'].dt.date >= start_date) &
                (sample_data['transactions']['timestamp'].dt.date <= end_date)
            ]
        
        # Render dashboard sections
        self.render_header()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Credit Risk", 
            "🛡️ Fraud Detection", 
            "📈 Model Performance", 
            "🧪 Interactive Testing"
        ])
        
        with tab1:
            self.render_credit_risk_section(sample_data['credit_applications'])
        
        with tab2:
            self.render_fraud_detection_section(sample_data['transactions'])
        
        with tab3:
            self.render_model_performance_section(
                sample_data['credit_applications'], 
                sample_data['transactions']
            )
        
        with tab4:
            self.render_interactive_testing()

# Main execution
if __name__ == "__main__":
    dashboard = BankingAIDashboard()
    dashboard.run_dashboard()
