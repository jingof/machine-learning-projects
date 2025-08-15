# api/main.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime
import uvicorn
import redis
import json
from contextlib import asynccontextmanager

# Import our models
from src.models.credit_risk_model import CreditRiskModel
from src.models.fraud_detection_model import FraudDetectionModel, FraudMonitoring
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
credit_model = None
fraud_model = None
fraud_monitor = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global credit_model, fraud_model, fraud_monitor, redis_client
    
    logger.info("Loading AI models...")
    
    # Initialize Redis
    try:
        redis_client = redis.Redis(**Config.REDIS_CONFIG)
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Load models (in production, load from saved files)
    try:
        credit_model = CreditRiskModel(Config.MODEL_CONFIG['credit_risk'])
        fraud_model = FraudDetectionModel(Config.MODEL_CONFIG['fraud_detection'])
        fraud_monitor = FraudMonitoring(fraud_model)
        logger.info("AI models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(
    title="Banking AI System API",
    description="Advanced AI system for credit risk assessment and fraud detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class CreditApplicationRequest(BaseModel):
    customer_id: str
    annual_income: float = Field(..., gt=0)
    employment_years: float = Field(..., ge=0)
    debt_to_income_ratio: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., gt=0)
    loan_purpose: str
    property_value: Optional[float] = None
    down_payment: Optional[float] = None
    
    # Additional fields for enhanced assessment
    total_debt: float = Field(default=0)
    total_credit_used: float = Field(default=0)
    total_credit_limit: float = Field(default=0)
    late_payments: int = Field(default=0)
    total_payments: int = Field(default=0)
    total_payments_amount: float = Field(default=0)
    credit_history_months: int = Field(default=0)
    number_of_accounts: int = Field(default=0)
    recent_credit_inquiries: int = Field(default=0)
    age: int = Field(default=35)

class CreditAssessmentResponse(BaseModel):
    customer_id: str
    risk_score: float
    risk_category: str
    approval_recommendation: bool
    confidence_score: float
    key_factors: List[str]
    explanation: Dict[str, Any]
    timestamp: datetime

class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float = Field(..., gt=0)
    timestamp: datetime
    merchant_category: Optional[str] = None
    payment_method: Optional[str] = None
    location: Optional[str] = None
    account_created: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class FraudDetectionResponse(BaseModel):
    transaction_id: str
    customer_id: str
    is_fraud: bool
    fraud_score: float
    risk_level: str
    confidence_score: float
    alerts: List[str]
    timestamp: datetime
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    redis_connected: bool

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded={
            "credit_risk": credit_model is not None,
            "fraud_detection": fraud_model is not None
        },
        redis_connected=redis_client is not None and redis_client.ping()
    )

# Credit Risk Assessment Endpoints
@app.post("/api/v1/credit/assess", response_model=CreditAssessmentResponse)
async def assess_credit_risk(request: CreditApplicationRequest):
    """Assess credit risk for a loan application"""
    
    if credit_model is None:
        raise HTTPException(status_code=503, detail="Credit risk model not available")
    
    start_time = datetime.now()
    
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction, probability = credit_model.predict(data)
        
        # Get explanation
        explanation = credit_model.explain_prediction(data)
        
        # Determine risk category and recommendation
        risk_score = float(probability[0])
        
        if risk_score < 0.3:
            risk_category = "LOW"
            approval_recommendation = True
        elif risk_score < 0.7:
            risk_category = "MEDIUM"
            approval_recommendation = True
        else:
            risk_category = "HIGH"
            approval_recommendation = False
        
        # Get feature importance for key factors
        feature_importance = credit_model.get_feature_importance()
        top_factors = sorted(feature_importance.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]
        key_factors = [factor[0] for factor in top_factors]
        
        # Calculate confidence score
        confidence_score = float(max(risk_score, 1 - risk_score))
        
        response = CreditAssessmentResponse(
            customer_id=request.customer_id,
            risk_score=risk_score,
            risk_category=risk_category,
            approval_recommendation=approval_recommendation,
            confidence_score=confidence_score,
            key_factors=key_factors,
            explanation=explanation,
            timestamp=datetime.now()
        )
        
        # Cache result in Redis if available
        if redis_client:
            try:
                cache_key = f"credit_assessment:{request.customer_id}"
                redis_client.setex(cache_key, 3600, json.dumps(response.dict(), default=str))
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Credit assessment completed in {processing_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Credit assessment failed: {e}")
        raise HTTPException(status_code=500, detail="Credit assessment failed")

@app.get("/api/v1/credit/history/{customer_id}")
async def get_credit_history(customer_id: str):
    """Get cached credit assessment history for a customer"""
    
    if not redis_client:
        raise HTTPException(status_code=503, detail="Cache service not available")
    
    try:
        cache_key = f"credit_assessment:{customer_id}"
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        else:
            raise HTTPException(status_code=404, detail="No credit history found")
            
    except Exception as e:
        logger.error(f"Failed to retrieve credit history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve credit history")

# Fraud Detection Endpoints
@app.post("/api/v1/fraud/detect", response_model=FraudDetectionResponse)
async def detect_fraud(request: TransactionRequest):
    """Detect fraud for a transaction"""
    
    if fraud_model is None or fraud_monitor is None:
        raise HTTPException(status_code=503, detail="Fraud detection model not available")
    
    start_time = datetime.now()
    
    try:
        # Process transaction through fraud monitor
        result = fraud_monitor.process_transaction(request.dict())
        
        # Generate alerts based on risk level
        alerts = []
        if result['risk_level'] == 'HIGH':
            alerts.append("High fraud risk detected - manual review required")
        if result['fraud_score'] > 0.9:
            alerts.append("Critical fraud score - block transaction immediately")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = FraudDetectionResponse(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            is_fraud=result['is_fraud'],
            fraud_score=result['fraud_score'],
            risk_level=result['risk_level'],
            confidence_score=min(result['fraud_score'] * 2, 1.0),
            alerts=alerts,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
        # Log high-risk transactions
        if result['fraud_score'] > 0.7:
            logger.warning(f"High-risk transaction detected: {request.transaction_id}, "
                         f"Score: {result['fraud_score']:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {e}")
        raise HTTPException(status_code=500, detail="Fraud detection failed")

@app.get("/api/v1/fraud/statistics")
async def get_fraud_statistics():
    """Get current fraud detection statistics"""
    
    if fraud_monitor is None:
        raise HTTPException(status_code=503, detail="Fraud monitoring not available")
    
    return fraud_monitor.get_statistics()

@app.get("/api/v1/fraud/alerts")
async def get_recent_alerts(limit: int = 10):
    """Get recent fraud alerts"""
    
    if fraud_monitor is None:
        raise HTTPException(status_code=503, detail="Fraud monitoring not available")
    
    return fraud_monitor.get_recent_alerts(limit)

# Model Management Endpoints
@app.post("/api/v1/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)"""
    
    def retrain_task():
        logger.info("Starting model retraining...")
        # In production, this would retrain models with new data
        # For now, just log the action
        logger.info("Model retraining completed")
    
    background_tasks.add_task(retrain_task)
    
    return {"message": "Model retraining started", "status": "accepted"}

@app.get("/api/v1/models/status")
async def get_model_status():
    """Get status of loaded models"""
    
    status = {
        "credit_risk_model": {
            "loaded": credit_model is not None,
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat()
        },
        "fraud_detection_model": {
            "loaded": fraud_model is not None,
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat()
        }
    }
    
    return status

# Batch Processing Endpoints
@app.post("/api/v1/credit/batch-assess")
async def batch_credit_assessment(applications: List[CreditApplicationRequest]):
    """Process multiple credit applications in batch"""
    
    if credit_model is None:
        raise HTTPException(status_code=503, detail="Credit risk model not available")
    
    if len(applications) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    results = []
    
    for app in applications:
        try:
            # Convert to DataFrame and predict
            data = pd.DataFrame([app.dict()])
            prediction, probability = credit_model.predict(data)
            
            risk_score = float(probability[0])
            risk_category = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW"
            
            results.append({
                "customer_id": app.customer_id,
                "risk_score": risk_score,
                "risk_category": risk_category,
                "approval_recommendation": risk_score < 0.7
            })
            
        except Exception as e:
            logger.error(f"Failed to process application for {app.customer_id}: {e}")
            results.append({
                "customer_id": app.customer_id,
                "error": "Processing failed"
            })
    
    return {"results": results, "processed_count": len(results)}

# WebSocket endpoint for real-time monitoring
@app.websocket("/ws/fraud-monitor")
async def fraud_websocket(websocket):
    """WebSocket endpoint for real-time fraud monitoring"""
    
    await websocket.accept()
    
    try:
        while True:
            # Send current statistics every 5 seconds
            if fraud_monitor:
                stats = fraud_monitor.get_statistics()
                await websocket.send_json(stats)
            
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_CONFIG["host"],
        port=Config.API_CONFIG["port"],
        workers=1,  # Set to 1 for development
        reload=True
    )
