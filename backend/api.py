from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List
import joblib
import pandas as pd
import json
from pathlib import Path

# =========================================================================
# PYDANTIC MODELS
# =========================================================================

class WineFeatures(BaseModel):
    """Wine physicochemical properties"""
    fixed_acidity: float = Field(..., ge=3.0, le=16.0, description="Tartaric acid (g/dm³)")
    volatile_acidity: float = Field(..., ge=0.0, le=2.0, description="Acetic acid (g/dm³)")
    citric_acid: float = Field(..., ge=0.0, le=1.5, description="Citric acid (g/dm³)")
    residual_sugar: float = Field(..., ge=0.0, le=20.0, description="Residual sugar (g/dm³)")
    chlorides: float = Field(..., ge=0.0, le=1.0, description="Sodium chloride (g/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=0.0, le=100.0, description="Free SO2 (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=0.0, le=300.0, description="Total SO2 (mg/dm³)")
    density: float = Field(..., ge=0.98, le=1.01, description="Density (g/cm³)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH level")
    sulphates: float = Field(..., ge=0.0, le=3.0, description="Potassium sulphate (g/dm³)")
    alcohol: float = Field(..., ge=8.0, le=15.0, description="Alcohol content (% by volume)")

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }

class WinePrediction(BaseModel):
    """Wine quality prediction response"""
    quality: str
    confidence: float
    probabilities: Dict[str, float]
    quality_score_range: str
    recommendation: str
    top_factors: List[Dict]

class ModelInfo(BaseModel):
    """Model metadata response"""
    name: str
    version: str
    model_type: str
    accuracy: float
    balanced_accuracy: float
    classes: List[str]
    features: List[str]

# =========================================================================
# FASTAPI APP
# =========================================================================

app = FastAPI(
    title="Wine Quality Prediction API",
    description="AI-powered wine quality assessment using Decision Tree Pipeline",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# LOAD MODEL (PIPELINE)
# =========================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "backend/model"
MODEL_PATH = MODEL_DIR / "model.joblib"
SCHEMA_PATH = MODEL_DIR / "schema.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Load pipeline (contains scaler + SMOTE + model)
try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Pipeline loaded from: {MODEL_PATH}")
    print(f"   Contains: StandardScaler + SMOTE + DecisionTree")
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    pipeline = None

# Load metadata
try:
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"✅ Schema and metadata loaded")
except Exception as e:
    print(f"⚠️ Could not load schema/metadata: {e}")
    schema = {}
    metadata = {}

# Feature importance (from training)
FEATURE_IMPORTANCE = {
    "alcohol": 0.32,
    "sulphates": 0.18,
    "volatile acidity": 0.15,  # Note: spaces in dataset
    "total sulfur dioxide": 0.10,
    "density": 0.08,
    "citric acid": 0.06,
    "pH": 0.04,
    "fixed acidity": 0.03,
    "chlorides": 0.02,
    "residual sugar": 0.01,
    "free sulfur dioxide": 0.01
}

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def convert_to_dataset_format(wine_dict: Dict) -> Dict:
    """
    Convert API input (underscores) to dataset format (spaces)
    
    API uses: fixed_acidity
    Dataset uses: fixed acidity
    """
    mapping = {
        'fixed_acidity': 'fixed acidity',
        'volatile_acidity': 'volatile acidity',
        'citric_acid': 'citric acid',
        'residual_sugar': 'residual sugar',
        'chlorides': 'chlorides',
        'free_sulfur_dioxide': 'free sulfur dioxide',
        'total_sulfur_dioxide': 'total sulfur dioxide',
        'density': 'density',
        'pH': 'pH',
        'sulphates': 'sulphates',
        'alcohol': 'alcohol'
    }
    
    return {mapping.get(k, k): v for k, v in wine_dict.items()}

def get_recommendation(quality: str, confidence: float) -> str:
    """Generate recommendation based on prediction"""
    recommendations = {
        "Premium": {
            "high": "🍷 Excellent wine! Premium quality with exceptional characteristics. Recommended for special occasions or cellaring.",
            "medium": "🍷 Very good wine with notable qualities. Great for enjoying now or short-term storage.",
            "low": "🍷 Good wine, though with some variability. Suitable for everyday drinking."
        },
        "Satisfactory": {
            "high": "🍷 Solid, dependable wine. Good for casual occasions and everyday enjoyment.",
            "medium": "🍷 Decent wine with acceptable quality. Suitable for cooking or casual drinking.",
            "low": "🍷 Average quality wine. Best consumed soon or used in recipes."
        },
        "Substandard": {
            "high": "⚠️ Below-average quality detected. May have defects or unbalanced characteristics.",
            "medium": "⚠️ Substandard wine with noticeable flaws. Not recommended for drinking.",
            "low": "⚠️ Poor quality wine. Consider quality improvement or use for non-drinking purposes."
        }
    }
    
    if confidence >= 0.8:
        conf_level = "high"
    elif confidence >= 0.6:
        conf_level = "medium"
    else:
        conf_level = "low"
    
    return recommendations[quality][conf_level]

def get_top_factors(wine_data: Dict[str, float]) -> list:
    """Get top contributing factors for this wine"""
    factors = []
    
    for feature, value in wine_data.items():
        # Get importance (feature name has spaces in dataset)
        importance = FEATURE_IMPORTANCE.get(feature, 0)
        
        factors.append({
            "feature": feature.title(),
            "value": round(value, 2),
            "importance": round(importance, 4)
        })
    
    # Sort by importance
    factors.sort(key=lambda x: x['importance'], reverse=True)
    return factors[:5]  # Return top 5

# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "service": "Wine Quality Prediction API",
        "status": "operational",
        "version": "2.0.0",
        "model": "Decision Tree Pipeline (StandardScaler + SMOTE + Classifier)",
        "classes": ["Substandard", "Satisfactory", "Premium"],
        "endpoints": {
            "predict": "POST /api/predict",
            "batch": "POST /api/predict/batch",
            "info": "GET /api/info",
            "health": "GET /api/health",
            "options": "GET /api/options",
            "feature_importance": "GET /api/feature-importance"
        },
        "documentation": "/docs"
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/api/info", response_model=ModelInfo)
def get_info():
    """Get model information and metadata"""
    if not metadata:
        raise HTTPException(status_code=503, detail="Metadata not available")
    
    return ModelInfo(
        name=schema.get("name", "Wine Quality Predictor"),
        version=schema.get("version", "2.0.0"),
        model_type=schema.get("model_type", "Pipeline"),
        accuracy=metadata.get("performance", {}).get("test_accuracy", 0),
        balanced_accuracy=metadata.get("performance", {}).get("balanced_accuracy", 0),
        classes=schema.get("classes", ["Substandard", "Satisfactory", "Premium"]),
        features=schema.get("features", [])
    )

@app.get("/api/options")
def get_options():
    """Get typical value ranges for wine features"""
    return {
        "typical_ranges": {
            "fixed_acidity": {"min": 4.6, "max": 15.9, "typical": 8.3, "unit": "g/dm³"},
            "volatile_acidity": {"min": 0.12, "max": 1.58, "typical": 0.5, "unit": "g/dm³"},
            "citric_acid": {"min": 0.0, "max": 1.0, "typical": 0.3, "unit": "g/dm³"},
            "residual_sugar": {"min": 0.9, "max": 15.5, "typical": 2.5, "unit": "g/dm³"},
            "chlorides": {"min": 0.012, "max": 0.611, "typical": 0.09, "unit": "g/dm³"},
            "free_sulfur_dioxide": {"min": 1.0, "max": 72.0, "typical": 15.0, "unit": "mg/dm³"},
            "total_sulfur_dioxide": {"min": 6.0, "max": 289.0, "typical": 46.0, "unit": "mg/dm³"},
            "density": {"min": 0.99, "max": 1.00, "typical": 0.996, "unit": "g/cm³"},
            "pH": {"min": 2.74, "max": 4.01, "typical": 3.31, "unit": "pH"},
            "sulphates": {"min": 0.33, "max": 2.0, "typical": 0.66, "unit": "g/dm³"},
            "alcohol": {"min": 8.4, "max": 14.9, "typical": 10.4, "unit": "% vol"}
        },
        "quality_indicators": {
            "high_alcohol": "Alcohol > 11% often indicates better quality",
            "low_volatile_acidity": "Volatile acidity < 0.6 is preferred (less vinegar taste)",
            "balanced_pH": "pH between 3.0-3.5 is ideal",
            "moderate_sulfites": "Total SO2 around 50-100 mg/dm³ is optimal"
        },
        "feature_importance_order": list(FEATURE_IMPORTANCE.keys())
    }

@app.post("/api/predict", response_model=WinePrediction)
def predict_quality(wine: WineFeatures):
    """
    Predict wine quality from physicochemical properties
    
    Returns quality class (Substandard/Satisfactory/Premium), confidence, and recommendations
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    try:
        # Convert input to dictionary
        wine_dict = wine.dict()
        
        # Convert to dataset format (spaces instead of underscores)
        wine_dataset_format = convert_to_dataset_format(wine_dict)
        
        # Create DataFrame with correct column names
        df = pd.DataFrame([wine_dataset_format])
        
        # Pipeline handles scaling + SMOTE automatically!
        prediction = pipeline.predict(df)[0]
        probabilities = pipeline.predict_proba(df)[0]
        
        # Format probabilities
        proba_dict = {
            cls: float(prob) 
            for cls, prob in zip(pipeline.classes_, probabilities)
        }
        
        confidence = max(proba_dict.values())
        
        # Quality score range mapping
        quality_ranges = {
            "Substandard": "3-4 (Basic wine)",
            "Satisfactory": "5-6 (Good wine)",
            "Premium": "7-9 (Excellent wine)"
        }
        
        # Get recommendation
        recommendation = get_recommendation(prediction, confidence)
        
        # Get top contributing factors
        top_factors = get_top_factors(wine_dataset_format)
        
        return WinePrediction(
            quality=prediction,
            confidence=confidence,
            probabilities=proba_dict,
            quality_score_range=quality_ranges[prediction],
            recommendation=recommendation,
            top_factors=top_factors
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/batch")
def predict_batch(wines: List[WineFeatures]):
    """
    Predict quality for multiple wines
    
    Useful for batch analysis of wine samples
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    try:
        results = []
        
        for idx, wine in enumerate(wines):
            wine_dict = wine.dict()
            wine_dataset_format = convert_to_dataset_format(wine_dict)
            df = pd.DataFrame([wine_dataset_format])
            
            prediction = pipeline.predict(df)[0]
            probabilities = pipeline.predict_proba(df)[0]
            
            proba_dict = {
                cls: float(prob) 
                for cls, prob in zip(pipeline.classes_, probabilities)
            }
            
            results.append({
                "wine_id": idx + 1,
                "quality": prediction,
                "confidence": float(max(proba_dict.values())),
                "probabilities": proba_dict
            })
        
        # Summary statistics
        quality_counts = {}
        for result in results:
            quality = result["quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        return {
            "predictions": results,
            "summary": {
                "total_wines": len(results),
                "quality_distribution": quality_counts,
                "average_confidence": sum(r["confidence"] for r in results) / len(results)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/api/feature-importance")
def get_feature_importance():
    """Get feature importance rankings"""
    return {
        "feature_importance": [
            {
                "rank": idx + 1,
                "feature": feature.title(),
                "importance": importance,
                "description": schema.get("feature_descriptions", {}).get(feature, "")
            }
            for idx, (feature, importance) in enumerate(
                sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
            )
        ],
        "interpretation": {
            "alcohol": "Higher alcohol content strongly correlates with better quality",
            "sulphates": "Adequate sulphates preserve wine and enhance flavor",
            "volatile_acidity": "Lower volatile acidity (less vinegar) improves quality"
        }
    }

# =========================================================================
# RUN SERVER
# =========================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🍷 WINE QUALITY PREDICTION API")
    print("="*70)
    print(f"Pipeline loaded: {pipeline is not None}")
    print("Server:          http://127.0.0.1:8000")
    print("API Docs:        http://127.0.0.1:8000/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)