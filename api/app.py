"""FastAPI application for PPG glucose estimation service."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.models.hybrid_model import HybridCNNGRU
from src.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from src.quality.validator import SignalQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PPG Glucose Estimation API",
    description="Non-invasive blood glucose estimation from PPG signals",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and preprocessor
MODEL = None
PREPROCESSOR = None
VALIDATOR = None


class PredictionRequest(BaseModel):
    """Request model for PPG prediction."""
    
    ppg: List[float] = Field(..., description="PPG signal window (30Hz)")
    fs: float = Field(30.0, description="Sampling frequency")
    
    class Config:
        schema_extra = {
            "example": {
                "ppg": [0.1, 0.2, -0.1] * 100,  # 300 samples
                "fs": 30.0,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for PPG prediction."""
    
    glucose_mgdl: float = Field(..., description="Predicted glucose in mg/dL")
    quality: float = Field(..., description="Signal quality score (0-1)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "glucose_mgdl": 95.5,
                "quality": 0.85,
                "confidence": 0.78,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup."""
    global MODEL, PREPROCESSOR, VALIDATOR
    
    try:
        # Load configuration
        config = PreprocessingConfig()
        PREPROCESSOR = PreprocessingPipeline(config)
        VALIDATOR = SignalQualityValidator()
        
        # Load model
        model_path = Path("models/best.ckpt")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            MODEL = HybridCNNGRU()
            MODEL.load_state_dict(checkpoint["state_dict"])
            MODEL.eval()
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}")
            # Create dummy model for testing
            MODEL = HybridCNNGRU()
            MODEL.eval()
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/")
async def root():
    """Root endpoint with web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPG Glucose Estimation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .container { max-width: 800px; margin: auto; }
            .input-group { margin: 20px 0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            #result { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PPG Glucose Estimation API</h1>
            <p>Non-invasive blood glucose estimation from photoplethysmography signals</p>
            
            <div class="input-group">
                <h3>API Documentation</h3>
                <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
            </div>
            
            <div class="input-group">
                <h3>Quick Test</h3>
                <button onclick="testPrediction()">Test with Random Signal</button>
                <div id="result"></div>
            </div>
        </div>
        
        <script>
            async function testPrediction() {
                const ppg = Array.from({length: 300}, () => Math.random() * 2 - 1);
                
                try {
                    const response = await fetch('/v1/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ppg: ppg, fs: 30})
                    });
                    
                    const data = await response.json();
                    document.getElementById('result').innerHTML = `
                        <h4>Prediction Result:</h4>
                        <p>Glucose: ${data.glucose_mgdl.toFixed(1)} mg/dL</p>
                        <p>Quality: ${(data.quality * 100).toFixed(1)}%</p>
                        <p>Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
                    `;
                } catch (error) {
                    document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        version="1.0.0",
    )


@app.get("/version")
async def version():
    """Get API version."""
    return {"version": "1.0.0", "model_version": "hybrid-cnn-gru-v1"}


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict glucose from PPG window.
    
    Args:
        request: PPG signal window and parameters
        
    Returns:
        Glucose prediction with quality metrics
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to numpy array
        ppg_array = np.array(request.ppg, dtype=np.float32)
        
        # Validate window length
        expected_length = int(10 * request.fs)  # 10 seconds
        if len(ppg_array) != expected_length:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_length} samples, got {len(ppg_array)}"
            )
        
        # Validate signal quality
        quality_report = VALIDATOR.validate(ppg_array)
        
        if not quality_report.is_valid:
            return PredictionResponse(
                glucose_mgdl=0.0,
                quality=quality_report.overall_quality,
                confidence=0.0,
            )
        
        # Prepare input tensor
        input_tensor = torch.FloatTensor(ppg_array).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            glucose_pred = MODEL(input_tensor).item()
        
        # Clip to reasonable range
        glucose_pred = np.clip(glucose_pred, 40, 400)
        
        return PredictionResponse(
            glucose_mgdl=glucose_pred,
            quality=quality_report.overall_quality,
            confidence=quality_report.confidence,
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/v1/predict/stream")
async def predict_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming predictions.
    
    Accepts continuous PPG chunks and returns rolling glucose estimates.
    """
    await websocket.accept()
    
    buffer = []
    window_size = 300  # 10s at 30Hz
    hop_size = 150  # 5s hop
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_text()
            chunk = json.loads(data)
            
            # Add to buffer
            buffer.extend(chunk["ppg"])
            
            # Process windows
            while len(buffer) >= window_size:
                window = np.array(buffer[:window_size], dtype=np.float32)
                
                # Validate and predict
                quality_report = VALIDATOR.validate(window)
                
                if quality_report.is_valid:
                    input_tensor = torch.FloatTensor(window).unsqueeze(0)
                    with torch.no_grad():
                        glucose = MODEL(input_tensor).item()
                    glucose = np.clip(glucose, 40, 400)
                else:
                    glucose = 0.0
                
                # Send result
                result = {
                    "glucose_mgdl": glucose,
                    "quality": quality_report.overall_quality,
                    "confidence": quality_report.confidence,
                    "timestamp": chunk.get("timestamp", 0),
                }
                await websocket.send_text(json.dumps(result))
                
                # Slide buffer
                buffer = buffer[hop_size:]
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8080)