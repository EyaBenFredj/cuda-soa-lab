from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Weights(BaseModel):
    w: float
    b: float

class AggregationRequest(BaseModel):
    weights_list: List[Weights]

@app.post("/aggregate")
async def aggregate(request: AggregationRequest):
    """Aggregate weights from multiple clients"""
    if not request.weights_list:
        raise HTTPException(status_code=400, detail="No weights provided")
    
    logger.info(f"Aggregating weights from {len(request.weights_list)} clients")
    
    # Federated averaging
    w_global = sum(weight.w for weight in request.weights_list) / len(request.weights_list)
    b_global = sum(weight.b for weight in request.weights_list) / len(request.weights_list)
    
    logger.info(f"Aggregation completed: w_global={w_global:.4f}, b_global={b_global:.4f}")
    
    return {
        "w_global": w_global,
        "b_global": b_global,
        "clients_count": len(request.weights_list),
        "message": "Federated averaging completed successfully"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "aggregator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")