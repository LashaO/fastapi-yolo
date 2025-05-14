import json
import logging
from fastapi import FastAPI
from app.routers import predict_router
from app.utils.yolo_handler import YOLOHandler

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
    
    Initializes the YOLOHandler and loads model configurations from a JSON file.
    """
    yolo_handler = YOLOHandler()
    
    # Load model configurations from JSON file
    with open('app/model_config.json', 'r') as f:
        config = json.load(f)

    logger.info("Server has started")

    for model_cfg in config["models"]:
        yolo_handler.load_model(
            model_id=model_cfg["model_id"],
            model_path=model_cfg["model_path"],
            imgsz=model_cfg["imgsz"],
            conf=model_cfg["conf"],
            device="mps"
        )
    
    app.state.yolo_handler = yolo_handler

app.include_router(predict_router.router, prefix="/predict", tags=["Prediction"])