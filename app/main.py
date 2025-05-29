import json
import os
import logging
import argparse
from fastapi import FastAPI
from app.routers import predict_router
from app.utils.yolo_handler import YOLOHandler

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'device': 'cpu',
    'host': '0.0.0.0',
    'port': 8000,
    'reload': False,
    'workers': 1
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FastAPI YOLO Application')
    parser.add_argument('--device', type=str, default=DEFAULT_CONFIG['device'], 
                       help=f'Device to run the model on (e.g., cpu, cuda, mps). Default: {DEFAULT_CONFIG["device"]}')
    parser.add_argument('--host', type=str, default=DEFAULT_CONFIG['host'],
                       help=f'Host to run the server on. Default: {DEFAULT_CONFIG["host"]}')
    parser.add_argument('--port', type=int, default=DEFAULT_CONFIG['port'],
                       help=f'Port to run the server on. Default: {DEFAULT_CONFIG["port"]}')
    parser.add_argument('--reload', action='store_true', default=DEFAULT_CONFIG['reload'],
                       help='Enable auto-reload. Default: False')
    parser.add_argument('--workers', type=int, default=DEFAULT_CONFIG['workers'],
                       help=f'Number of worker processes. Default: {DEFAULT_CONFIG["workers"]}')
    return parser.parse_args()

def configure_app(fastapi_app: FastAPI, device: str = 'cpu') -> None:
    """Configure the FastAPI application with YOLO models.
    
    Args:
        fastapi_app: The FastAPI application instance
        device: Device to load models on (e.g., 'cpu', 'cuda')
    """
    yolo_handler = YOLOHandler()
    
    config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info("Loading models...")

    for model_cfg in config["models"]:
        yolo_handler.load_model(
            model_id=model_cfg["model_id"],
            model_path=model_cfg["model_path"],
            device=device,
            imgsz=model_cfg.get("imgsz", 640),
            conf=model_cfg.get("conf", 0.25)
        )
    
    fastapi_app.state.yolo_handler = yolo_handler

# Configure the app with default values when imported
configure_app(app, device=DEFAULT_CONFIG['device'])

# Include routers
app.include_router(predict_router.router)

# This will only run when the app is started directly with 'python -m app.main'
if __name__ == "__main__":
    args = parse_args()
    import uvicorn
    
    # Reconfigure with command line args
    configure_app(app, device=args.device)
    logger.info(f"Server starting on {args.host}:{args.port}")
    
    uvicorn.run("app.main:app", 
                host=args.host, 
                port=args.port, 
                reload=args.reload, 
                workers=args.workers)