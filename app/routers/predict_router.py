import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import httpx
import asyncio
from pathlib import Path
from app.schemas.model_response import ModelResponse

logger = logging.getLogger(__name__)

router = APIRouter()

N = 1
predict_semaphore = asyncio.Semaphore(N)

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    model_id: str
    image_url: Optional[str] = None
    image_path: Optional[str] = None

@router.post("/")
async def predict(
    request: Request,
    body: PredictionRequest,
    file: Optional[UploadFile] = File(None)
):
    """
    Endpoint to predict objects in an image using a specified model.

    Args:
        request (Request): FastAPI request object.
        model_id (str): Identifier of the model to use for prediction.
        image_url (Optional[str], optional): URL of the image to predict. Defaults to None.
        image_path (Optional[str], optional): Path to a local image file. Defaults to None.
        file (Optional[UploadFile], optional): Uploaded image file. Defaults to None.

    Returns:
        ModelResponse: JSON response containing prediction results.

    Raises:
        HTTPException: If no image source is provided or if image loading fails.
    """
    logger.info("Received prediction request for model_id: %s", model_id)
    
    # Validate that at least one image source is provided
    if not any([image_url, image_path, file]):
        raise HTTPException(status_code=400, detail="At least one image source (URL, path, or file) must be provided")
    
    # Get image bytes based on the provided source
    image_bytes = None
    
    if image_url:
        logger.info("Downloading image from URL: %s", image_url)
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
        if response.status_code != 200:
            logger.error("Failed to download image from URL: %s", image_url)
            raise HTTPException(status_code=400, detail="Failed to download image from URL.")
        image_bytes = response.content
    
    elif image_path:
        logger.info("Loading image from path: %s", image_path)
        try:
            # Convert to absolute path
            image_path = str(Path(image_path).resolve())
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        except Exception as e:
            logger.error("Failed to load image from path: %s", image_path)
            raise HTTPException(status_code=400, detail=f"Failed to load image from path: {str(e)}")
    
    elif file:
        logger.info("Processing uploaded file")
        image_bytes = await file.read()
    elif file:
        logger.info("Reading uploaded file")
        image_bytes = await file.read()
    else:
        logger.error("No image provided")
        raise HTTPException(status_code=400, detail="No image provided.")

    yolo_handler = request.app.state.yolo_handler
    async with predict_semaphore:
        logger.info("Running prediction for model_id: %s", model_id)
        results = yolo_handler.predict(model_id, image_bytes)
    
    results = results[0]

    names = yolo_handler.models[model_id]["model"].names

    bboxes = results.boxes.xywh.tolist()
    scores = results.boxes.conf.tolist()
    class_ids = results.boxes.cls.tolist()
    class_names = [names[class_id] for class_id in class_ids]

    thetas = [0]*len(bboxes)

    logger.info("Prediction completed for model_id: %s", model_id)

    return ModelResponse(
        bboxes=bboxes,
        scores=scores,
        thetas=thetas,
        class_ids=class_ids,
        class_names=class_names
    )