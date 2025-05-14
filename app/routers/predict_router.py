import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import httpx
import asyncio
from app.schemas.model_response import ModelResponse

logger = logging.getLogger(__name__)

router = APIRouter()

N = 1
predict_semaphore = asyncio.Semaphore(N)

@router.post("/")
async def predict(
    request: Request,
    model_id: str,
    image_url: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """
    Endpoint to predict objects in an image using a specified model.

    Args:
        request (Request): FastAPI request object.
        model_id (str): Identifier of the model to use for prediction.
        image_url (Optional[str], optional): URL of the image to predict. Defaults to None.
        file (Optional[UploadFile], optional): Uploaded image file. Defaults to None.

    Returns:
        ModelResponse: JSON response containing prediction results.
    """
    logger.info("Received prediction request for model_id: %s", model_id)
    
    if image_url:
        logger.info("Downloading image from URL: %s", image_url)
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
        if response.status_code != 200:
            logger.error("Failed to download image from URL: %s", image_url)
            raise HTTPException(status_code=400, detail="Failed to download image from URL.")
        image_bytes = response.content
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