from fastapi import APIRouter, HTTPException, Form
from typing  import Optional
from output_messages.output import ListMessage, BasicMessage
from managers.model_manager import *
from .image_controller import check_image_exists_sync, read


router = APIRouter()

import logging
from logger import Logger

logger_term = Logger()

@router.post('/models', response_model = ListMessage)
async def get_available_models(notationType:str = Form(...), 
                         manuscriptType:str = Form(...), 
                         collection:Optional[str] = Form(...),
                         document:Optional[str] = Form(...),
                         classifierModelType:Optional[str] = Form(...)):
    
    modelsList = getModelList("", notationType, manuscriptType, collection, document, classifierModelType)
    #print(type(modelsList[0]))
    return ListMessage(message=modelsList)


@router.post('/image/{id}/e2e', response_model=JSONResponse)
async def e2e_classify(id, model:str = Form(...), left:int = Form(...), top:int = Form(...), right:int = Form(...), bottom:int = Form(...), predictions:Optional[int] = Form(...)):
    try:
        model = getE2EModel(model)
    except Exception as e:
        raise HTTPException(404, f"The requested model ({model}) does not exist in our database")

    if not check_image_exists_sync(id):
        logger_term.LogError(f"Image {id} does not exist")
        raise HTTPException(404, f"Image {id} does not exist")

    try:
        logger_term.LogInfo("Loading image")
        target_image = read(id, left, top, right, bottom)
    except Exception as e:
        raise HTTPException(400, f"Error reading and cropping the image: {e}")

    predictions = model.predict(image = target_image)
    result = [{
                "shape": x[0].split(":")[0],
                "position": x[0].split(":")[1],
                "start": x[1],
                "end": x[2]
                } for x in predictions]

    logger_term.LogInfo(result)

    return result



