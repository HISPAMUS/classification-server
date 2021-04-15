from fastapi import APIRouter, HTTPException, Form
from typing  import Optional
from output_messages.output import ListMessage, RegionsResponse, BasicMessage, SymbolsResponse, TranslationResponse
from managers.model_manager import *
from .image_controller import check_image_exists_sync, read, read_simple, crop
from keras import backend as K

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


@router.post('/image/{id}/e2e')
async def e2e_classify(id, model:str = Form(...), left:int = Form(...), top:int = Form(...), right:int = Form(...), bottom:int = Form(...), predictions:Optional[int] = Form(...)):
    try:
        model = getE2EModel(model)
    except Exception as e:
        logger_term.LogError(f'There was an error loading the model -> {e}')
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
    #logger_term.LogInfo(predictions)
    result = [{
                "shape": x[0].split(":")[0],
                "position": x[0].split(":")[1],
                "start": x[1],
                "end": x[2]
                } for x in predictions]

    #logger_term.LogInfo(result)
    
    #Cerrar la sesion?
    model.close()
    return result

@router.post('/translate')
async def agn2sem_translate(model:str = Form(...), agnostic:str = Form(...)):
    try:
        model = getTranslationModel(model)
    except Exception as e:
        logger_term.LogError(f'There was an error loading the model -> {e}')
        raise HTTPException(404, f"The requested model ({model}) does not exist in our database")

    prediction = model.predict(input = agnostic)
    model.close()

    return TranslationResponse(semantic=prediction)

@router.post('/image/{id}/docAnalysis', response_model = RegionsResponse)
async def document_analysis_classify(id, model:str = Form(...)):
    logger_term.LogInfo(f"Starting method")
    if not check_image_exists_sync(id):
        logger_term.LogError(f"Image {id} not found")
        raise HTTPException(404, f'Image [{id}] does not exist')
    try:
        image = read_simple(id)
    except Exception as e:
        raise HTTPException(400, f"Error reading the image {id}: {e}")
    
    documentAnalysisModel = getDocumentAnalysisModel(model)

    boundings = []

    if model == "simple-lan":
        boundings = documentAnalysisModel.predict(image)
    else:
        bboxes = predict_regions(documentAnalysisModel.getModel(), image, block_size=(512,512,3))
        for contour in bboxes:
            boundings.append({"x0": contour[1], "y0": contour[0], "xf": contour[3], "yf": contour[2], "regionType": "staff"})

    boundings.append({"regionType":"undefined"})
    boundings.append({"regionType":"title"})
    boundings.append({"regionType":"text"})
    boundings.append({"regionType":"author"})
    boundings.append({"regionType":"empty_staff"})
    boundings.append({"regionType":"lyrics"})
    boundings.append({"regionType":"multiple_lyrics"})
    boundings.append({"regionType":"other"})
    boundings.append({"regionType":"chords"})

    logger_term.LogInfo(boundings)

    return RegionsResponse(regions=boundings)


@router.post('/image/{id}/symbol', response_model = SymbolsResponse)
@router.post('/image/{id}/bbox', response_model = SymbolsResponse)
async def symbol_classify(id, model:str = Form(...), left:int = Form(...), top:int = Form(...), right:int = Form(...), bottom:int = Form(...), predictions:Optional[int] = Form(...)):
    if not check_image_exists_sync(id):
        logger_term.LogError(f"Image {id} not found")
        raise HTTPException(404, f'Image [{id}] does not exist')
    
    try:
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        n = int(predictions)
    except ValueError as e:
        logger_term.LogError(f"Wrong input values - {e}")
        raise HTTPException(404, f'Wrong input values - {e}')

    try:
        shape_image, position_image = crop(id, left, top, right, bottom)
    except Exception as e:
        logger_term.LogError(f"Error cropping image - {e}")
        raise HTTPException(400, f'Error cropping image - {e}')
    
    try:
        model = getSymbolsRecogintionModel(model)
    except Exception as e:
       logger_term.LogError(f"Error loading model - {e}")
       raise HTTPException(400, f"Error loading the model {e}")

    shape, position = model.predict(shape_image, position_image, n)
    
    if shape is None or position is None:
        raise HTTPException(400, "Error predicting symbols")
    
    K.clear_session()
    result = { 'shape': shape, 'position': position }
    return result



