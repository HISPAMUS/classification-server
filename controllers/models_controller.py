from fastapi import APIRouter, HTTPException, Form
from typing  import Optional
from output_messages.output import ListMessage, BasicMessage
from managers.model_manager import *
from .image_controller import check_image_exists_sync, read

router = APIRouter()

@router.get('/models', response_model = ListMessage)
async def get_available_models(notationType:str = Form(...), 
                         manuscriptType:str = Form(...), 
                         collection:Optional[str] = Form(...),
                         document:Optional[str] = Form(...),
                         classifierModelType:Optional[str] = Form(...)):
    
    modelsList = getModelList("", notationType, manuscriptType, collection, document, classifierModelType)
    #print(type(modelsList[0]))
    return ListMessage(message=modelsList)


@router.post('/image/{id}/e2e', response_model = ListMessage)
async def e2e_classify(id, model:str = Form(...), left:int = Form(...), top:int = Form(...), right:int = Form(...), bottom:int = Form(...), predictions:Optional[int] = Form(...)):
    try: 
        model = getE2EModel(model)
    except IOError as e:
        return HTTPException(404, f"The requested model ({model}) does not exist in our database")

    if not check_image_exists_sync(id):
        return HTTPException(404, f"The requested image to be analyzed ({id}) does not exist in the server")

    try:
        target_image = read()
    except Exception as e:
        return HTTPException(400, f"Error reading and cropping the image")

    predictions = model.predict(image = target_image)
    result = [{
                "shape": x[0].split(":")[0],
                "position": x[0].split(":")[1],
                "start": x[1],
                "end": x[2]
                } for x in predictions]

    return ListMessage(message=result)

