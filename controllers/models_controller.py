from fastapi import APIRouter, HTTPException, Form
from typing  import Optional
from output_messages.output import ListMessage
from managers.model_manager import *

router = APIRouter()

@router.get('/models', response_model = ListMessage)
def get_available_models(notationType:str = Form(...), 
                         manuscriptType:str = Form(...), 
                         collection:Optional[str] = Form(...),
                         document:Optional[str] = Form(...),
                         classifierModelType:Optional[str] = Form(...)):
    
    modelsList = getModelList("", notationType, manuscriptType, collection, document, classifierModelType)
    #print(type(modelsList[0]))
    return ListMessage(message=modelsList)

    
