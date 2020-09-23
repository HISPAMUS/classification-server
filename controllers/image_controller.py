from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from output_messages.output import BasicMessage
from pathlib import Path
import cv2

image_storage_path = "images/"

from logger import Logger

logger_term = Logger()

router = APIRouter()

@router.post('/image', response_model=BasicMessage)
async def save_image(id:str = Form(...), image:UploadFile = File(...)):
    imagecontents = await image.read()
    with open(image_storage_path + id + ".jpg", "wb") as destiny_file:
        destiny_file.write(imagecontents)
            
    return BasicMessage(message = f'Image {id} has been stored correctly')

@router.get('/image/{id}', response_model=BasicMessage, responses={404: {'description': 'Image not found'}})
async def check_if_image_exists(id):
    if Path(image_storage_path + id + ".jpg").exists():
        return BasicMessage(message = f'Image {id} exists in storage')
    else:
        raise HTTPException(404, f'The requested image {id} was not found')

@router.delete('/image/{id}', response_model=BasicMessage)
async def delete_image(id):
    if Path(image_storage_path + id + ".jpg").exists():
        os.remove(image_storage_path + id + ".jpg")
        return BasicMessage(message=f'Image {id} removed correctly')
    else:
        raise HTTPException(404, f'The requested image {id} was not found')

def check_image_exists_sync(id):
    return Path(image_storage_path + id + ".jpg").exists()

def read(id, left, top, right, bottom):
    try:
        image = read_simple(id)
        return image[top:bottom, left:right]
    except Exception as e:
        logger_term.LogInfo(e)

def read_simple(id):
    return cv2.imread(image_storage_path + id + ".jpg")


#TODO implement this method when I understand what to do with the MuRet IIF server
def get_image_fromURL(url):
    pass
