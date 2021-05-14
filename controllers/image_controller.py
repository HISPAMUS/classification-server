from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from output_messages.output import BasicMessage
from pathlib import Path
import cv2
import numpy as np
import requests
from httpx import AsyncClient
import asyncio
import io
import base64
image_storage_path = "images/"

from logger import Logger

logger_term = Logger()

router = APIRouter()

img_height = 40
img_width = 40
img_pos_width = 112
img_pos_height = 224

client = AsyncClient()

@router.post('/image', response_model=BasicMessage)
async def save_image(id:str = Form(...), image:UploadFile = File(...)):
    imagecontents = await image.read()
    with open(image_storage_path + id + ".jpg", "wb") as destiny_file:
        destiny_file.write(imagecontents)
            
    return BasicMessage(message = f'Image {id} has been stored correctly')

@router.post('/image_url', response_model=BasicMessage)
async def save_image_url(id:str = Form(...), url:str = Form(...)):
    return BasicMessage(message=f"Image {id} has been stored correctly")

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

def read(id, left, top, right, bottom, crop):
    try:
        image = read_simple(id)
        return image[top:bottom, left:right]
    except Exception as e:
        logger_term.LogInfo(e)

def read_simple(id):
    return cv2.imread(image_storage_path + id + ".jpg")


#TODO implement this method when I understand what to do with the MuRet IIF server
async def get_image_fromURL(url, left, top, right, bottom, crop):
    response = None
    response = await client.get(url)
    logger_term.LogInfo(f"Encoding: {response}")
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #logger_term.LogInfo(np.array(image))
    logger_term.LogInfo(image.shape)
    if crop:
        return image[top:bottom, left:right]
    return image


def crop(id, left, top, right, bottom):
        image = cv2.imread(image_storage_path + id + ".jpg")

        shape_image = image[top:bottom, left:right]
        #cv2.imwrite('debug_shape.png', shape_image)
        shape_image = [cv2.resize(shape_image, (img_width, img_height))]
        shape_image = np.asarray(shape_image).reshape(1, img_height, img_width, 3)
        shape_image = (255. - shape_image) / 255.

        # Position [mirror effect for boxes close to the limits]
        image_height, image_width, _  = image.shape

        center_x = left + (right - left) / 2
        center_y = top + (bottom - top) / 2

        pos_left = int(max(0, center_x - img_pos_width / 2))
        pos_right = int(min(image_width, center_x + img_pos_width / 2))
        pos_top = int(max(0, center_y - img_pos_height / 2))
        pos_bottom = int(min(image_height, center_y + img_pos_height / 2))

        pad_left = int(abs(min(0, center_x - img_pos_width / 2)))
        pad_right = int(abs(min(0, image_width - (center_x + img_pos_width / 2))))
        pad_top = int(abs(min(0, center_y - img_pos_height / 2)))
        pad_bottom = int(abs(min(0, image_height - (center_y + img_pos_height / 2))))

        position_image = image[pos_top:pos_bottom, pos_left:pos_right]
        position_image = np.stack(
            [np.pad(position_image[:, :, c],
                    [(pad_top, pad_bottom), (pad_left, pad_right)],
                    mode='symmetric')
             for c in range(3)], axis=2)

        #cv2.imwrite('debug_position.png', position_image)

        position_image = np.asarray(position_image).reshape(1, img_pos_height, img_pos_width, 3)
        position_image = (255. - position_image) / 255.

        return (shape_image, position_image)
    
def crop_from_image(image, left, top, right, bottom):
        #image = cv2.imread(image_storage_path + id + ".jpg")

        shape_image = image[top:bottom, left:right]
        #cv2.imwrite('debug_shape.png', shape_image)
        shape_image = [cv2.resize(shape_image, (img_width, img_height))]
        shape_image = np.asarray(shape_image).reshape(1, img_height, img_width, 3)
        shape_image = (255. - shape_image) / 255.

        # Position [mirror effect for boxes close to the limits]
        image_height, image_width, _  = image.shape

        center_x = left + (right - left) / 2
        center_y = top + (bottom - top) / 2

        pos_left = int(max(0, center_x - img_pos_width / 2))
        pos_right = int(min(image_width, center_x + img_pos_width / 2))
        pos_top = int(max(0, center_y - img_pos_height / 2))
        pos_bottom = int(min(image_height, center_y + img_pos_height / 2))

        pad_left = int(abs(min(0, center_x - img_pos_width / 2)))
        pad_right = int(abs(min(0, image_width - (center_x + img_pos_width / 2))))
        pad_top = int(abs(min(0, center_y - img_pos_height / 2)))
        pad_bottom = int(abs(min(0, image_height - (center_y + img_pos_height / 2))))

        position_image = image[pos_top:pos_bottom, pos_left:pos_right]
        position_image = np.stack(
            [np.pad(position_image[:, :, c],
                    [(pad_top, pad_bottom), (pad_left, pad_right)],
                    mode='symmetric')
             for c in range(3)], axis=2)

        #cv2.imwrite('debug_position.png', position_image)

        position_image = np.asarray(position_image).reshape(1, img_pos_height, img_pos_width, 3)
        position_image = (255. - position_image) / 255.

        return (shape_image, position_image)
