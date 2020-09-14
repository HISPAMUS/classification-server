from pydantic import BaseModel
from typing import List

class BasicMessage(BaseModel):
    message:str

class ListMessage(BaseModel):
    message:List[dict]
