from pydantic import BaseModel
from typing import List, Dict

class BasicMessage(BaseModel):
    message:str

class ListMessage(BaseModel):
    message:List[dict]

class RegionsResponse(BaseModel):
    regions:List[dict]

