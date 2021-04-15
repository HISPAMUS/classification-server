from pydantic import BaseModel
from typing import List, Dict

class BasicMessage(BaseModel):
    message:str

class ListMessage(BaseModel):
    message:List[dict]

class RegionsResponse(BaseModel):
    regions:List[dict]

class SymbolsResponse(BaseModel):
    shape:List[str]
    position:List[str]

class TranslationResponse(BaseModel):
    semantic:str

