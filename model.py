from pydantic import BaseModel


class VideoURL(BaseModel):
    url: str


class AnalyzeRequest(BaseModel):
    text: str
