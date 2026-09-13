"""Transcript-related schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default="en", description="Spoken language code")


class TranscriptionResponse(BaseModel):
    text: str
    filename: Optional[str] = None
    language: Optional[str] = "en"
    duration: Optional[float] = None
    segments: List[Segment] = Field(default_factory=list)
