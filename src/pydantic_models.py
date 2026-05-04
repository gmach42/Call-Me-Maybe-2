from pydantic import BaseModel, Field
from typing import Any


class PromptItem(BaseModel):
    prompt: str


class FunctionParameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: dict[str, Any]


class FunctionCallResult(BaseModel):
    function: str = Field(..., description="Chosen function name")
    arguments: dict[str, Any]
