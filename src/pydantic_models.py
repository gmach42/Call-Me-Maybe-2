from pydantic import BaseModel
from typing import Any


class PromptItem(BaseModel):
    """A single natural-language prompt from the input file."""

    prompt: str


class FunctionParameter(BaseModel):
    """Describes one parameter of a function (its JSON type)."""

    type: str


class FunctionDefinition(BaseModel):
    """Full definition of a callable function."""

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: dict[str, Any]


class FunctionCallResult(BaseModel):
    """Result of processing one prompt: chosen function + extracted args."""

    prompt: str
    name: str
    parameters: dict[str, Any]
