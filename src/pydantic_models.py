"""Pydantic models shared across the pipeline: prompts, functions, results."""

from enum import Enum
from pydantic import BaseModel, field_validator, model_validator
from typing import Any


class ParamType(str, Enum):
    """The 4 types actually handled by constrained_decoding.generate_value().

    Both python-style ("int", "str", ...) and JSON-schema-style
    ("integer", "string", ...) spellings are accepted, since
    functions_definition.json may use either.
    """

    INT = "int"
    INTEGER = "integer"
    FLOAT = "float"
    NUMBER = "number"
    STR = "str"
    STRING = "string"
    BOOL = "bool"
    BOOLEAN = "boolean"

    def __str__(self) -> str:
        # Without this, str(ParamType.INT) is "ParamType.INT" (Enum's
        # default), which would leak into the prompt text built in
        # pipeline._base_prompt() via f"{k}: {v.type}".
        return self.value


class PromptItem(BaseModel):
    """A single natural-language prompt from the input file."""

    prompt: str


class FunctionParameter(BaseModel):
    """Describes one parameter of a function (its JSON type)."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """Full definition of a callable function."""

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: dict[str, Any]

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Function 'name' must not be empty")
        return v

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Function 'description' must not be empty")
        return v

    @field_validator("returns")
    @classmethod
    def returns_type_is_supported(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "type" not in v:
            raise ValueError("Function 'returns' must specify a 'type'")
        try:
            ParamType(v["type"])
        except ValueError:
            raise ValueError(
                f"Unsupported return type {v['type']!r}: must be one of "
                "int, number, str or bool"
            ) from None
        return v

    @model_validator(mode="after")
    def parameters_not_blank(self) -> "FunctionDefinition":
        for param_name in self.parameters:
            if not param_name.strip():
                raise ValueError("Parameter names must not be empty")
        return self


class FunctionCallResult(BaseModel):
    """Result of processing one prompt: chosen function + extracted args."""

    prompt: str
    name: str
    parameters: dict[str, Any]
