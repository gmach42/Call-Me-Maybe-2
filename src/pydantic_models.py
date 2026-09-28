"""Pydantic models shared across the pipeline: prompts, functions, results."""

from pydantic import BaseModel, field_validator, model_validator
from typing import Any, Literal, get_args

# The types actually handled by constrained_decoding.generate_value().
# Both python-style ("int", "str", ...) and JSON-schema-style
# ("integer", "string", ...) spellings are accepted, since
# functions_definition.json may use either.
SUPPORTED_TYPES = Literal[
    "int", "integer", "float", "number", "str", "string", "bool", "boolean",
]


class PromptItem(BaseModel):
    """Pydantic model of a single natural-language prompt."""

    prompt: str


class FunctionParameter(BaseModel):
    """Pydantic model describing one parameter of a function."""

    type: SUPPORTED_TYPES


class FunctionDefinition(BaseModel):
    """Pydantic model of a full definition of a callable function."""

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: dict[str, Any]

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Reject a blank or whitespace-only function name."""
        if not v.strip():
            raise ValueError("Function 'name' must not be empty")
        return v

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        """Reject a blank or whitespace-only function description."""
        if not v.strip():
            raise ValueError("Function 'description' must not be empty")
        return v

    @field_validator("returns")
    @classmethod
    def returns_type_is_supported(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Reject a 'returns' dict missing a supported 'type'."""
        if "type" not in v:
            raise ValueError("Function 'returns' must specify a 'type'")
        if v["type"] not in get_args(SUPPORTED_TYPES):
            raise ValueError(
                f"Unsupported return type {v['type']!r}: must be one of "
                "int, number, str or bool"
            )
        return v

    @model_validator(mode="after")
    def parameters_not_blank(self) -> "FunctionDefinition":
        """Reject any parameter with a blank or whitespace-only name."""
        for param_name in self.parameters:
            if not param_name.strip():
                raise ValueError("Parameter names must not be empty")
        return self


class FunctionCallResult(BaseModel):
    """Pydantic model of the result of processing one prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any]
