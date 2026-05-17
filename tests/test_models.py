import pytest
from pydantic import ValidationError

from src.pydantic_models import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionParameter,
    PromptItem,
)


def test_prompt_item_validation() -> None:
    item = PromptItem(prompt="What is the sum of 2 and 3?")
    assert item.prompt == "What is the sum of 2 and 3?"


def test_function_definition_validation() -> None:
    definition = FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers.",
        parameters={
            "a": FunctionParameter(type="number"),
            "b": FunctionParameter(type="number"),
        },
        returns={"type": "number"},
    )
    assert definition.name == "fn_add_numbers"
    assert definition.parameters["a"].type == "number"


def test_function_call_result_validation() -> None:
    result = FunctionCallResult(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={
            "a": 2.0,
            "b": 3.0
        },
    )
    assert result.name == "fn_add_numbers"
    assert result.parameters == {"a": 2.0, "b": 3.0}


def test_function_call_result_rejects_missing_name() -> None:
    with pytest.raises(ValidationError):
        FunctionCallResult.model_validate({
            "prompt": "hello",
            "parameters": {}
        })
