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
        function="fn_add_numbers",
        arguments={"a": 2, "b": 3},
    )

    assert result.arguments == {"a": 2, "b": 3}


def test_function_call_result_rejects_missing_arguments() -> None:
    with pytest.raises(ValidationError):
        FunctionCallResult.model_validate({"function": "fn_add_numbers"})
