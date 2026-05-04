"""Simple examples showing how to interact with the local LLM SDK."""

from __future__ import annotations

from llm_sdk.llm_sdk import Small_LLM_Model


def greedy_next_token_id(logits: list[float]) -> int:
    """Return the id of the most likely next token."""
    return max(range(len(logits)), key=logits.__getitem__)


def generate_text(
    model: Small_LLM_Model,
    prompt: str,
    max_new_tokens: int = 20,
) -> str:
    """Generate a short greedy completion for a given prompt."""
    token_ids = model.encode(prompt)[0].tolist()

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(token_ids)
        next_token_id = greedy_next_token_id(logits)
        token_ids.append(next_token_id)

    return model.decode(token_ids)


def run_demo() -> None:
    """Run a few simple prompts against the LLM."""
    model = Small_LLM_Model()

    prompts = [
        "Say hello to Alice.",
        "What is 2 + 3?",
        "Explain what a token is in one short sentence.",
    ]

    for prompt in prompts:
        response = generate_text(model, prompt)
        print("=" * 60)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")


if __name__ == "__main__":
    run_demo()
