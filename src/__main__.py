from llm_sdk.llm_sdk import Small_LLM_Model
from .demo_llm import run_demo


def main() -> None:
    # model = Small_LLM_Model()

    # prompt = "What is the sum of 2 and 3?"
    # inputs_ids = model.encode(prompt)[0].tolist()

    # logits = model.get_logits_from_input_ids(inputs_ids)
    # next_token_id = max(range(len(logits)), key=lambda i: logits[i])

    # generated_ids = inputs_ids + [next_token_id]
    # text = model.decode(generated_ids)

    # print("Prompt:", prompt)
    # print("Next token id:", next_token_id)
    # print("Generated text:", text)
    run_demo()


if __name__ == "__main__":
    main()
