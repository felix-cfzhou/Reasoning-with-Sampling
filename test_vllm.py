from vllm import LLM, SamplingParams


def main():
    # Use a tiny model to save time and memory
    model_name = "facebook/opt-125m"

    prompts = ["The future of AI is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

    # This step initializes the engine and allocates the KV cache
    llm = LLM(model=model_name)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {generated_text!r}")


if __name__ == "__main__":
    main()
