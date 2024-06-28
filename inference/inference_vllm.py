from vllm import LLM, SamplingParams
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--prompt_path", type=str, default="")


args = parser.parse_args()

with open(args.prompt_path, "r") as f:
    prompts = f.readlines()

prompt_template = "<用户>{}<AI>"

prompts = [prompt_template.format(prompt.strip()) for prompt in prompts]

params_dict = {
    "n": 1,
    "best_of": 1,
    "presence_penalty": 1.0,    
    "frequency_penalty": 0.0,
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1,
    "early_stopping": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 1000,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True,
}

# Create a sampling params object.
sampling_params = SamplingParams(**params_dict)

# Create an LLM.
llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype='bfloat16')
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
for prompt in prompts:
    outputs = llm.generate(prompt, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("================")
        # find the first <用户> and remove the text before it.
        clean_prompt = prompt[prompt.find("<用户>")+len("<用户>"):]

        print(f"""<用户>: {clean_prompt.replace("<AI>", "")}""")
        print(f"<AI>:")
        print(generated_text)
