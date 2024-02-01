from vllm import LLM, SamplingParams
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--prompt_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="")


args = parser.parse_args()

with open(args.prompt_path, "r") as f:
    data_list = json.load(f)
    prompts = [data["prompt"] for data in data_list]

prompt_template = "{}"

prompts = [prompt_template.format(prompt.strip()) for prompt in prompts]

params_dict = {
    "n": 1,
    "best_of": None,
    "presence_penalty": 1.0,    
    "frequency_penalty": 0.0,
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1.0,
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
for data in data_list:
    prompt = data["prompt"]
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
        data['cpm_new'] = generated_text
        with open(args.prompt_path+args.output_path, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))
            f.write(",\n")

