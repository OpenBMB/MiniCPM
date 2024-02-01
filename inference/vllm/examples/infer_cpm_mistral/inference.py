import argparse

from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default="")

args = parser.parse_args()

# Sample prompts.
prompts = [
    "北京烤鸭真好吃，正好有一家北京烤鸭店，我想去吃北京烤鸭",
    "def reverse_list(list):",
    "Beijing is the capital of",
    "1 + 1 = ",
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    '''0123456789'''*810 + " what is next number?",
    "您好，三加二等于多少？",
    "我是",
    ""
]

params_dict = {
    "n": 1,
    "best_of": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 1,
    "top_p": 0.9,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1.0,
    "early_stopping": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 100,
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
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
