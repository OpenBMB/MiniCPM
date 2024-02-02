from typing import Dict
from typing import List
from typing import Tuple

import argparse
import gradio as gr
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
args = parser.parse_args()


# init model torch dtype
torch_dtype = args.torch_dtype
if torch_dtype =="" or torch_dtype == "bfloat16":
    torch_dtype = "bfloat16"
elif torch_dtype == "float32":
    torch_dtype = "float32"
else:
    raise ValueError(f"Invalid torch dtype: {torch_dtype}")

# init model and tokenizer
path = args.model_path
llm = LLM(model=path, tensor_parallel_size=1, dtype=torch_dtype)

# init gradio demo host and port
server_name=args.server_name
server_port=args.server_port

def vllm_gen(dialog: List, top_p: float, temperature: float, max_dec_len: int):
    """generate model output with huggingface api

    Args:
        query (str): actual model input.
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): Strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        str: real-time generation results of hf model
    """    
    prompt = ""
    assert len(dialog) % 2 == 1
    for info in dialog:
        if info["role"] == "user":
            prompt += "<用户>" + info["content"]
        else:
            prompt += "<AI>" + info["content"]
    prompt += "<AI>"
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,    
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop": None,
        "stop_token_ids": None,
        "ignore_eos": False,
        "max_tokens": max_dec_len,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)[0]
    generated_text = outputs.outputs[0].text
    return generated_text


def generate(chat_history: List, query: str, top_p: float, temperature: float, max_dec_len: int):
    """generate after hitting "submit" button

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        query (str): query of current round
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n], [q_n+1, a_n+1]]. chat_history + QA of current round.
    """    
    assert query != "", "Input must not be empty!!!"
    # apply chat template
    model_input = []
    for q, a in chat_history:
        model_input.append({"role": "user", "content": q})
        model_input.append({"role": "assistant", "content": a})
    model_input.append({"role": "user", "content": query})
    # yield model generation
    model_output = vllm_gen(model_input, top_p, temperature, max_dec_len)
    chat_history.append([query, model_output])
    return gr.update(value=""), chat_history


def regenerate(chat_history: List, top_p: float, temperature: float, max_dec_len: int):
    """re-generate the answer of last round's query

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. chat_history
    """    
    assert len(chat_history) >= 1, "History is empty. Nothing to regenerate!!"
    # apply chat template
    model_input = []
    for q, a in chat_history[:-1]:
        model_input.append({"role": "user", "content": q})
        model_input.append({"role": "assistant", "content": a})
    model_input.append({"role": "user", "content": chat_history[-1][0]})
    # yield model generation
    model_output = vllm_gen(model_input, top_p, temperature, max_dec_len)
    chat_history[-1][1] = model_output
    return gr.update(value=""), chat_history


def clear_history():
    """clear all chat history

    Returns:
        List: empty chat history
    """    
    return []


def reverse_last_round(chat_history):
    """reverse last round QA and keep the chat history before

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records

    Returns:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n-1, a_n-1]]. chat_history without last round.
    """    
    assert len(chat_history) >= 1, "History is empty. Nothing to reverse!!"
    return chat_history[:-1]


# launch gradio demo
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""# MiniCPM Gradio Demo""")

    with gr.Row():
        with gr.Column(scale=1):
            top_p = gr.Slider(0, 1, value=0.8, step=0.1, label="top_p")
            temperature = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="temperature")
            max_dec_len = gr.Slider(1, 1024, value=1024, step=1, label="max_dec_len")
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(bubble_full_width=False, height=400)
            user_input = gr.Textbox(label="User", placeholder="Input your query here!", lines=8)
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")
                regen = gr.Button("Regenerate")
                reverse = gr.Button("Reverse")

    submit.click(generate, inputs=[chatbot, user_input, top_p, temperature, max_dec_len], outputs=[user_input, chatbot])
    regen.click(regenerate, inputs=[chatbot, top_p, temperature, max_dec_len], outputs=[user_input, chatbot])
    clear.click(clear_history, inputs=[], outputs=[chatbot])
    reverse.click(reverse_last_round, inputs=[chatbot], outputs=[chatbot])

demo.queue()
demo.launch(server_name=server_name, server_port=server_port, show_error=True)
