from typing import Dict
from typing import List
from typing import Tuple

import argparse
import gradio as gr
import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TextIteratorStreamer
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="")
args = parser.parse_args()

# init model and tokenizer
path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)


def hf_gen(dialog: List, top_p: float, temperature: float, max_dec_len: int):
    """generate model output with huggingface api

    Args:
        query (str): actual model input.
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): Strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        str: real-time generation results of hf model
    """    
    inputs = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(inputs, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(
        enc,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_dec_len,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    answer = ""
    for new_text in streamer:
        answer += new_text
        yield answer[4 + len(inputs):]


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
    chat_history.append([query, ""])
    for answer in hf_gen(model_input, top_p, temperature, max_dec_len):
        chat_history[-1][1] = answer.strip("</s>")
        yield gr.update(value=""), chat_history


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
    for answer in hf_gen(model_input, top_p, temperature, max_dec_len):
        chat_history[-1][1] = answer.strip("</s>")
        yield gr.update(value=""), chat_history


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
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="temperature")
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
demo.launch(server_name="127.0.0.1", show_error=True)
