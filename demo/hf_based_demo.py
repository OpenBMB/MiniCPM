from typing import List
import argparse
import gradio as gr
import torch
from threading import Thread
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-2B-dpo-fp16")
parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
args = parser.parse_args()

# init model torch dtype
torch_dtype = args.torch_dtype
if torch_dtype == "" or torch_dtype == "bfloat16":
    torch_dtype = torch.bfloat16
elif torch_dtype == "float32":
    torch_dtype = torch.float32
elif torch_dtype == "float16":
    torch_dtype = torch.float16
else:
    raise ValueError(f"Invalid torch dtype: {torch_dtype}")

# init model and tokenizer
path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, device_map="cuda:0", trust_remote_code=True)

model_architectures = model.config.architectures[0]


def check_model_v(img_file_path: str = None):
    '''
    check model is MiniCPMV
    Args:
        img_file_path (str): Image filepath

    Returns:
        Ture if model is MiniCPMV else False
    '''
    if "MiniCPMV" in model_architectures:
        return True
    if isinstance(img_file_path, str):
        gr.Warning('Only MiniCPMV model can support Image')
    return False


if check_model_v():
    model = model.to(dtype=torch.bfloat16)


# init gradio demo host and port
server_name = args.server_name
server_port = args.server_port

def hf_gen(dialog: List, top_p: float, temperature: float, repetition_penalty: float, max_dec_len: int):
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
    enc = tokenizer(inputs, return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(
        enc,
        do_sample=True,
        top_k=0,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_dec_len,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    answer = ""
    for new_text in streamer:
        answer += new_text
        yield answer[4 + len(inputs):]


def hf_v_gen(dialog: List, top_p: float, temperature: float, repetition_penalty: float, max_dec_len: int,
             img_file_path: str):
    """generate model output with huggingface api

    Args:
        query (str): actual model input.
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): Strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.
        img_file_path (str): Image filepath.

    Yields:
        str: real-time generation results of hf model
    """
    assert isinstance(img_file_path, str), 'Image must not be empty'
    img = Image.open(img_file_path).convert('RGB')

    generation_kwargs = dict(
        image=img,
        msgs=dialog,
        context=None,
        tokenizer=tokenizer,
        sampling=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_dec_len
    )
    res, context, _ = model.chat(**generation_kwargs)
    return res


def generate(chat_history: List, query: str, top_p: float, temperature: float, repetition_penalty: float, max_dec_len: int,
             img_file_path: str = None):
    """generate after hitting "submit" button

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        query (str): query of current round
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.
        img_file_path (str): Image filepath.

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
    if check_model_v():
        chat_history[-1][1] = hf_v_gen(model_input, top_p, temperature, repetition_penalty, max_dec_len, img_file_path)
        yield gr.update(value=""), chat_history
        return

    for answer in hf_gen(model_input, top_p, temperature, repetition_penalty, max_dec_len):
        chat_history[-1][1] = answer.strip("</s>")
        yield gr.update(value=""), chat_history


def regenerate(chat_history: List, top_p: float, temperature: float, repetition_penalty: float, max_dec_len: int,
               img_file_path: str = None):
    """re-generate the answer of last round's query

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.
        img_file_path (str): Image filepath.

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
    if check_model_v():
        chat_history[-1][1] = hf_v_gen(model_input, top_p, temperature, repetition_penalty, max_dec_len, img_file_path)
        yield gr.update(value=""), chat_history
        return

    for answer in hf_gen(model_input, top_p, temperature, repetition_penalty, max_dec_len):
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
            temperature = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="temperature")
            repetition_penalty = gr.Slider(0.1, 2.0, value=1.1, step=0.1, label="repetition_penalty")
            max_dec_len = gr.Slider(1, 1024, value=1024, step=1, label="max_dec_len")
            img_file_path = gr.Image(label="upload image", type='filepath', show_label=False)

        with gr.Column(scale=5):
            chatbot = gr.Chatbot(bubble_full_width=False, height=400)
            user_input = gr.Textbox(label="User", placeholder="Input your query here!", lines=8)
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")
                regen = gr.Button("Regenerate")
                reverse = gr.Button("Reverse")

    img_file_path.change(check_model_v, inputs=[img_file_path], outputs=[])

    submit.click(generate, inputs=[chatbot, user_input, top_p, temperature, repetition_penalty,
                                   max_dec_len, img_file_path], outputs=[user_input, chatbot])
    regen.click(regenerate, inputs=[chatbot, top_p, temperature, repetition_penalty,
                                    max_dec_len, img_file_path], outputs=[user_input, chatbot])
    clear.click(clear_history, inputs=[], outputs=[chatbot])
    reverse.click(reverse_last_round, inputs=[chatbot], outputs=[chatbot])

demo.queue()
demo.launch(server_name=server_name, server_port=server_port, show_error=True)
