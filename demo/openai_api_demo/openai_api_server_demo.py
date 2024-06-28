import gc
import json
import os
import time
from threading import Thread

import tiktoken
import torch
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from loguru import logger
from sse_starlette.sse import EventSourceResponse

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

MODEL_PATH = os.environ.get('MODEL_PATH', 'openbmb/MiniCPM-2B-dpo-fp16')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-m3')


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # clean cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(
        id="MiniCPM-2B"
    )
    return ModelList(
        data=[model_card]
    )


def generate_minicpm(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(inputs, return_tensors="pt").to(model.device)
    input_echo_len = len(enc["input_ids"][0])

    if input_echo_len >= model.config.max_length:
        logger.error(f"Input length larger than {model.config.max_length}")
        return
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = {
        **enc,
        "do_sample": True if temperature > 1e-5 else False,
        "top_k": 0,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    eos_token = tokenizer.eos_token
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    response = ""
    for new_text in streamer:
        new_text = new_text.split(eos_token)[0] if eos_token in new_text else new_text
        response += new_text
        current_length = len(new_text)
        yield {
            "text": response[5 + len(inputs):],
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": current_length - input_echo_len,
                "total_tokens": len(response),
            },
            "finish_reason": "",
        }
    thread.join()
    gc.collect()
    torch.cuda.empty_cache()


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    embeddings = [embedding_model.encode(text) for text in request.input]
    embeddings = [embedding.tolist() for embedding in embeddings]

    def num_tokens_from_string(string: str) -> int:
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    response = {
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": CompletionUsage(
            prompt_tokens=sum(len(text.split()) for text in request.input),
            completion_tokens=0,
            total_tokens=sum(num_tokens_from_string(text) for text in request.input),
        )
    }
    return response


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 2048,
        echo=False,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
    )
    logger.debug(f"==== request ====\n{gen_params}")
    input_tokens = sum(len(tokenizer.encode(msg.content)) for msg in request.messages)
    if request.stream:
        async def stream_response():
            previous_text = ""
            for new_response in generate_minicpm(model, tokenizer, gen_params):
                delta_text = new_response["text"][len(previous_text):]
                previous_text = new_response["text"]
                delta = DeltaMessage(content=delta_text, role="assistant")
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None
                )
                chunk = {
                    "model": request.model,
                    "id": "",
                    "choices": [choice_data.dict(exclude_none=True)],
                    "object": "chat.completion.chunk"
                }
                yield json.dumps(chunk) + "\n"

        return EventSourceResponse(stream_response(), media_type="text/event-stream")

    else:
        generated_text = ""
        for response in generate_minicpm(model, tokenizer, gen_params):
            generated_text = response["text"]
        generated_text = generated_text.strip()
        output_tokens = len(tokenizer.encode(generated_text))
        usage = UsageInfo(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=output_tokens + input_tokens
        )
        message = ChatMessage(role="assistant", content=generated_text)
        logger.debug(f"==== message ====\n{message}")
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop",
        )
        return ChatCompletionResponse(
            model=request.model,
            id="",
            choices=[choice_data],
            object="chat.completion",
            usage=usage
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
                                                 trust_remote_code=True)
    embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cuda")
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
