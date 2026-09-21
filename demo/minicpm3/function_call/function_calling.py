#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from minicpm_tool_parser import fc2dict
import json

model_path = "openbmb/MiniCPM3-4B"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    }
]
messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    },
    {
        "role": "user",
        "content": "Hi, can you tell me the delivery date for my order? The order id is 1234 and 4321.",
    },
    # {
    #    "content": "",
    #    "tool_calls": [
    #        {
    #            "type": "function",
    #            "function": {
    #                "name": "get_delivery_date",
    #                "arguments": {"order_id": "1234"},
    #            },
    #            "id": "call_b4ab0b4ec4b5442e86f017fe0385e22e",
    #        },
    #        {
    #            "type": "function",
    #            "function": {
    #                "name": "get_delivery_date",
    #                "arguments": {"order_id": "4321"},
    #            },
    #            "id": "call_628965479dd84794bbb72ab9bdda0c39",
    #        },
    #    ],
    #    "role": "assistant",
    # },
    # {
    #    "role": "tool",
    #    "content": '{"delivery_date": "2024-09-05", "order_id": "1234"}',
    #    "tool_call_id": "call_b4ab0b4ec4b5442e86f017fe0385e22e",
    # },
    # {
    #    "role": "tool",
    #    "content": '{"delivery_date": "2024-09-05", "order_id": "4321"}',
    #    "tool_call_id": "call_628965479dd84794bbb72ab9bdda0c39",
    # },
    # {
    #    "content": "Both your orders will be delivered on 2024-09-05.",
    #    "role": "assistant",
    #    "thought": "\nI have the information you need, both orders will be delivered on the same date, 2024-09-05.\n",
    # },
]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True
)
llm = LLM(model_path, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000)


def fake_tool_execute(toolcall):
    data = {
        "delivery_date": "2024-09-05",
        "order_id": toolcall.get("function", {})
        .get("arguments", {})
        .get("order_id", "order_id"),
    }
    return json.dumps(data)


while True:
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text
    msg = fc2dict(response)
    if (
        "tool_calls" in msg
        and msg["tool_calls"] is not None
        and len(msg["tool_calls"]) > 0
    ):
        messages.append(msg)
        print(msg)
        for toolcall in msg["tool_calls"]:
            tool_response = fake_tool_execute(toolcall)
            tool_msg = {
                "role": "tool",
                "content": tool_response,
                "tool_call_id": toolcall["id"],
            }
            messages.append(tool_msg)
            print(tool_msg)
    else:
        messages.append(msg)
        print(msg)
        break
