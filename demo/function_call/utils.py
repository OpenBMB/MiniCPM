import ast
import json
import keyword
import traceback
import uuid
from collections import deque
from copy import deepcopy
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

from datamodel_code_generator import DataModelType
from datamodel_code_generator.format import PythonVersion
from datamodel_code_generator.model import get_data_model_types
from datamodel_code_generator.parser.jsonschema import JsonSchemaParser
from jsonschema import Draft202012Validator, exceptions, validate

from transformers import LlamaTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import TensorType


logger = getLogger(__name__)


def decode_function_call(
    sequence: str,
    tool_call_start="<|tool_call_start|>",
    tool_call_end="<|tool_call_end|>",
    thought_start="<|thought_start|>",
    thought_end="<|thought_end|>",
):
    if thought_end in sequence and thought_start in sequence:
        thought_string, sequence = sequence.rsplit(thought_end, 1)
        thought_string = thought_string.split(thought_start, 1)[1]
    else:
        thought_string = ""
    if tool_call_start in sequence and tool_call_end in sequence:
        tool_call_string, content = sequence.rsplit(tool_call_end, 1)
        tool_call_string = tool_call_string.split(tool_call_start, 1)[1]
        try:
            tool_calls = []
            tool_call_string = tool_call_string.strip()
            if tool_call_string.startswith("```"):
                tool_call_string = tool_call_string.lstrip("```").strip()
                if tool_call_string.startswith("python"):
                    tool_call_string = tool_call_string.lstrip("python").strip()
            if tool_call_string.endswith("```"):
                tool_call_string = tool_call_string.rstrip("```").strip()
            for kw in keyword.kwlist:
                tool_call_string = tool_call_string.replace(
                    "," + kw + "=", "," + kw + "_="
                )
                tool_call_string = tool_call_string.replace(
                    " " + kw + "=", " " + kw + "_="
                )
                tool_call_string = tool_call_string.replace(
                    "(" + kw + "=", "(" + kw + "_="
                )

            parsed = ast.parse(tool_call_string)

            for elem in parsed.body:
                assert isinstance(elem.value, ast.Call)
                calls = resolve_ast_call(elem.value)

                for func_name, func_args in calls.items():
                    new_args = {}
                    for k, v in func_args.items():
                        for kw in keyword.kwlist:
                            if k == kw + "_":
                                k = kw
                        new_args[k] = v

                    this_one = {"name": func_name, "arguments": new_args}
                    tool_calls.append(this_one)

            return {
                "content": content.strip(),
                "tool_calls": [
                    {
                        "type": "function",
                        "function": tool_call,
                        "id": "call_" + uuid.uuid4().hex,
                    }
                    for tool_call in tool_calls
                ],
                "role": "assistant",
            }
        except:
            logger.error(traceback.format_exc())
            return {
                "content": content.strip(),
                "role": "assistant",
                "thought": thought_string,
            }
    else:
        return {
            "content": sequence.strip(),
            "role": "assistant",
            "thought": thought_string,
        }


def check_messages(conversation: List[Dict[str, str]], tools: List[Dict]):
    if tools is not None:
        for tool in tools:
            if "type" not in tool or tool["type"] != "function":
                raise ValueError(f"Tool {tool} is not valid")
            if "name" not in tool["function"]:
                raise ValueError(f"Tool {tool} is not valid")
            if "parameters" not in tool["function"] or not check_tool(
                tool["function"]["parameters"]["properties"]
            ):
                raise ValueError(f"Tool {tool} is not valid")
    for message in conversation:
        if (
            message["role"] == "assistant"
            and "tool_calls" in message
            and len(message["tool_calls"]) > 0
        ):
            for tool_call in message["tool_calls"]:
                if "id" not in tool_call:
                    raise ValueError(f"Tool call {tool_call} is not valid")
                if tool_call["type"] != "function":
                    raise ValueError(f"Tool call {tool_call} is not valid")
                if "function" not in tool_call:
                    raise ValueError(f"Tool call {tool_call} is not valid")
                if not check_tool(tool_call["function"]):
                    raise ValueError(
                        f"Tool call function {tool_call['function']} is not valid"
                    )
        elif message["role"] == "tool":
            if "tool_call_id" not in message:
                raise ValueError(f"Tool message {message['content']} is not valid")


def check_tool(tool_schema):
    try:
        Draft202012Validator.check_schema(tool_schema)
        return True
    except exceptions.SchemaError as e:
        print(f"SchemaError: {e}")
        return False


def check_args(args, tool_schema):
    try:
        validate(instance=args, schema=tool_schema)
        return True
    except exceptions.ValidationError as e:
        print(f"Data failed validation: {e}")
        return False


def message_format(msg, system_suffix="", user_prefix=""):
    if "thought" in msg and msg["thought"] is not None and len(msg["thought"]) > 0:
        thought_prefix = f"<|thought_start|>\n{msg['thought']}\n<|thought_end|>\n"
    else:
        thought_prefix = ""
    if msg["role"] == "assistant":
        content = msg.get("content", "")
        if content is None:
            content = ""
        if (
            "tool_calls" in msg
            and msg["tool_calls"] is not None
            and len(msg["tool_calls"]) > 0
        ):

            def add_quotes(variable):
                if isinstance(variable, str):
                    return repr(variable)
                else:
                    return str(variable)

            tool_calls = []
            for _tool_call in msg["tool_calls"]:
                if _tool_call is None:
                    continue
                tool_call = _tool_call["function"]
                tool_name = tool_call["name"]
                if "arguments" not in tool_call or tool_call["arguments"] is None:
                    continue
                if isinstance(tool_call["arguments"], str):
                    try:
                        tool_call["arguments"] = json.loads(tool_call["arguments"])
                    except:
                        continue
                args = ",".join(
                    [k + "=" + add_quotes(v) for k, v in tool_call["arguments"].items()]
                )
                tool_calls.append(f"{tool_name}({args})")

            content = (
                thought_prefix
                + "<|tool_call_start|>\n```python\n"
                + "\n".join(tool_calls).strip()
                + "\n```\n<|tool_call_end|>\n"
                + content
            )
            # msg["tool_call_string"] = "\n".join(tool_calls).strip()
            msg["content"] = content
        else:
            content = thought_prefix + content
            msg["content"] = content
    elif msg["role"] == "user":
        msg["content"] = user_prefix + "\n" + msg["content"]
    elif msg["role"] == "system":
        msg["content"] = msg["content"] + "\n" + system_suffix
    msg["content"] = msg["content"].strip()
    return msg


def jsonschema_to_code(jsonschema: dict) -> str:
    input_text = json.dumps(jsonschema)
    data_model_types = get_data_model_types(
        DataModelType.PydanticBaseModel,
        PythonVersion.PY_310,
    )
    parser = JsonSchemaParser(
        source=input_text,
        data_model_type=data_model_types.data_model,
        data_model_root_type=data_model_types.root_model,
        data_model_field_type=data_model_types.field_model,
        data_type_manager_type=data_model_types.data_type_manager,
        target_python_version=PythonVersion.PY_310,
        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
        field_constraints=True,
    )
    results = parser.parse()
    return results


def transform_function(function: dict):
    """turn json format of function into signature"""
    params, default_params = [], []
    for prop_name, prop in function["parameters"]["properties"].items():
        if "default" in prop:
            default_params.append(f'{prop_name}={repr(prop["default"])}')
        elif prop_name not in function["parameters"].get("required", []):
            default_params.append(f"{prop_name}={repr(None)}")
        else:
            params.append(prop_name)
    ps = ", ".join(params + default_params)
    res = "def {f_name}({ps}):\n".format(f_name=function["name"], ps=ps)
    f_des = function.get("description", "")
    content = jsonschema_to_code(function["parameters"])
    if "class" in content:
        i = content.index("class")
        # print(content[:i])
        content = content[i:]
    classes, args = content.split("class Model(BaseModel):", 1)
    lint_msg = f'    """\n    {f_des}\n    Args:\n{args}\n    """\n'
    res += lint_msg
    if len(classes) > 0:
        res = classes + res
    return res


def input_format(messages: List[Dict], tools: List[Dict], add_to_system=True):
    """
    Process the input messages, global_arguments, tools, tool_choice,
        and convert it into a input string.
    The global arguments and tools can not be both empty.
    parameters:
        messages: List[Dict]
            the input messages
            For example:
        tools: List[Dict]
            the tools list you can use
            For example:
    """
    messages = deepcopy(messages)
    tools = deepcopy(tools)
    if tools is not None and len(tools) > 0:
        header = "from enum import Enum\nfrom typing import List, Dict, Optional\nfrom pydantic import BaseModel, Field\n\n"
        tools_string = header
        for tool in tools:
            try:
                tools_string += "\n\n" + transform_function(tool)
            except:
                pass
        tools_template = """# Functions
Here is a list of functions that you can invoke:
```python
{tools}
```

# Function Call Rule and Output Format
- If the user's question can be answered without calling any function, please answer the user's question directly. In this situation, you should return your thought and answer the user's question directly.
- If the user cannot be answered without calling any function, and the user does not provide enough information to call functions, please ask the user for more information. In this situation, you should return your thought and ask the user for more information.
- If the user's question cannot be answered without calling any function, and the user has provided enough information to call functions to solve it, you should call the functions. In this situation, the assistant should return your thought and call the functions.
- Use default parameters unless the user has specified otherwise.
- You should answer in the following format:

<|thought_start|>
{{explain why the user's question can be answered without calling a function or why you should ask the user for more information or why you should call one or more functions and your plan to solve the user's question.}}
<|thought_end|>
<|tool_call_start|>
```python
func1(params_name=params_value, params_name2=params_value2...)
func2(params)
```
<|tool_call_end|>
{{answer the user's question directly or ask the user for more information}}
"""
        tools_string = tools_template.format(tools=tools_string).strip()
    else:
        tools_string = ""

    if add_to_system:
        return [
            message_format(msg, system_suffix=tools_string, user_prefix="")
            for msg in messages
        ]
    else:
        return [
            message_format(msg, system_suffix="", user_prefix=tools_string)
            for msg in messages
        ]


# This is a modified version of
# https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/utils.py
# Thanks to the gorilla team for the original implementation
def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output
