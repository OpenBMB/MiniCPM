import openai
import json
import keyword
import ast
import uuid
from transformers import AutoTokenizer
from argparse import ArgumentParser

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
        output = "..." if value.value is Ellipsis else value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value  # type: ignore
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
            value,
            ast.NameConstant):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
            value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = ast.literal_eval(ast.unparse(value))  # type: ignore
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)  # type: ignore
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = ast.literal_eval(
            ast.unparse(  # type: ignore
                value.body[0].value))  # type: ignore
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)  # type: ignore
        except Exception as e:
            output = (
                ast.unparse(value.value) + "[" +  # type: ignore
                ast.unparse(value.slice) + "]")  # type: ignore
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def parse_tool_for_minicpm3(
    sequence: str,
    tool_call_start="<|tool_call_start|>",
    tool_call_end="<|tool_call_end|>",
):
    try:
        if tool_call_start in sequence and tool_call_end in sequence:
            tool_call_string, content = sequence.rsplit(tool_call_end, 1)
            tool_call_string = tool_call_string.split(tool_call_start, 1)[1]
            tool_calls = []
            tool_call_string = tool_call_string.strip()
            if tool_call_string.startswith("```"):
                tool_call_string = tool_call_string[3:].strip()
                if tool_call_string.startswith("python"):
                    tool_call_string = tool_call_string.lstrip(
                        "python").strip()
            if tool_call_string.endswith("```"):
                tool_call_string = tool_call_string[:-3].strip()
            for kw in keyword.kwlist:
                tool_call_string = tool_call_string.replace(
                    "," + kw + "=", "," + kw + "_=")
                tool_call_string = tool_call_string.replace(
                    " " + kw + "=", " " + kw + "_=")
                tool_call_string = tool_call_string.replace(
                    "(" + kw + "=", "(" + kw + "_=")
            need_replace = False
            replaced_tool_call_string = tool_call_string.replace("-","_")
            if replaced_tool_call_string != tool_call_string:
                need_replace = True
                tool_call_string = replaced_tool_call_string
            parsed: ast.Module = ast.parse(tool_call_string)

            for elem in parsed.body:
                assert isinstance(elem.value, ast.Call)  # type: ignore
                calls = resolve_ast_call(elem.value)  # type: ignore

                for func_name, func_args in calls.items():
                    new_args = {}
                    for k, v in func_args.items():
                        for kw in keyword.kwlist:
                            if k == kw + "_":
                                k = kw
                        new_args[k] = v

                    this_one = {"name": func_name, "arguments": new_args}
                    tool_calls.append({ "id":str(uuid.uuid4()),"function":this_one,"type":"function"})
            if need_replace:
                for tool_call in tool_calls:
                    tool_call["function"]["name"] = tool_call["function"]["name"].replace("_","-")
            return tool_calls
        else:
            return []
    except:
        return []

def generate(
        client,
        tokenizer,
        model : str,
        messages_minicpm : list,
        tools : list
    ):
    prompt = tokenizer.apply_chat_template(
        messages_minicpm, tools=tools, tokenize=False, add_generation_prompt=True
    )

    response = client.completions.create(model=model,prompt = prompt,max_tokens = 8192)

    response_dict = response.model_dump()


    first_choice = response_dict['choices'][0]            
    
    return first_choice["text"]

example_messages_history = [
    {
      "role": "system",
      "content": "You are an intelligent assistant with access to various tools. Your task is to answer questions by using these tools when needed.\n\nCRITICAL INSTRUCTIONS:\n\n1. Tool use is expected and encouraged, especially when information cannot be inferred from the conversation context. However, if you have gathered enough information to confidently provide the final answer, you may do so directly — but only after tool usage has been attempted or proven unnecessary.\n\n2. DO NOT describe or talk about using tools — actually CALL them using the tool_calls mechanism.\n\n3. NEVER fabricate answers. Always rely on tool results or clearly indicate when no useful result is available.\n\n4. If a tool returns an error OR fails to provide useful or new information (e.g., empty results, no content, or repeated output), DO NOT call it again with the same inputs. Avoid repeating the same failed tool calls. If a tool fails, try alternative tools if available.\n\n5. You MUST consider previous tool_calls and tool responses when deciding what to do next. Use this history to avoid redundant or circular behavior.\n\n6. If ALL relevant tools have been tried and none provide helpful results, you may gracefully end the conversation with a best-effort response, acknowledging that tools did not yield a definitive answer.\n\n7. When delivering the final answer, use the following format:\n   - First provide a concise analysis or summary of your reasoning and tool findings.\n   - Then end with: **\"The answer is: [your final answer]\"**\n\nTECHNICAL DETAILS:\n\n- For any step involving tool use, your response must include a \"tool_calls\" field.\n- The only valid response without tool_calls is when delivering the FINAL ANSWER after attempting or ruling out tool usage.\n\nEXAMPLES OF ACCEPTABLE BEHAVIOR:\n- Trying a tool, analyzing the response, and choosing a different tool when appropriate.\n- Avoiding re-use of failed tool calls by checking prior results.\n- Stopping and concluding if all tool paths have been exhausted.\n\nNEVER:\n- Repeat failed tool calls unnecessarily.\n- Respond with general knowledge if tools are required to verify the answer.\n\nRemember: your goal is to reason with tool assistance. Use tools thoughtfully and adaptively to solve the user's question.\n        "
    },
    {
      "role": "user",
      "content": "I'm searching for movie theaters in Hangzhou and wondering about the weather forecast for this evening."
    },
    {
      "role": "assistant",
      "content": "<|tool_call_start|>\n```python\nsearchPOI(city=\"杭州\",extensions=\"base\",keywords=\"电影院\")\n```\n<|tool_call_end|>\nI'll help you find movie theaters in Hangzhou and check the weather forecast for this evening. Let me gather that information for you."
    },
    {
      "role": "tool",
      "content": "{\"status\":\"1\",\"count\":229,\"pois\":[{\"id\":\"B0FFIQG5YR\",\"name\":\"万达影城(砂之船国际生活广场店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"解放东路8号砂之船国际生活广场B1层\",\"location\":\"120.213532,30.244310\",\"tel\":\"0571-81106343;0571-81106969\",\"distance\":[],\"photos\":[]},{\"id\":\"B0H63L9Z18\",\"name\":\"SFC上影国际影城(高德置地广场店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"市民中心站G口旁(高德置地广场5楼)\",\"location\":\"120.208656,30.242733\",\"tel\":\"0571-87390565\",\"distance\":[],\"photos\":[]},{\"id\":\"B0J0XOKKZF\",\"name\":\"星光嘉映影城(杭州来福士店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"来福士中心07层03号\",\"location\":\"120.213035,30.248775\",\"tel\":\"17300922881\",\"distance\":[],\"photos\":[]},{\"id\":\"B0I6AMXU40\",\"name\":\"万象影城(杭州万象城店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"富春路701号杭州万象城5楼\",\"location\":\"120.214835,30.251509\",\"tel\":\"15657105178\",\"distance\":[],\"photos\":[]},{\"id\":\"B023B0BI2S\",\"name\":\"卢米埃影城(银泰百货庆春店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"景昙路18-26号庆春银泰6层\",\"location\":\"120.205439,30.260063\",\"tel\":[],\"distance\":[],\"photos\":[]},{\"id\":\"B0FFG10DAO\",\"name\":\"金逸影城-脱口秀剧场(杭州五福天虹购物中心B座店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"新塘路108号杭州五福天虹购物中心B座6层606号\",\"location\":\"120.209629,30.264973\",\"tel\":\"0571-87702702;0571-87706773\",\"distance\":[],\"photos\":[]},{\"id\":\"B0G0SZ0C22\",\"name\":\"保利万和CFR国际影城(钱江世纪城店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"钱江世纪城钱江世纪公园A区13幢\",\"location\":\"120.239038,30.244492\",\"tel\":\"0571-83822017\",\"distance\":[],\"photos\":[]},{\"id\":\"B0FFFV2EZ4\",\"name\":\"德信影城(万泰城店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"望江东路332号万泰城3-4层\",\"location\":\"120.192637,30.229449\",\"tel\":\"0571-87173399;0571-87710207\",\"distance\":[],\"photos\":[]},{\"id\":\"B0FFIQ15SN\",\"name\":\"至潮影城(庆春路店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"庆春路87号锦和大厦5层\",\"location\":\"120.176942,30.257545\",\"tel\":\"13336013203\",\"distance\":[],\"photos\":[]},{\"id\":\"B0H2PCRWZ9\",\"name\":\"INF无限时空电影剧场\",\"type\":\"体育休闲服务;影剧院;剧场\",\"address\":\"建国北路286号凤起农贸市场一号楼二层(桐江小院楼上)\",\"location\":\"120.181682,30.265485\",\"tel\":\"17767151776\",\"distance\":[],\"photos\":[]},{\"id\":\"B0FFI2YFAE\",\"name\":\"中影国际影城(星光大道二期店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"闻涛路1766号华联·星光大道2期4层\",\"location\":\"120.206202,30.211833\",\"tel\":\"0571-88997727\",\"distance\":[],\"photos\":[]},{\"id\":\"B0GRC5CDQF\",\"name\":\"西戏·XIXI LIVE(星澜里店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"西兴街道西兴路2333号星澜大厦4幢301室\",\"location\":\"120.221481,30.215414\",\"tel\":\"13516855490\",\"distance\":[],\"photos\":[]},{\"id\":\"B0GR07XGIF\",\"name\":\"海马国际影城(江和美亲子广场店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"江和美亲子广场4层(三堡地铁站B1口步行230米)\",\"location\":\"120.227284,30.268501\",\"tel\":\"0571-85771502\",\"distance\":[],\"photos\":[]},{\"id\":\"B0J6CCKZ3T\",\"name\":\"万达影城(星耀城1期九宜城店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"星耀城1期3层(江陵路地铁站A口步行350米)\",\"location\":\"120.215352,30.212717\",\"tel\":\"0571-81138116\",\"distance\":[],\"photos\":[]},{\"id\":\"B023B08Q8G\",\"name\":\"新华影都(庆春路店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"娃哈哈同乐舫大酒店3层\",\"location\":\"120.169271,30.257827\",\"tel\":\"0571-87046523;0571-87212554\",\"distance\":[],\"photos\":[]},{\"id\":\"B0FFG58CAX\",\"name\":\"海上明珠国际影城(银泰百货西湖店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"延安路98号西湖银泰城A馆5层\",\"location\":\"120.164970,30.243828\",\"tel\":\"0571-87002038\",\"distance\":[],\"photos\":[]},{\"id\":\"B023B19440\",\"name\":\"中影·国际影城(杭州滨江星光大道店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"江南大道228号星光国际广场2幢4层\",\"location\":\"120.209128,30.208094\",\"tel\":\"0571-88924880;0571-88924988\",\"distance\":[],\"photos\":[]},{\"id\":\"B0IAJ167YH\",\"name\":\"尚橙电影工场(利星名品广场店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"中山南路77号利星名品广场4层\",\"location\":\"120.168520,30.228409\",\"tel\":\"0571-56668320;0571-56668321;13646859933\",\"distance\":[],\"photos\":[]},{\"id\":\"B0HADOGYLE\",\"name\":\"百美汇影城(杭州嘉里中心店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"延安路385号杭州嘉里中心L4层\",\"location\":\"120.161926,30.260416\",\"tel\":\"0571-81181775\",\"distance\":[],\"photos\":[]},{\"id\":\"B0H6OAS72Q\",\"name\":\"德信影城(杭州之翼购物中心店)\",\"type\":\"体育休闲服务;影剧院;电影院\",\"address\":\"鸿泰路133号杭州之翼购物中心4F\",\"location\":\"120.221653,30.294449\",\"tel\":\"0571-88881601\",\"distance\":[],\"photos\":[]}],\"mapUrl\":\"https://restapi.amap.com/v3/staticmap?key=9baeae2ef50243c0d49141fa31c1dcf9&zoom=12&size=750*500&scale=2&location=120.213532,30.244310&markers=mid,0xFF0000,A:120.213532,30.244310&markers=mid,0xFF0000,B:120.208656,30.242733&markers=mid,0xFF0000,C:120.213035,30.248775&markers=mid,0xFF0000,D:120.214835,30.251509&markers=mid,0xFF0000,E:120.205439,30.260063&markers=mid,0xFF0000,F:120.209629,30.264973&markers=mid,0xFF0000,G:120.239038,30.244492&markers=mid,0xFF0000,H:120.192637,30.229449&markers=mid,0xFF0000,I:120.176942,30.257545&markers=mid,0xFF0000,J:120.181682,30.265485&markers=mid,0xFF0000,K:120.206202,30.211833&markers=mid,0xFF0000,L:120.221481,30.215414&markers=mid,0xFF0000,M:120.227284,30.268501&markers=mid,0xFF0000,N:120.215352,30.212717&markers=mid,0xFF0000,O:120.169271,30.257827&markers=mid,0xFF0000,P:120.164970,30.243828&markers=mid,0xFF0000,Q:120.209128,30.208094&markers=mid,0xFF0000,R:120.168520,30.228409&markers=mid,0xFF0000,S:120.161926,30.260416&markers=mid,0xFF0000,T:120.221653,30.294449&path=120.213532,30.244310;120.208656,30.242733;120.213035,30.248775;120.214835,30.251509;120.205439,30.260063;120.209629,30.264973;120.239038,30.244492;120.192637,30.229449;120.176942,30.257545;120.181682,30.265485;120.206202,30.211833;120.221481,30.215414;120.227284,30.268501;120.215352,30.212717;120.169271,30.257827;120.164970,30.243828;120.209128,30.208094;120.168520,30.228409;120.161926,30.260416;120.221653,30.294449\"}",
    }
]

with open("available_tool_example.json","r") as f:
    available_tools = json.load(f)["available_tools"]

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--tokenizer_path",type=str,required=True)
    argument_parser.add_argument("--base_url",type=str,required=True)
    argument_parser.add_argument("--model",type=str,required=True)
    argument_parser.add_argument("--output_path",type=str,required=True)
    args = argument_parser.parse_args()
    
    client = openai.OpenAI(
        api_key="1",
        base_url=args.base_url
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    generate_result = generate(
        client=client,
        tokenizer=tokenizer,
        model=args.model,
        messages_minicpm=example_messages_history,
        tools=available_tools
    )

    tool_calls = parse_tool_for_minicpm3(generate_result)
    result = []
    
    for message in example_messages_history:
        if message["role"] == "system":
            result.append(
                {"system":message["content"]}
            )
        elif message["role"] == "user":
            result.append(
                {"human":message["content"]}
            )
        elif message["role"] == "assistant":
            history_tool_calls = parse_tool_for_minicpm3(
                message["content"]
            )
            result.append(
                {
                    "gpt":message["content"],
                    "function_call":history_tool_calls
                }
            )
        elif message["role"] == "tool":
            if "observation" not in result[-1]:
                result[-1]["observation"] = []
            result[-1]["observation"].append(message["content"])
    result.append(
        {
            "gpt": generate_result,
            "function_call":tool_calls
        }
    )
    
    with open(args.output_path,"w") as f:
        json.dump(result,f,ensure_ascii=False,indent=4)