from contextlib import contextmanager
from codetiming import Timer
@contextmanager
def _timer(name: str, timing_raw):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last
    
from buffer import SurveyManager
from buffer import BufferManager_V2 as BufferManager
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import argparse
from pydantic import BaseModel
import json
import aiohttp


app = FastAPI()

# 允许跨域（如果前端和后端端口不同需要加上）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def post_to_frontend(payload):
    print(f"Sending payload to frontend: {payload}")  # Log the payload being sent
    for ws in list(active_connections):
        try:
            await ws.send_text(payload)
        except Exception as e:
            print(f"Error sending to WebSocket: {e}")
            active_connections.remove(ws)


def write_to_json(data, path):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))

class OriginalvLLMRollout:
    def __init__(self, model_name_or_path):
        # init vLLM 
        self.rollout_model = LLM(   
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            top_k=20,
            max_tokens=2748,
        )

    def generate(self, input_texts):
        generated_texts = []
        completions = self.rollout_model.generate(input_texts, self.sampling_params, use_tqdm=False)
        for output in completions:
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        return generated_texts
    
    def chat(self, input_messages):
        generated_texts = []
        completions = self.rollout_model.chat(input_messages, self.sampling_params, use_tqdm=False)
        for output in completions:
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        return generated_texts

async def rollout_with_env(querys, batch_size, max_turns, model_path, url, 
                     deploy_port=None):
    """
    Args:
        querys: [string]
    """
    ###############################
    #### splited by batch size ####
    ###############################
    n = len(querys) // batch_size
    batch_querys = []
    for i in range(n+1):
        temp_data = querys[i*batch_size: (i+1)*batch_size]
        if len(temp_data) > 0:
            batch_querys.append(temp_data)
    print("QUERY NUMBER with BATCH: ", [len(x) for x in batch_querys])
    
    ###################
    #### init vllm ####
    ###################
    vllm_manager = OriginalvLLMRollout(model_path)

    ############################
    #### init Format Reward ####
    ############################
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    total_rollout_data = []
    for querys in batch_querys:
        ###########################################
        #### acquire env configs and init envs ####
        ###########################################
        buffer_manager = BufferManager(querys)

        while True:
            # Break at max-turns
            if buffer_manager.step >= max_turns:
                break
                
            ###############################
            #### prepare input prompts ####
            ###############################
            messagess_todo = buffer_manager.build_prompt_for_generator()
            # breakpoint()

            # Break when no tasks
            if len(messagess_todo) == 0:
                break
            
            ##########################
            #### generate by vLLM ####
            ##########################
            timing_raw = {}
            with _timer('vllm sampling', timing_raw):
                # response_texts = vllm_manager.chat(messagess_todo)
                response_texts = await asyncio.to_thread(vllm_manager.chat, messagess_todo)
        
            ##################################
            #### preprocess the responses ####
            ##################################
            # 对response的详细处理可以集成到环境类中，因环境而异, 先对Response进行预处理
            extracted_results = []
            for response_text in response_texts:
                result = BufferManager.parse_generator_response(response_text)
                extracted_results.append(result)
                
            #################################################
            #### execute in environment and get feedback ####
            #################################################
            payload = {
                "tool_calls": [x["tool_call"] for x in extracted_results]
            }
            if buffer_manager.step <=2:
                payload["topk"] = 20
            with _timer('get env feedback', timing_raw):
                # env_response_batched = requests.post(url, json=payload).json()
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as resp:
                        env_response_batched = await resp.json()

            ###################################
            #### postprocess the feedbacks ####
            ###################################
            with _timer('postprocessing', timing_raw):
                buffer_manager.update_trajectory(extracted_results, env_response_batched)
                buffer_manager.step += 1
            
            print(timing_raw)
            
            if deploy_port is not None:
                now_text = json_to_markdown(buffer_manager.batch_rollout_data[-1])
                now_search_keywords= buffer_manager.batch_rollout_data[-1]["trajectory"][-1]["search_keywords"]
                now_update = buffer_manager.batch_rollout_data[-1]["trajectory"][-1]["answer_thought"]
                next_update = buffer_manager.batch_rollout_data[-1]["trajectory"][-1]["tool_call_thought"]
                now_query = buffer_manager.batch_rollout_data[-1]["query"]
                trajs = buffer_manager.batch_rollout_data[-1]["trajectory"]
                updated_success = buffer_manager.batch_rollout_data[-1]["trajectory"][-1]["update_success"]
                if updated_success:
                    for traj in reversed(trajs):
                        if len(traj["summarys"]) > 0:
                            break
                    summary_num = len(traj["summarys"])
                    if summary_num == 0:
                        summary_text = "No summaries yet."
                    else:
                        summary_text = "\n".join(traj["summarys"])
                    frontend_payload = {
                        "markdown": now_text,
                        "searchKeywords": now_search_keywords,
                        "nowUpdate": now_update,
                        "nextUpdate": next_update,
                        "query": now_query,
                        "papers": summary_text
                    }
                    frontend_payload = json.dumps(frontend_payload, ensure_ascii=False)
                    try:
                        await post_to_frontend(frontend_payload)
                    except Exception as e:
                        print(f"Error posting to frontend: {e}")


            
        for item in  buffer_manager.batch_rollout_data:
            item["survey_text"] = SurveyManager.convert_survey_dict_to_str(item["state"]["current_survey"])

        total_rollout_data.extend(buffer_manager.batch_rollout_data)
        #####################################
        #### clear all envs and shutdown ####
        #####################################
        del buffer_manager
            
    return total_rollout_data


def json_to_markdown(json_data):
    text = SurveyManager.convert_survey_dict_to_str(json_data["state"]["current_survey"])
    all_summarys = {}
    for traj in json_data["trajectory"]:
        for item in traj["summarys"]:
            split_text = item.split("\n")
            bibkey = split_text[0].split(":")[1].strip()
            title_begin_index = item.find("Title:") + len("Title:")
            title_end_index = item.find("Abstract:")
            title = item[title_begin_index:title_end_index].strip()
            arxivid = bibkey.split("arxivid")[-1].strip()
            html = f"arxiv.org/abs/{arxivid}"
            all_summarys[bibkey] =  f"[{title}](https://{html})"
            
    reg = r"\\cite\{(.+?)\}"
    placeholder_reg = re.compile(r"^#\d+$")
    reg_bibkeys = re.findall(reg, text)
    bibkeys = []
    for bibkey in reg_bibkeys:
        single_bib = bibkey.split(",")
        for bib in single_bib:
            if not placeholder_reg.match(bib):
                bib = bib.strip()
                if bib and bib != "*" and bib not in bibkeys:
                    bibkeys.append(bib)
    
    bibkeys_index = {bibkey: i+1 for i, bibkey in enumerate(bibkeys)}
    
    def replace_bibkey(bibkey):
        bibkey = bibkey.group(1)
        single_bib = bibkey.split(",")
        new_bibs = []
        for bib in single_bib:
            if not placeholder_reg.match(bib):
                bib = bib.strip()
                if bib and bib != "*":
                    if bib in bibkeys_index:
                        new_bibs.append(f"{bibkeys_index[bib]}")
                    else:
                        print(f"Warning: {bib} not found in bibkeys")
        if len(new_bibs) > 0:
            return "[" + ",".join(new_bibs) + "]"
        else:
            return ""
    text = re.sub(reg, replace_bibkey, text)
    reference_text = "\n\n".join([f"[{i}] {all_summarys[bibkey]}" for bibkey, i in bibkeys_index.items()])
    text += "\n## References\n" + reference_text
    return text
        
async def test_surveyGen(model_path, out_path,querys, url, deploy_port=None):

    total_rollout_data = await rollout_with_env(querys, 1, 1000, model_path, url, deploy_port)
    all_md_texts = []
    for json_data in total_rollout_data:
        md_text = json_to_markdown(json_data)
        all_md_texts.append(md_text)
    
    all_md_texts = "\n\n".join(all_md_texts)
    with open(out_path, 'w', encoding='utf8') as f:
        f.write(all_md_texts)
    
    # with jsonlines.open(out_path, 'w') as writer:
    #     for item in total_rollout_data:
    #         writer.write(item)



class QueryRequest(BaseModel):
    query: str

@app.post("/generate_survey")
async def generate_survey(request: QueryRequest):
    global args  # Ensure args is accessible
    # 这里可以根据需要处理查询
    model_path = args.model_path
    out_path = args.output_file
    query = request.query
    querys = [query]  # 将查询转换为列表
    url = args.retriver_url
    deploy_port = args.port if args.port is not None else None
    try:
        await test_surveyGen(model_path, out_path, querys, url, deploy_port)
        return {"status": "success", "message": "Survey generated successfully."}
    except Exception as e:
        print(f"Error generating survey: {e}")
        return {"status": "error", "message": str(e)}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run survey generation with vLLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--query", type=str, required=True, help="Query to generate survey.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output Markdown file.")
    parser.add_argument("--retriver_url", type=str, default="http://localhost:8400", help="URL of the retriever service.")
    parser.add_argument("--port", type=str, default=None, help="Deploy port, default is None, which means not deploy.")
    args = parser.parse_args()

    if args.port is not None:
        import uvicorn
        uvicorn.run(app, host="localhost", port=int(args.port))# log_level="debug")
        
    # Run the survey generation
    else:
        asyncio.run(
            test_surveyGen(
                model_path=args.model_path,
                out_path=args.output_file,
                querys=[args.query],
                url=args.retriver_url
            )
        )
    