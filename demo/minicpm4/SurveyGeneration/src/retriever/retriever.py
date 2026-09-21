import faiss
from fastapi import FastAPI
import torch
import pandas as pd
from collections import defaultdict
import pandas as pd
import jsonlines
from transformers import AutoModel, AutoTokenizer
import uvicorn
import asyncio
from pydantic import BaseModel
from typing import List,  Optional
import re
import json
import asyncio
import argparse


app = FastAPI()



model_name = "openbmb/MiniCPM-Embedding-Light"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16).to("cuda") 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

co = faiss.GpuMultipleClonerOptions()
co.shard = True
co.useFloat16 = True

faiss_index_path = "./index/index_abstract.faiss"  # Replace with your FAISS index path"
faiss_index = faiss.read_index(faiss_index_path)
faiss_index = faiss.index_cpu_to_all_gpus(faiss_index,co=co)

corpus_path = "./data/arxiv.jsonl"
with jsonlines.open(corpus_path) as f:
    paper_data = list(f)
paper_dict = {}
item_key = "text"


index_path = "./index/str_int_ids_abstract.csv"
index_df = pd.read_csv(index_path,converters={0: lambda x: str(x),1: lambda x: int(x)})
index_df.columns = ["str_id", "int_id"]
index_dict = index_df.set_index("int_id")["str_id"].to_dict()


for item in paper_data:
    paper_dict[item["bibkey"]] = item[item_key]

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

class MessageRequest(BaseModel):
    tool_calls: List
    topk: Optional[int] = 10
    
@app.post("/")
async def search_text_batch(request:MessageRequest):
    tool_calls = request.tool_calls
    topk = request.topk
    results = []
    finalize_indices = []
    search_engine_indices = []
    for i in range(len(tool_calls)):
        try:
            tool_calls[i]["name"]
        except KeyError:
            finalize_indices.append(i)
            continue
        if tool_calls[i]["name"] == "search_engine":
            search_engine_indices.append(i)
        elif tool_calls[i]["name"] == "finalize":
            finalize_indices.append(i)
        else:
            finalize_indices.append(i)
    

    tasks = []
    for i in range(len(tool_calls)):
        if i in search_engine_indices:
            tasks.append(call_search_engine(tool_calls[i], topk))
    search_task_results = await asyncio.gather(*tasks)
    num_search = 0
    num_finalize = 0
    for i in range(len(tool_calls)):
        if i in finalize_indices:
            search_keywords, bibkeys,abstracts, done, score = "",[], [], True, 0.0
            num_finalize += 1
        elif i in search_engine_indices:
            search_keywords, bibkeys, abstracts, done, score = search_task_results[num_search]
            num_search += 1
        
        titles = []
        for abstract in abstracts:
            try:
                title = abstract.split("\n")[1]
                title = title.split(":")[1].strip()
                titles.append(title)
            except:
                titles.append("")
        results.append({ "search_keywords":search_keywords, "summarys":abstracts, "done":done, "score":score, "titles":titles, "bibkeys":bibkeys})
    
        
    return results

def extract_tool_call(text: str):
    text = text.strip()

    pattern = r"<tool_call>(.*?)</tool_call>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    tool_text = match.group(1)
    try:
        tool_call = json.loads(tool_text)
    except json.JSONDecodeError:
        return None
    return tool_call if isinstance(tool_call, dict) else None


def get_response(queries,ref):
    text_raw = paper_dict[str(ref)]
    text_raw = tokenizer(text_raw,  max_length=8192, truncation=True)
    text_raw = tokenizer.decode(text_raw["input_ids"])
   
    response = text_raw
    response = f"bibkey: {str(ref)}\n"+response
    return response

async def call_search_engine(tool_call, topk=10):
    try:
        queries = tool_call["arguments"]["query"]
        if isinstance(queries, str):
            queries = [queries]
        else:
            queries = list(queries)
            
        if len(queries) == 0:
            return "", [], [], False, 0.0
        results = defaultdict(dict)
        query_embedding_to_text,_ =  model.encode_query(queries, max_length=512, show_progress_bar=False)
        _,results = faiss_index.search(query_embedding_to_text, topk)
        result2query = {}
        merge_rrf = defaultdict(float)
        for i in range(len(results)):
            for j in range(len(results[i])):
                merge_rrf[results[i][j]] += 1/(j+1)
                result2query[results[i][j]] = queries[i]
        results = sorted(merge_rrf.items(), key=lambda x: x[1], reverse=True)
        
        results = [x[0] for x in results][:topk]
        
        # new_queries = [result2query[result] for result in results]
        queries = ",".join(queries)
       
        # bibkeys = [str(results[i]) for i in range(len(results))]
        bibkeys = [str(index_dict[results[i]]) for i in range(len(results))]
        response = [f"bibkey: {bibkey}\n{paper_dict[bibkey]}" for bibkey in bibkeys]
        return queries,bibkeys , response, False, 0.0
    except Exception as e:
        print(f"Error in call_search_engine: {e}")
        return "",[], [], False, 0.0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI application.")
    parser.add_argument("--port", type=int, default=8400, help="Port to run the FastAPI application on.")
    args = parser.parse_args()
    
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)