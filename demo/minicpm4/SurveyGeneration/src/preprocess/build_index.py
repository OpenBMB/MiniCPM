import torch.distributed
import faiss
import pandas as pd
import faiss
import numpy as np
import jsonlines, json
from transformers import AutoModel
import os
import torch
'''
data format:
{
    "bibkey": "some_bibkey",
    "text": "The abstract or text of the paper."
}
example:
{   
    "bibkey": "arxivid1234.5678",
    "text": "Title: A Study on Something\nAbstract: This paper discusses the findings of a study on something important in the field of research.\nAuthors: John Doe"
}
'''

model_name = "openbmb/MiniCPM-Embedding-Light"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16).to("cuda") 

input_path = "./data/arxiv.jsonl"

with jsonlines.open(input_path) as f:
    survey_data = list(f)


xids = [item["bibkey"] for item in survey_data]
passages = [item["text"] for item in survey_data]

embeddings_doc_dense, _ = model.encode_corpus(passages, max_length=1024)


# faiss save index
index = faiss.IndexFlatIP(embeddings_doc_dense.shape[1])
id_map_index = faiss.IndexIDMap(index)
index = faiss.index_cpu_to_all_gpus(id_map_index)

x_ids_int = np.array(np.arange(len(xids)))

str_int_ids = {}
for i in range(len(xids)):
    str_int_ids[xids[i]] = x_ids_int[i]
str_int_ids_df = pd.DataFrame(str_int_ids, index=[0]).T.reset_index()
str_int_ids_df.columns = ["str_id", "int_id"]
str_int_ids_df.to_csv("./index/str_int_ids_abstract.csv", index=False)

index.add_with_ids(embeddings_doc_dense, x_ids_int)

index = faiss.index_gpu_to_cpu(index)
faiss.write_index(index, "./index/index_abstract.faiss")
