import json
import pandas as pd

with open("prompts/cpm-8b-1210_zh_cpm_0129.jsondpo.v2-dpo.jsonl", 'r')  as fin:
    data = json.load(fin)
    
# with open("prompts/cpm-8b-1210_zh_cpm_0129.jsondpo.v2-dpo.xlxs", 'w')  as fout:
df = pd.DataFrame(data)
df.to_excel("prompts/cpm-8b-1210_zh_cpm_0129.jsondpo.v2-dpo.xlsx")
