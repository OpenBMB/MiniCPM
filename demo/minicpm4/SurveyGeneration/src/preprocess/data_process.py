# curl -L -o ~/Downloads/arxiv.zip\
#   https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv


import jsonlines

input_path = './data/arxiv-metadata-oai-snapshot.json'
output_path = './data/arxiv.jsonl'

new_data = []
with jsonlines.open(input_path, 'r') as reader:
    for item in reader:
        new_item = {
            'bibkey': f"arxivid{item['id']}",
            'text': f"Title: {item['title']}\nAbstract: {item['abstract']}\nAuthors: {item['authors']}",
        }
        new_data.append(new_item)

with jsonlines.open(output_path, 'w') as writer:
    for item in new_data:
        writer.write(item)