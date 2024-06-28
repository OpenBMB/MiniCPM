"""
这是一个简单的OpenAI接口代码,由于 MiniCPM-2B的限制，该脚本：
1. 没有工具调用功能
2. 没有System Prompt
3. 最大支持文本 4096 长度

运行本代码需要：
1. 启动本地服务，本方案使用的是 AutoModelForCausalLM.from_pretrained 读入模型，没有进行优化，可以根据需要自行修改。
2. 通过此代码进行请求。
"""

from openai import OpenAI

base_url = "http://127.0.0.1:8000/v1/"
client = OpenAI(api_key="MiniCPM-2B", base_url=base_url)

def chat(use_stream=True):
    messages = [
        {
            "role": "user",
            "content": "tell me a story"
        }
    ]
    response = client.chat.completions.create(
        model="MiniCPM-2B",
        messages=messages,
        stream=use_stream,
        max_tokens=4096,  # need less than 4096 tokens
        temperature=0.8,
        top_p=0.8
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            print(content)
    else:
        print("Error:", response.status_code)


def embedding():
    response = client.embeddings.create(
        model="bge-m3",
        input=["hello, I am MiniCPM-2B"],
    )
    embeddings = response.data[0].embedding
    print("Embedding_Success：", len(embeddings))


if __name__ == "__main__":
    chat(use_stream=True)


