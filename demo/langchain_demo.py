"""
你只需要最少6g显存(足够)的显卡就能在消费级显卡上体验流畅的rag。

使用方法：
1. 运行pull_request/rag/langchain_demo.py
2. 上传pdf/txt文件(同一目录下可传多个)
3. 输入问题。

极低显存(4g)使用方法：
1. 根据MiniCPM/quantize/readme.md进行量化，推荐量化MiniCPM-1B-sft-bf16
2. 将cpm_model_path修改为量化后模型地址
3. 保证encode_model_device设置为cpu
"""


from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from argparse import ArgumentParser
from langchain.llms.base import LLM
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from langchain.prompts import PromptTemplate
from pydantic.v1 import Field
import re
import gradio as gr

parser = ArgumentParser()
# 大语言模型参数设置
parser.add_argument(
    "--cpm_model_path",
    type=str,
    default="openbmb/MiniCPM-1B-sft-bf16",
)
parser.add_argument(
    "--cpm_device", type=str, default="cuda:0", choices=["auto", "cuda:0"]
)
parser.add_argument("--backend", type=str, default="torch", choices=["torch", "vllm"])

# 嵌入模型参数设置
parser.add_argument(
    "--encode_model", type=str, default="BAAI/bge-base-zh"
)
parser.add_argument(
    "--encode_model_device", type=str, default="cpu", choices=["cpu", "cuda:0"]
)
parser.add_argument("--query_instruction", type=str, default="")
parser.add_argument(
    "--file_path", type=str, default="/root/ld/pull_request/rag/红楼梦.pdf"
)

# 生成参数
parser.add_argument("--top_k", type=int, default=3)
parser.add_argument("--top_p", type=float, default=0.7)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max_new_tokens", type=int, default=4096)
parser.add_argument("--repetition_penalty", type=float, default=1.02)

# retriever参数设置
parser.add_argument("--embed_top_k", type=int, default=5)
parser.add_argument("--chunk_size", type=int, default=256)
parser.add_argument("--chunk_overlap", type=int, default=50)
args = parser.parse_args()


def clean_text(text):
    """
    清理文本，去除中英文字符、数字及常见标点。
    
    参数:
    text (str): 需要清理的原始文本。
    
    返回:
    str: 清理后的文本。
    """
    # 定义需要去除的字符模式：中文、英文、数字、常见标点
    pattern = r'[\u4e00-\u9fa5]|[A-Za-z0-9]|[.,;!?()"\']'

    # 使用正则表达式替换这些字符为空字符串
    cleaned_text = re.sub(pattern, "", text)

    # 去除多余的空格
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    return cleaned_text


class MiniCPM_LLM(LLM):
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)

    def __init__(self, model_path: str):
        """
        继承langchain的MiniCPM模型
        
        参数:
        model_path (str): 需要加载的MiniCPM模型路径。

        返回:
        self.model: 加载的MiniCPM模型。
        self.tokenizer: 加载的MiniCPM模型的tokenizer。
        """
        super().__init__()
        if args.backend == "vllm":
            from vllm import LLM

            self.model = LLM(
                model=model_path, trust_remote_code=True, enforce_eager=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16
            ).to(args.cpm_device)
            self.model = self.model.eval()

    def _call(self, prompt, stop: Optional[List[str]] = None):
        """
        langchain.llm的调用
        
        参数:
        prompt (str): 传入的prompt文本

        返回:
        responds (str): 模型在prompt下生成的文本
        """
        if args.backend == "torch":
            inputs = self.tokenizer("<用户>{}".format(prompt), return_tensors="pt")
            inputs = inputs.to(args.cpm_device)
            # Generate
            generate_ids = self.model.generate(
                inputs.input_ids,
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            responds = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            # responds, history = self.model.chat(self.tokenizer, prompt, temperature=args.temperature, top_p=args.top_p, repetition_penalty=1.02)
        else:
            from vllm import SamplingParams

            params_dict = {
                "n": 1,
                "best_of": 1,
                "presence_penalty": args.repetition_penalty,
                "frequency_penalty": 0.0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_beam_search": False,
                "length_penalty": 1,
                "early_stopping": False,
                "stop": None,
                "stop_token_ids": None,
                "ignore_eos": False,
                "max_tokens": args.max_new_tokens,
                "logprobs": None,
                "prompt_logprobs": None,
                "skip_special_tokens": True,
            }
            sampling_params = SamplingParams(**params_dict)
            prompt = "<用户>{}<AI>".format(prompt)
            responds = self.model.generate(prompt, sampling_params)
            responds = responds[0].outputs[0].text

        return responds

    @property
    def _llm_type(self) -> str:
        return "MiniCPM_LLM"


# 加载PDF和TXT文件
def load_documents(file_paths):
    """
        加载文本和pdf文件中的字符串，并进行简单的清洗
        
        参数:
        file_paths (str or list): 传入的文件地址或者文件列表

        返回:
        documents (list): 读取的文本列表
    """
    files_list = []
    if type(file_paths) == list:
        files_list = file_paths
    else:
        files_list = [file_paths]
    documents = []
    for file_path in files_list:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        doc = loader.load()
        doc[0].page_content = clean_text(doc[0].page_content)
        documents.extend(doc)

    return documents


def load_models():
    """
    加载模型和embedding模型
    
    返回:
    llm: MiniCPM模型
    embedding_models: embedding模型
    """
    llm = MiniCPM_LLM(model_path=args.cpm_model_path)
    embedding_models = HuggingFaceBgeEmbeddings(
        model_name=args.encode_model,
        model_kwargs={"device": args.encode_model_device},  # 或者 'cuda' 如果你有GPU
        encode_kwargs={
            "normalize_embeddings": True,  # 是否归一化嵌入
            "show_progress_bar": True,  # 是否显示进度条
            "convert_to_numpy": True,  # 是否将输出转换为numpy数组
            "batch_size": 8,  # 批处理大小'
        },
        query_instruction=args.query_instruction,
    )
    return llm, embedding_models


# 分割并嵌入文档
def embed_documents(documents, embedding_models):
    """
    对文档进行分割和嵌入
    
    参数:
    documents (list): 读取的文本列表
    embedding_models: embedding模型

    返回:
    vectorstore:向量数据库
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(texts, embedding_models)
    return vectorstore


def create_prompt_template():
    """
    创建自定义的prompt模板
    
    返回:
    PROMPT:自定义的prompt模板
    """
    custom_prompt_template = """请使用以下内容片段对问题进行最终回复，如果内容中没有提到的信息不要瞎猜，严格按照内容进行回答，不要编造答案，如果无法从内容中找到答案，请回答“片段中未提及，无法回答”，不要编造答案。
    Context:
    {context}

    Question: {question}
    FINAL ANSWER:"""
    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return PROMPT


# 创建RAG链
def create_rag_chain(llm, prompt):
    # qa=load_qa_with_sources_chain(llm, chain_type="stuff")
    qa = prompt | llm
    return qa


def analysis_links(docs):
    """
    分析链接
    
    参数:
    docs (list): 读取的文本列表
    
    返回:
    links_string:相关文档引用字符串，docname page content

    示例:
    >>> docs = [
    ...     {'source': 'Document1', 'page': 1, 'content': 'This is the first document.'},
    ...     {'source': 'Document2', 'page': 2, 'content': 'This is the second document.'}
    ... ]
    >>> extract_links(docs)
    'Document1 page:1 \n\nThis is the first document.\nDocument2 page:2 \n\nThis is the second document.'
    """
    links_string = ""
    for i in docs:
        i.metadata["source"] = i.metadata["source"].split("/")[-1]
        i.metadata["content"] = i.page_content
        links_string += f"{i.metadata['source']} page:{i.metadata['page']}\n\n{i.metadata['content']}\n\n"
    return links_string


# 主函数
def main():
    # 加载文档
    documents = load_documents(args.file_path)

    # 嵌入文档
    vectorstore = embed_documents(documents, embedding_models)

    # 自建prompt模版
    Prompt = create_prompt_template()

    # 创建RAG链
    rag_chain = create_rag_chain(llm, Prompt)

    # 用户查询
    while True:
        query = input("请输入查询：")
        if query == "exit":
            break
        docs = vectorstore.similarity_search(query, k=args.embed_top_k)
        all_links = analysis_links(docs)
        final_result = rag_chain.invoke({"context": all_links, "question": query})
        # result = rag_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        print(final_result)


exist_file = None


def process_query(file, query):
    global exist_file, documents, vectorstore, rag_chain

    if file != exist_file:

        # 加载文档
        documents = load_documents(file if isinstance(file, list) else file.name)

        # 嵌入文档
        vectorstore = embed_documents(documents, embedding_models)

        # 自建prompt模版
        Prompt = create_prompt_template()

        # 创建RAG链
        rag_chain = create_rag_chain(llm, Prompt)

        exist_file = file

    # 搜索并获取结果
    docs = vectorstore.similarity_search(query, k=args.embed_top_k)
    all_links = analysis_links(docs)
    final_result = rag_chain.invoke({"context": all_links, "question": query})
    # result = rag_chain({"input_documents": docs, "question": query}, return_only_outputs=False)
    print(final_result)
    final_result = final_result.split("FINAL ANSWER:")[-1]
    return final_result, all_links


if __name__ == "__main__":

    llm, embedding_models = load_models()

    # 如果不需要web界面可以直接运行main函数
    #main()

    with gr.Blocks(css="#textbox { height: 380%; }") as demo:
        with gr.Row():
            with gr.Column():
                link_content = gr.Textbox(label="link_content", lines=30, max_lines=40)
            with gr.Column():
                file_input = gr.File(label="upload_files", file_count="multiple")
                final_anser = gr.Textbox(label="final_anser", lines=5, max_lines=10)
                query_input = gr.Textbox(
                    label="User",
                    placeholder="Input your query here!",
                    lines=5,
                    max_lines=10,
                )
                submit_button = gr.Button("Submit")
        submit_button.click(
            fn=process_query,
            inputs=[file_input, query_input],
            outputs=[final_anser, link_content],
        )
    demo.launch(share=True, show_error=True)
