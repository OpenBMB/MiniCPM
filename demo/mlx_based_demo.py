"""
使用 MLX 快速推理 MiniCPM

如果你使用 Mac 设备进行推理，可以直接使用MLX进行推理。
由于 MiniCPM 暂时不支持 mlx 格式转换。您可以下载由 MLX 社群转换好的模型 [MiniCPM-2B-sft-bf16-llama-format-mlx](https://huggingface.co/mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx)。

并安装对应的依赖包


```bash
pip install mlx-lm
```

这是一个简单的推理代码，使用 Mac 设备推理 MiniCPM-2
```python
python -m mlx_lm.generate --model mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx --prompt "hello, tell me a joke." --trust-remote-code
```

"""

from mlx_lm import load, generate
from jinja2 import Template

def chat_with_model():
    model, tokenizer = load("mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx")
    print("Model loaded. Start chatting! (Type 'quit' to stop)")

    messages = []
    chat_template = Template(
        "{% for message in messages %}{% if message['role'] == 'user' %}{{'<用户>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        messages.append({"role": "user", "content": user_input})
        response = generate(model, tokenizer, prompt=chat_template.render(messages=messages), verbose=True)
        print("Model:", response)
        messages.append({"role": "ai", "content": response})


chat_with_model()
