# Deploy MiniCPM5-2B with vLLM on Ascend NPU (vllm-ascend)

[vllm-ascend](https://github.com/vllm-project/vllm-ascend) is the official Ascend NPU backend for vLLM. It ships as a **platform plugin**: you install it alongside a matching vLLM release, and vLLM auto-discovers it at import time — the model code, the OpenAI-compatible server and the sampling API are all identical to the CUDA path described in [`vllm.md`](./vllm.md).

MiniCPM5-2B uses the `LlamaForCausalLM` architecture, so it runs on the generic Ascend path with **no custom kernels and no `--model-impl` override**.

## Install

The path of least resistance is the prebuilt image, which already contains CANN, `torch_npu` and both wheels:

```bash
docker run -it --net=host --shm-size=1g \
    --device /dev/davinci0 --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    quay.io/ascend/vllm-ascend:v0.18.0
```

To install into an existing CANN environment instead:

```bash
pip install vllm==0.18.0 vllm-ascend==0.18.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

`torch` resolves to the **CPU** build (`2.9.0+cpu`) — that is correct and expected. Device support comes from `torch_npu`, not from `torch` itself.

## Environment setup

Source the CANN environment before every run. Without it the plugin loads but device init fails. Adjust these paths if CANN or NNAL is installed elsewhere:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

The `vllm_ascend_C` extension links against the `torch` and `torch_npu` shared libraries, so they must be on the loader path:

```bash
export LD_LIBRARY_PATH=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib:\
$(python3 -c "import torch_npu,os;print(os.path.dirname(torch_npu.__file__))")/lib:$LD_LIBRARY_PATH
```

Select devices with `ASCEND_RT_VISIBLE_DEVICES` — **`CUDA_VISIBLE_DEVICES` has no effect**:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0        # single card
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3  # four cards, pair with --tensor-parallel-size 4
```

Verify the toolchain before loading a model:

```bash
npu-smi info                      # health, HBM usage, running processes
python3 -c "import torch, torch_npu; print(torch_npu.npu.is_available(), torch_npu.npu.device_count())"
# True 8
```

> If `npu-smi` fails with `libc_sec.so: cannot open shared object file`, the driver libraries are missing from the loader path. Add them:
> `export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH`

## OpenAI-compatible server

Identical to the CUDA invocation — only the environment differs:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /path/to/MiniCPM5-2B \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --port 8000
```

Confirm the plugin took over — this line must appear in the startup log:

```
INFO [__init__.py:239] Platform plugin ascend is activated
```

### Tuning knobs

| Flag | Default | When to change |
| --- | --- | --- |
| `--max-model-len` | `131072` (native 128K) | drop to `8192` / `32768` to cut KV-cache reservation and shorten warmup |
| `--gpu-memory-utilization` | `0.9` | fraction of **HBM**, not host RAM; lower it when sharing a card |
| `--tensor-parallel-size` | `1` | must equal the number of devices in `ASCEND_RT_VISIBLE_DEVICES` |
| `--dtype` | `bfloat16` | keep bf16; 910B has native bf16 support and fp16 buys nothing |
| `--enforce-eager` | unset | skips ACL graph capture — use to isolate a graph-mode bug, at a real latency cost |

### Ascend-specific notes

- **`--quantization` / AWQ / GPTQ** are not interchangeable with the CUDA kernels. Ascend quantization goes through [msModelSlim](https://gitee.com/ascend/msit); a GPTQ checkpoint built for CUDA will not load.
- **Graph capture logs say "CUDA graphs".** The progress bar reads `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)` even on NPU — the string is hardcoded upstream in vLLM. The captured graphs are ACL graphs. Cosmetic only.
- **First start is slow.** Warmup (profile + KV cache alloc + graph capture) takes ~75 s for a 2B model on one 910B3. `torch.compile` artifacts are cached under `~/.cache/vllm/torch_compile_cache/`, so subsequent starts are faster.
- **`EngineCore died unexpectedly` at exit is normal.** For offline (`LLM(...)`) scripts this is teardown noise printed after your results, not a failure. Check whether your output was produced before treating it as an error.

The Docker command above is a minimal single-device example. Device nodes and driver mounts may differ across Ascend hardware and image versions; for multi-card serving, expose the required devices and set `--tensor-parallel-size` accordingly.


## Chat completions

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "用一句话解释什么是张量并行。"}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

MiniCPM5-2B is a think-only model. Use `enable_thinking: true`, `temperature: 1.0`, and `top_p: 0.95` for generation.

> **Thinking is on by default.** The bundled chat template enables it when `enable_thinking` is absent, and the reasoning text is returned inline inside a `<think>...</think>` block in `message.content` (not in a separate `reasoning` field, which stays `null` unless you configure a reasoning parser). MiniCPM5-2B is think-only, so keep `enable_thinking` enabled. Budget `max_tokens` accordingly — a short limit can be consumed by the `<think>` block before the final answer starts.

## Sample run

```bash
$ curl -sS http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiniCPM5-2B","messages":[{"role":"user","content":"1+1=?"}],
         "temperature":1.0,"top_p":0.95,"max_tokens":256,
         "chat_template_kwargs":{"enable_thinking":true}}'
{
  "choices": [{
    "message": {"role": "assistant", "content": "<think>...</think>2"},
    "finish_reason": "stop"
  }]
}
```

Streaming (`"stream": true`) and `GET /v1/models` are available through the standard vLLM OpenAI-compatible server.

## Offline / batched inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/MiniCPM5-2B",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    trust_remote_code=True,
)

out = llm.chat(
    [[{"role": "user", "content": "用一句话解释 GQA。"}]],
    SamplingParams(temperature=1.0, top_p=0.95, max_tokens=512),
    chat_template_kwargs={"enable_thinking": True},
)
print(out[0].outputs[0].text)
```

Run it with the environment from [Environment setup](#environment-setup) sourced:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python3 offline_infer.py
```

## Tool calling

MiniCPM5-2B emits **XML-style** tool calls:

```
<function name="get_weather"><param name="city">Beijing</param></function>
```

The tags `<function`, `<param`, `</param>` and `</function>` are **special tokens** in this tokenizer — set `"skip_special_tokens": false` if you want to see them in raw `/v1/completions` output.

No generic parser handles this shape: the other XML-style parsers wrap calls in `<tool_call>` and write parameters as `<parameter=city>`, not `<param name="city">`. MiniCPM5 needs its own parser, which vLLM has shipped as `minicpm5` since **v0.23.0** (v0.22.1 and earlier do not have it).

Parsing happens in vLLM's OpenAI server layer, above the platform backend, so none of this is Ascend-specific: `vllm-ascend` is not involved and the behaviour matches CUDA exactly.

### Requirements

`vllm-ascend` pins **the same version number as vLLM** ("vLLM (the same version as vllm-ascend)", upstream README), so the parser comes with the 0.23.0 stack as a whole:

| Component | Version |
| --- | --- |
| vllm-ascend / vLLM | 0.23.0 |
| CANN | 9.1.0 |
| torch / torch-npu | 2.10.0 / 2.10.0.post4 |

### Serve

No plugin and no patch — the parser is built in:

```bash
vllm serve /path/to/MiniCPM5-2B \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 --max-model-len 32768 --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser minicpm5
```

Passing `tools` in a request implies `tool_choice: "auto"`, which **requires both flags**. Without them the request fails with:

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

### Request

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

Non-streaming and streaming agree:

```jsonc
// tool needed
"content": "",
"tool_calls": [{
  "id": "call_91c904e4d81aa931",
  "type": "function",
  "function": {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}
}],
"finish_reason": "tool_calls"

// no tool needed — no spurious call
"content": "Hello.",
"tool_calls": [],
"finish_reason": "stop"
```

> These responses were produced with the same upstream parser file, run on an older vLLM with a compatibility shim; the 0.23.0 stack itself was not available on the test hardware.

## References

- [vllm-ascend](https://github.com/vllm-project/vllm-ascend) · [docs](https://vllm-ascend.readthedocs.io/)
- [MiniCPM vLLM (CUDA) deployment guide](./vllm.md)
- [Ascend CANN](https://www.hiascend.com/software/cann)
