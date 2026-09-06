---
name: minicpm5-deploy-vllm-ascend
description: Deploy MiniCPM5-2B with vLLM on Huawei Ascend NPU using vLLM-Ascend. Use when the user mentions vLLM-Ascend, Ascend NPU, Huawei Ascend, CANN, torch_npu, davinci devices, or wants an OpenAI-compatible MiniCPM5 server on Ascend hardware.
---

# Deploy MiniCPM5-2B with vLLM-Ascend

Use this skill for the BF16 / FP16 `openbmb/MiniCPM5-2B` checkpoint on Huawei Ascend NPU. `vllm-ascend` is installed alongside a matching vLLM release and supplies the Ascend backend.

## Required input

| Variable | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-2B` | required |
| `PORT` | `8000` | `8000` |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | `0` |
| `CTX_LEN` | `32768` | `32768`; the native maximum is `131072` |
| `MEM_FRAC` | `0.85` | `0.85` |
| `TP_SIZE` | `1` | number of visible devices |

Accept either the Hugging Face id or a local directory containing the MiniCPM5-2B model files.

## Prerequisites

Use a compatible Linux Ascend environment with CANN, PyTorch, and TorchNPU installed. Keep the vLLM and vLLM-Ascend versions matched. The basic serving recipe below follows the tested `vllm==0.18.0` / `vllm-ascend==0.18.0` pair.

Device visibility is controlled by `ASCEND_RT_VISIBLE_DEVICES`; `CUDA_VISIBLE_DEVICES` does not select Ascend devices.

## Install

The prebuilt image is the simplest option for a compatible Ascend host. This is a minimal single-device example; adjust device nodes and driver mounts for the host hardware and image documentation.

```bash
docker run -it --net=host --shm-size=1g \
    --device /dev/davinci0 --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    quay.io/ascend/vllm-ascend:v0.18.0
```

For an existing CANN environment:

```bash
pip install vllm==0.18.0 vllm-ascend==0.18.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

For built-in MiniCPM5 tool-call parsing, use the matching `vllm==0.23.0` / `vllm-ascend==0.23.0` stack described in [Tool calling](#tool-calling). The parser responses in the cookbook were produced with the upstream parser file on an older vLLM and a compatibility shim because the 0.23.0 stack was not available on the test hardware.

## Prepare the environment

Run the CANN and NNAL setup scripts before starting vLLM. Change the paths if those components are installed elsewhere.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib:\
$(python3 -c "import torch_npu,os;print(os.path.dirname(torch_npu.__file__))")/lib:$LD_LIBRARY_PATH
```

Check the driver and Python packages before loading the model:

```bash
npu-smi info
python3 -c "import torch, torch_npu; print(torch_npu.npu.is_available(), torch_npu.npu.device_count())"
```

If `npu-smi` reports a missing driver library, add the corresponding driver `lib64` directories to `LD_LIBRARY_PATH` according to the host installation.

## Launch the server

```bash
vllm serve "${MODEL_PATH}" \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --max-model-len ${CTX_LEN:-32768} \
    --gpu-memory-utilization ${MEM_FRAC:-0.85} \
    --tensor-parallel-size ${TP_SIZE:-1} \
    --trust-remote-code \
    --port ${PORT:-8000}
```

Set `TP_SIZE` to the number of devices in `ASCEND_RT_VISIBLE_DEVICES` when using tensor parallelism.

## Validate chat completions

MiniCPM5-2B is think-only. Use `temperature=1.0`, `top_p=0.95`, and `enable_thinking=true`:

```bash
curl http://localhost:${PORT:-8000}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

Expect HTTP 200 and an answer containing `2` in `choices[0].message.content`. Leave enough `max_tokens` for the model's reasoning text.

## Tool calling

MiniCPM5-2B emits XML-style tool calls. This section requires matching vLLM and vLLM-Ascend `0.23.0`; that release includes the `minicpm5` parser in the OpenAI server layer, so no parser plugin file is needed:

```bash
pip install vllm==0.23.0
pip install vllm-ascend==0.23.0 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

```bash
vllm serve "${MODEL_PATH}" \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --max-model-len ${CTX_LEN:-32768} \
    --gpu-memory-utilization ${MEM_FRAC:-0.85} \
    --tensor-parallel-size ${TP_SIZE:-1} \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser minicpm5 \
    --port ${PORT:-8000}
```

When sending `tools`, use both server flags above and the same 2B sampling settings:

```json
{
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
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": true}
}
```

## Common failures

- `torch_npu` cannot import: source the CANN/NNAL environment and check that the installed PyTorch, TorchNPU, CANN, vLLM, and vLLM-Ascend versions match the supported stack.
- No devices are visible: check `npu-smi info`, `ASCEND_RT_VISIBLE_DEVICES`, and the Docker device/driver mounts. Do not use `CUDA_VISIBLE_DEVICES` for this backend.
- HBM allocation or 128K startup failure: lower `--max-model-len`, then lower `--gpu-memory-utilization`.
- Tensor-parallel initialization failure: make `--tensor-parallel-size` match the number of visible Ascend devices.
- Tool requests fail with an auto-tool-choice error: launch with both `--enable-auto-tool-choice` and `--tool-call-parser minicpm5`.

## Reference

See [`docs/deployment/vllm-ascend.md`](../../docs/deployment/vllm-ascend.md) for the human-readable cookbook and the [official vLLM-Ascend installation guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html) for hardware-specific setup.
