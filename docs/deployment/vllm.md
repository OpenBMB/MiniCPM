# Deploy MiniCPM5-1B and MiniCPM5-2B with vLLM

vLLM `>=0.21` supports MiniCPM5-1B and MiniCPM5-2B natively — no custom kernels. For production-grade throughput and OpenAI-compatible chat completions, this is the recommended path.

For MiniCPM5-15B-A2.5B, see [MiniCPM5-15B-A2.5B](#minicpm5-15b-a25b).

## Install

```bash
pip install "vllm>=0.21"          # latest (CUDA 13.x driver hosts)
# pip install "vllm==0.10.1.1"    # fallback for CUDA 12.x driver hosts
```

## OpenAI-compatible server

```bash
vllm serve openbmb/MiniCPM5-2B \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

### Tuning knobs

| Flag | Default | When to change |
| --- | --- | --- |
| `--max-model-len` | `131072` (native 128K) | drop to `8192` / `32768` to free KV-cache on small GPUs |
| `--gpu-memory-utilization` | `0.85` | drop on **shared** GPUs — vLLM hard-fails if `(free / total) < value` |
| `--dtype` | `bfloat16` | `float16` for older GPUs (newer NVIDIA GPUs prefer bf16) |
| `--enforce-eager` | unset | set if CUDA graphs OOM on tiny VRAM budgets |

## Chat completions

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | --- | --- |
| MiniCPM5-2B Think | `true` | 1.0 | 0.95 |
| MiniCPM5-1B Think | `true` | 0.9 | 0.95 |
| MiniCPM5-1B No-think | `false` | 0.7 | 0.95 |

## Sample run

```bash
$ curl -sS http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiniCPM5-2B","messages":[{"role":"user","content":"1+1=?"}],"temperature":1.0,"top_p":0.95,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":true}}'
{
  "id": "chatcmpl-...",
  "model": "MiniCPM5-2B",
  "choices": [{"message": {"role": "assistant", "content": "2"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 14, "completion_tokens": 2, "total_tokens": 16}
}
```

## Offline / batched inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="openbmb/MiniCPM5-2B", dtype="bfloat16", max_model_len=131072)
out = llm.chat(
    [[{"role": "user", "content": "用一句话解释 GQA。"}]],
    SamplingParams(temperature=1.0, top_p=0.95, max_tokens=512),
    chat_template_kwargs={"enable_thinking": True},
)
print(out[0].outputs[0].text)
```

## Tool calling (plugin)

MiniCPM5-1B and MiniCPM5-2B emit **XML-style** tool calls. The vLLM-side parser ([vllm-project/vllm#43175](https://github.com/vllm-project/vllm/pull/43175)) was merged to `main` on 2026-05-27 but is **not in any pip release yet** — `v0.22.0` (2026-05-29) was cut before that merge and the file is absent from the v0.22.0 tree.

As a bridge, this repo ships the parser at [`tool_parsers/minicpm5xml_tool_parser.py`](../../tool_parsers/minicpm5xml_tool_parser.py) (the same file as the upstream PR). Load it via vLLM's `--tool-parser-plugin`:

```bash
vllm serve openbmb/MiniCPM5-2B \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 --max-model-len 131072 --port 8000 \
    --enable-auto-tool-choice \
    --tool-parser-plugin /path/to/MiniCPM/tool_parsers/minicpm5xml_tool_parser.py \
    --tool-call-parser minicpm5
```

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
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

Once vLLM `v0.23` (or later) is released with the parser baked in, drop `--tool-parser-plugin` and use only `--tool-call-parser minicpm5`.

## MiniCPM5-15B-A2.5B

The preview in [vLLM PR #68748](https://github.com/vllm-project/vllm/pull/68748) registers `MiniCPM5MoEForCausalLM` and supports BF16 and compressed-tensors INT4 W4A16. The launch below uses one H800 with TP=1.

### Prepare the preview runtime

The validation used the official CUDA nightly built at `9c1d45c21836370104d4eb51d3b46eda20719c47` with the PR's Python code overlaid. Set `MODEL_ROOT` to the host directory containing the target, its fast tokenizer and, for DSpark, the matching draft snapshot. Use complete local directories so all model files are available inside the container.

```bash
git init vllm-minicpm5
cd vllm-minicpm5
git fetch --depth 1 https://github.com/cyxu0401/vllm.git 9d6347741dffb309e014475b49d1a752025d4e4a
git checkout --detach FETCH_HEAD

MODEL_ROOT="/absolute/path/to/checkpoints"
docker run --rm -it --gpus device=0 --ipc=host \
    -p 127.0.0.1:8000:8000 \
    -v "$PWD:/vllm-src:ro" -v "$MODEL_ROOT:/models:ro" \
    --entrypoint /bin/bash \
    vllm/vllm-openai@sha256:663d493862a5d2c3f212f28d1bf5a5fd76dff08ec76dd1ebd72a7d230621be9a
```

In the container, overlay the five Python files onto the image's vLLM package. The overlay keeps the image's compiled extensions and uses a separate directory for the PR files.

```bash
VLLM_PACKAGE=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm").origin).parent)')
OVERLAY=$(mktemp -d)
cp -as "$VLLM_PACKAGE" "$OVERLAY/vllm"

for file in \
    model_executor/models/minicpm5_moe.py \
    model_executor/models/registry.py \
    transformers_utils/config.py \
    transformers_utils/configs/__init__.py \
    transformers_utils/configs/minicpm5_moe.py; do
    cp --remove-destination "/vllm-src/vllm/$file" "$OVERLAY/vllm/$file"
done

export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
cd "$OVERLAY"
```

### Choose the checkpoint and launch

Set `MODEL_DIR` to the BF16 snapshot and `TOKENIZER_DIR` to the target's Transformers-compatible fast tokenizer, including its chat template. Use the model directory for both when it includes the compatible tokenizer. Run any optional settings in the same shell before starting the server.

```bash
MODEL_DIR="/models/MiniCPM5-15B-A2.5B"
TOKENIZER_DIR="/models/MiniCPM5-15B-A2.5B-tokenizer"
extra=()
```

For the asymmetric INT4 checkpoint in compressed-tensors format, select its directory and use Marlin for both Linear and MoE layers.

```bash
MODEL_DIR="/models/MiniCPM5-15B-A2.5B-INT4"
extra+=(--quantization compressed-tensors --linear-backend marlin --moe-backend marlin)
```

To run DSpark with either precision, use the draft trained for the selected target.

```bash
DSPARK_DIR="/models/MiniCPM5-15B-A2.5B-DSpark"
SPEC_CONFIG=$(python3 -c 'import json, sys; print(json.dumps({"method": "dspark", "model": sys.argv[1], "num_speculative_tokens": 7}))' "$DSPARK_DIR")
extra+=(--speculative-config "$SPEC_CONFIG")
```

```bash
vllm serve "$MODEL_DIR" --tokenizer "$TOKENIZER_DIR" \
    --host 0.0.0.0 --port 8000 --served-model-name MiniCPM5-15B-A2.5B \
    --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 4096 \
    --max-num-seqs 8 --gpu-memory-utilization 0.80 --generation-config vllm \
    "${extra[@]}"
```

### Chat completion

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-15B-A2.5B",
        "messages": [{"role": "user", "content": "Explain grouped-query attention in one sentence."}],
        "temperature": 0,
        "max_tokens": 128
    }'
```

### Validation scope

At the pinned revision, one H800 completed four 48-token greedy requests for each BF16 and INT4 configuration with DSpark off and on. The INT4 trace showed Marlin for both Linear and MoE. BF16 matched greedy token IDs on all four prompts across draft modes. INT4 matched on two of four. The validation used `/v1/completions` with fixed input token IDs. [vLLM PR validation](https://github.com/vllm-project/vllm/pull/68748#validation)
