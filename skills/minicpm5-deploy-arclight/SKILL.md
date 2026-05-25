# Deploy ArcLight from Source (CPU)

ArcLight is a lightweight C/C++ LLM inference framework for unified-memory systems. The recommended path is to build from source, then run a GGUF model with `al-gen`, `al-chat`, or `al-ppl`.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL` | `/path/to/MiniCPM5-1B-Q4_0.gguf` | required |
| `PROMPT` | `"Hello!"` | `"Hello!"` |
| `THREADS` | `4` | choose for the target CPU |
| `NUMA_MODE` | `none` or `tp` | `none` for first run |
| `NODES` | `1`, `2`, `4` | `1` with `NUMA_MODE=none` |
| `MAX_GEN` | `256` | `256` |

## Steps

### 1. Build from source

```bash
git clone https://github.com/OpenBMB/ArcLight.git
cd ArcLight

cmake -B build -DARCLIGHT_BACKEND=AUTO
cmake --build build --config Release -j 32
```

Use `ARCLIGHT_BACKEND=AUTO` by default. Set it explicitly only when needed:

- `X86`: force the x86 backend
- `NEON`: force the ARM NEON backend
- `NONE`: build without architecture-specific backend code

### 2. Prepare a GGUF model

ArcLight uses GGUF checkpoints from `llama.cpp`. Use a supported model family:

- MiniCPM5-1B
- Qwen3
- Llama2

For first validation, prefer a small quantized model such as `MiniCPM5-1B-Q4_0.gguf`.

### 3A. One-shot generation

```bash
./build/al-gen \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --numa none --nodes 1 \
    --threads ${THREADS} \
    --max_length 4096 \
    --max_gen ${MAX_GEN}
```

### 3B. Interactive chat

```bash
./build/al-chat \
    --model "${MODEL}" \
    --numa none --nodes 1 \
    --threads ${THREADS} \
    --max_length 4096 \
    --max_gen ${MAX_GEN}
```

To seed the first turn:

```bash
./build/al-chat \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --numa none --nodes 1 \
    --threads ${THREADS}
```

### 3C. Perplexity

```bash
./build/al-ppl \
    --model "${MODEL}" \
    --prompt "Good morning, Miss Lee!" \
    --numa none --nodes 1 \
    --threads ${THREADS}
```

## Many-core CPU / NUMA

Start with single-node mode for correctness:

```bash
./build/al-gen \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --numa none --nodes 1 \
    --threads ${THREADS}
```

Use cross-NUMA tensor parallelism on many-core machines:

```bash
./build/al-gen \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --numa tp --nodes ${NODES} \
    --threads ${THREADS}
```

Rules:

- `--numa none` requires `--nodes 1`.
- `--numa tp` requires `--nodes N` where `N > 1`.
- In the current version, `NODES` should be a power of 2.
- Choose `THREADS` so it can be evenly divided across `NODES`.
- `--numa pp` is reserved for future pipeline parallelism and is not implemented.

## Optional memory buffers

If allocation fails or the model is larger, pass manual buffer sizes:

```bash
./build/al-gen \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --numa none --nodes 1 \
    --threads ${THREADS} \
    --w_gb 4 --a_gb 8 --kv_gb 2 --work_gb 2
```

Meaning:

- `--w_gb`: weight buffer size
- `--a_gb`: activation buffer size
- `--kv_gb`: KV cache buffer size
- `--work_gb`: temporary workspace size

Increase `--kv_gb` for longer `--max_length`. Increase `--w_gb` for larger models.

## Validate

Run:

```bash
MODEL=/path/to/MiniCPM5-1B-Q4_0.gguf
THREADS=4

./build/al-gen \
    --model "${MODEL}" \
    --prompt "1+1=?" \
    --numa none --nodes 1 \
    --threads ${THREADS} \
    --max_gen 64
```

The reply should contain `2` or a short explanation that evaluates to `2`.

## Common pitfalls

- **Program aborts in single-node mode**: use `--numa none --nodes 1`.
- **Tensor parallel mode fails**: use `--numa tp --nodes N` with `N > 1`; in this version, `N` should be a power of 2.
- **Pipeline parallelism fails**: `--numa pp` is not implemented yet.
- **Model loading fails**: verify the checkpoint is GGUF and from a supported model family.
- **Out of memory**: increase `--w_gb`, `--a_gb`, `--kv_gb`, or `--work_gb`; longer contexts usually need a larger KV cache.
- **Poor CPU throughput**: check `--threads`, NUMA layout, and thread-core bindings; use `--print_binding 1 --print_perf 1` for diagnostics.

## When NOT to use

- Need CUDA server inference -> use a GPU-oriented runtime instead.
- Need Apple Silicon MLX inference -> use an MLX deployment path instead.
- Need a desktop GUI -> use a GUI runtime that supports GGUF models.
- Need pipeline parallelism -> wait for ArcLight `--numa pp` support.

## Reference

- [`Arclight`](https://github.com/OpenBMB/ArcLight)
- [llama.cpp GGUF documentation](https://github.com/ggml-org/llama.cpp)
