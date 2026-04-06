This document provides the technical pipeline for reproducing "Abliterated" models and quantizing them using **EvoPress**. Optimized for **Lightning AI Studios** and **Qwen 3.5 (Hybrid Mamba-Transformer)** architectures.

---

## ⚡ Critical Constraints & Strategy

> [!IMPORTANT]
> **GPU Scheduling:** Reserve **3 to 4 hours** of continuous A100 (80GB) time for Phases 2 through 5. Lightning AI extensions or instance timeouts will interrupt the database shredding and stitching processes, potentially corrupting the `ep_database` and requiring costly restarts. You will only be billed for the time you used.

> [!CAUTION]
> **Storage Management:** This pipeline generates ~350GB of temporary data. You **MUST** apply the symlink and cache-purge patches described in Phase 3 to avoid "Disk Full" crashes.

---

## Phase 1: Environment Setup (CPU Mode)
*Hardware: 4-Core CPU (Default/Free)*

1.  **System Dependencies & Authentication**
    ```bash
    sudo apt-get update && sudo apt-get install -y git-lfs build-essential cmake
    pip install -U "huggingface_hub[cli]" torch transformers accelerate safetensors datasets tqdm tiktoken dill
    python -c "from huggingface_hub import login; login()"
    ```

2.  **Clone & Compile Toolkits**
    ```bash
    # 1. Abliteration (Orion-zhen Biprojection)
    git clone https://github.com/Orion-zhen/abliteration.git
    cd abliteration && pip install -r requirements.txt && cd ..

    # 2. llama.cpp (Base Quantization)
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && cmake -B build && cmake --build build --config Release -j && pip install -r requirements.txt && cd ..

    # 3. EvoPress (gptq-gguf-toolkit)
    git clone https://github.com/IST-DASLab/gptq-gguf-toolkit
    cd gptq-gguf-toolkit && pip install -r requirements.txt && cd ..
    ```

3.  **Pre-Download Model**
    ```bash
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2')"
    ```

---

## Phase 2: Compute Core (Switch to 1x A100 GPU)
*Estimated Duration: 3-4 Hours*

1.  **Execute Abliteration**
    Create `abliteration/config.yaml`:
    ```yaml
    model: "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2"
    output_dir: "./abliterated_model"
    method: "full" # Biprojection + Norm-Preserving
    device: "cuda"
    ```
    Run:
    ```bash
    cd abliteration && python abliterate.py config.yaml && cd ..
    ```

2.  **Patch Tokenizer & Hardware-Level F16 Conversion**
    Fix the `TokenizersBackend` compatibility bug in `~/abliteration/abliterated_model/tokenizer_config.json`:
    - Change `"tokenizer_class": "TokenizersBackend"` to `"tokenizer_class": "Qwen2TokenizerFast"`.

    Convert to unquantized blueprint:
    ```bash
    python llama.cpp/convert_hf_to_gguf.py abliteration/abliterated_model --outfile abliterated-27B-F16.gguf --outtype f16
    ```

3.  **Generate Baseline "Gene Pool" Quants**
    ```bash
    mkdir -p base_quants
    for q in q2_k q3_k q4_k q5_k q6_k; do
        ./llama.cpp/build/bin/llama-quantize abliterated-27B-F16.gguf "base_quants/27B-${(U)q}.gguf" "$q"
    done
    ```

---

## Phase 3: Infrastructure Patches (FUSE & Hybrid Arch)
*Stay on A100 GPU*

1.  **Storage Headroom & Symlink Patch**
    ```bash
    # Clear HF Cache to reclaim ~130GB
    rm -rf ~/.cache/huggingface/hub/*
    pip cache purge && conda clean -a -y

    # Apply Zero-Copy Symlink Patch to Database Builder
    sed -i 's/cp "$model_path"/ln -sf "$(realpath "$model_path")"/g' ~/gptq-gguf-toolkit/mapper/build_ep_database.sh
    ```

2.  **Dynamic Architecture & FUSE Intercept**
    Apply this patch to `~/gptq-gguf-toolkit/mapper/gguf_splitter.py`. This fixes the cloud storage `EOVERFLOW` bug and stabilizes the Qwen 3.5 architecture metadata.

    ```python
    import sys
    import importlib.util
    import builtins
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.qwen2 import Qwen2Config

    # 1. CLOUD STORAGE FUSE BUG FIX (Errno 75)
    _orig_open = builtins.open
    def _patched_open(*args, **kwargs):
        f = _orig_open(*args, **kwargs)
        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
        if 'w' in mode and 'b' in mode:
            _orig_write = f.write
            def _chunked_write(data):
                chunk_size = 1024 * 1024 * 256 # 256MB Chunks
                try:
                    view = memoryview(data)
                    for i in range(0, len(view), chunk_size):
                        _orig_write(view[i:i+chunk_size])
                    return len(data)
                except Exception: return _orig_write(data)
            f.write = _chunked_write
        return f
    builtins.open = _patched_open

    # 2. QWEN 3.5 HYBRID ARCHITECTURE OVERRIDE
    _orig_qwen2_init = Qwen2Config.__init__
    def _patched_qwen2_init(self, **kwargs):
        kwargs.update({
            "hidden_size": 5120, 
            "intermediate_size": 17408, 
            "num_hidden_layers": 64, 
            "num_attention_heads": 40, 
            "num_key_value_heads": 8, 
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0
        })
        _orig_qwen2_init(self, **kwargs)
        self.model_type = "qwen35"
    Qwen2Config.__init__ = _patched_qwen2_init
    CONFIG_MAPPING.register("qwen35", Qwen2Config)

    # 3. GGUF COMPATIBILITY MAPPING
    import transformers.modeling_gguf_pytorch_utils as gu
    import transformers.integrations.ggml as gm
    if "qwen35" not in gu.GGUF_SUPPORTED_ARCHITECTURES: gu.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    gm.GGUF_CONFIG_MAPPING["qwen35"] = "qwen2"
    ```

---

## Phase 4: Database Shredding (Stay on A100 GPU)

1.  **Build the Layer Database**
    The F16 blueprint **must** be listed first as the Ground Truth.
    ```bash
    cd ~/gptq-gguf-toolkit/mapper
    rm -rf ep_database
    ./build_ep_database.sh --models \
        ../../abliterated-27B-F16.gguf \
        ../../base_quants/27B-Q6_K.gguf \
        ../../base_quants/27B-Q5_K.gguf \
        ../../base_quants/27B-Q4_K.gguf \
        ../../base_quants/27B-Q3_K.gguf \
        ../../base_quants/27B-Q2_K.gguf
    ```

2.  **Post-Database Cleanup**
    Perform this ONLY after the above command completes successfully. This frees ~150GB.
    ```bash
    rm -rf ~/gptq-gguf-toolkit/mapper/ep_database/models/*
    rm -rf ~/gptq-gguf-toolkit/mapper/ep_database/layers-hf/*
    ```

---

## Phase 5: EvoPress Stitching (Stay on A100 GPU)

Generate the mixed-precision models. The stitcher will automatically promote Mamba layers and Norms to high precision.

```bash
DB_PATH="./ep_database/layers-gguf"
ORIG_MODEL="../../abliterated-27B-F16.gguf"

# Generate Targets (3.5, 4.25, 5.0)
python3 gguf_stitcher.py "$DB_PATH" "../../Abliterated-27B-Mixed-3.5bpw.gguf" --original-model "$ORIG_MODEL" --default-quant-type Q3_K
python3 gguf_stitcher.py "$DB_PATH" "../../Abliterated-27B-Mixed-4.25bpw.gguf" --original-model "$ORIG_MODEL" --default-quant-type Q4_K
python3 gguf_stitcher.py "$DB_PATH" "../../Abliterated-27B-Mixed-5.0bpw.gguf" --original-model "$ORIG_MODEL" --default-quant-type Q5_K
```

---

## Phase 6: Release & Consolidation (Switch to CPU Mode)

1.  **Network-Safe Staging**
    Use `cat` to stream files to the upload folder to avoid FUSE-related `EOVERFLOW` errors.
    ```bash
    mkdir -p hf_upload
    
    # Move Mixed BPW Quants
    for f in ../../Abliterated-27B-Mixed-*.gguf; do
        cat "$f" > "hf_upload/$(basename "$f")" && rm "$f"
    done

    # Move Base Quants
    for q in Q2_K Q3_K Q4_K Q5_K Q6_K; do
        SOURCE="../../base_quants/27B-${q}.gguf"
        cat "$SOURCE" > "hf_upload/27B-${q}.gguf"
    done
    ```

2.  **Sequential HuggingFace Upload**
    Upload sequentially to prevent CPU Instance Out-Of-Memory (OOM) crashes.
    ```bash
    huggingface-cli login
    huggingface-cli repo create [username]/Qwen3.5-27B-Abliterated-EvoPress-GGUF
    
    cd hf_upload
    huggingface-cli upload [username]/[repo] README.md README.md --commit-message "Docs"
    
    for file in *.gguf; do
        huggingface-cli upload [username]/[repo] "$file" "$file" --commit-message "Upload $file"
    done
    ```

---

## Phase 7: Post-Build Maintenance (The Final Nuke)
Wipe all persistent storage artifacts to stop billing and leave a clean environment.

```bash
rm -rf ~/gptq-gguf-toolkit/mapper/ep_database
rm -rf ~/base_quants
rm -rf ~/hf_upload
rm ~/abliterated-27B-F16.gguf
```
