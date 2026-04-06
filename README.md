This document provides the technical pipeline for reproducing "Abliterated" models and quantizing them using **EvoPress**. Optimized for **Lightning AI Studios** and **Qwen 3.5 (Hybrid Mamba-Transformer)** architectures.

---

## 1. Phase 1: Environment & Tooling (CPU Mode)

**Hardware:** 4-Core CPU (Default/Free)

### 1.1 Install Dependencies
```bash
sudo apt-get update && sudo apt-get install -y git-lfs build-essential cmake
pip install -U "huggingface_hub[cli]" torch transformers accelerate safetensors datasets tqdm tiktoken dill
pip install "numpy<2" gguf --force-reinstall
pip install --no-deps --force-reinstall torchvision 
```

### 1.2 Clone Toolkits
```bash
# 1. Abliteration toolkit
git clone https://github.com/Orion-zhen/abliteration.git
cd abliteration && pip install -r requirements.txt && cd ..

# 2. llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j && pip install -r requirements.txt && cd ..

# 3. EvoPress (gptq-gguf-toolkit)
git clone https://github.com/IST-DASLab/gptq-gguf-toolkit
cd gptq-gguf-toolkit && pip install -r requirements.txt && cd ..
```

---

## 2. Phase 2: Abliteration & Preparation (GPU Mode)

**Hardware:** Switch to **1x A100 (80GB)**.

### 2.1 Run Abliteration
Create `abliteration/config.yaml` with the model path, then:
```bash
cd abliteration
python abliterate.py config.yaml
cd ..
```

### 2.2 Fix Tokenizer Config & Convert to F16
1. Edit `~/abliteration/abliterated_model/tokenizer_config.json`: Change `"tokenizer_class": "TokenizersBackend"` to `"tokenizer_class": "Qwen2TokenizerFast"`.
2. Convert to GGUF:
```bash
python llama.cpp/convert_hf_to_gguf.py ~/abliteration/abliterated_model --outfile /teamspace/studios/this_studio/abliterated-27B-F16.gguf
```

### 2.3 Generate "Gene Pool" Quants
```bash
mkdir -p /teamspace/studios/this_studio/base_quants
cd ~/llama.cpp
for q in q2_k q3_k q4_k q5_k q6_k; do
    OUTFILE="/teamspace/studios/this_studio/base_quants/27B-${q^^}.gguf"
    [ ! -f "$OUTFILE" ] && ./build/bin/llama-quantize /teamspace/studios/this_studio/abliterated-27B-F16.gguf "$OUTFILE" "$q"
done
```

---

## 3. Phase 3: Infrastructure Patches (FUSE & Hybrid Arch)

Lightning AI Studios use FUSE-based network drives. This patch fixes storage write hangs (Errno 75) and injects architectural metadata for Qwen 3.5.

```bash
cd ~/gptq-gguf-toolkit/mapper
git checkout gguf_splitter.py # Ensure clean state

cat << 'EOF' > patch_header.py
#!/usr/bin/env python3
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
            chunk_size = 1024 * 1024 * 256
            try:
                view = memoryview(data)
                for i in range(0, len(view), chunk_size):
                    _orig_write(view[i:i+chunk_size])
                return len(data)
            except Exception: return _orig_write(data)
        f.write = _chunked_write
    return f
builtins.open = _patched_open

# 2. BLOCK BROKEN TORCHVISION
_orig_find_spec = importlib.util.find_spec
def _patched_find_spec(name, package=None):
    if name.startswith("torchvision"): return None
    return _orig_find_spec(name, package)
importlib.util.find_spec = _patched_find_spec

# 3. HYBRID ARCHITECTURE OVERRIDE (Qwen 3.5 27B)
_orig_qwen2_init = Qwen2Config.__init__
def _patched_qwen2_init(self, **kwargs):
    kwargs.update({"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64, 
                   "num_attention_heads": 40, "num_key_value_heads": 8, "max_position_embeddings": 131072})
    _orig_qwen2_init(self, **kwargs)
    self.model_type = "qwen35"
Qwen2Config.__init__ = _patched_qwen2_init
CONFIG_MAPPING.register("qwen35", Qwen2Config)

import transformers.modeling_gguf_pytorch_utils as gu
import transformers.integrations.ggml as gm
if "qwen35" not in gu.GGUF_SUPPORTED_ARCHITECTURES: gu.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
gm.GGUF_CONFIG_MAPPING["qwen35"] = "qwen2"
EOF

tail -n +2 gguf_splitter.py > clean_splitter.py
cat patch_header.py clean_splitter.py > gguf_splitter.py
rm patch_header.py clean_splitter.py
chmod +x gguf_splitter.py
```

---

## 4. Phase 4: Database Build

The splitter shreds quants into individual tensors. Using symlinks (`ln -sf`) prevents disk exhaustion.

```bash
cd ~/gptq-gguf-toolkit/mapper
rm -rf ./ep_database
sed -i 's/cp "$model_path"/ln -sf "$(realpath "$model_path")"/g' build_ep_database.sh

./build_ep_database.sh --models \
    /teamspace/studios/this_studio/abliterated-27B-F16.gguf \
    /teamspace/studios/this_studio/base_quants/27B-Q2_K.gguf \
    /teamspace/studios/this_studio/base_quants/27B-Q3_K.gguf \
    /teamspace/studios/this_studio/base_quants/27B-Q4_K.gguf \
    /teamspace/studios/this_studio/base_quants/27B-Q5_K.gguf \
    /teamspace/studios/this_studio/base_quants/27B-Q6_K.gguf
```

---

## 5. Phase 5: Reconstruction (Automated Mixed-Precision)

The stitcher handles Hybrid Mamba architectures by pulling high-precision F16/F32 versions for Norms and sensitive Mamba tensors automatically.

```bash
cd ~/gptq-gguf-toolkit/mapper
STUDIO_PATH="/teamspace/studios/this_studio"
DB_PATH="./ep_database/layers-gguf"
ORIG_MODEL="$STUDIO_PATH/abliterated-27B-F16.gguf"

# 3.5 BPW
python3 gguf_stitcher.py "$DB_PATH" "$STUDIO_PATH/Abliterated-27B-Mixed-3.5bpw.gguf" \
    --original-model "$ORIG_MODEL" --default-quant-type Q3_K

# 4.25 BPW
python3 gguf_stitcher.py "$DB_PATH" "$STUDIO_PATH/Abliterated-27B-Mixed-4.25bpw.gguf" \
    --original-model "$ORIG_MODEL" --default-quant-type Q4_K

# 5.0 BPW
python3 gguf_stitcher.py "$DB_PATH" "$STUDIO_PATH/Abliterated-27B-Mixed-5.0bpw.gguf" \
    --original-model "$ORIG_MODEL" --default-quant-type Q5_K
```

---

## 6. Phase 6: Consolidation & Staging

Cloud filesystems often throw `EOVERFLOW` when using `mv` on files >2GB. To consolidate quants into a clean staging folder while avoiding system caches:

```bash
# 1. Create clean staging area
mkdir -p /teamspace/studios/this_studio/hf_upload

# 2. Consolidate Base Quants via streaming
for q in Q2_K Q3_K Q4_K Q5_K Q6_K; do
    SOURCE="/teamspace/studios/this_studio/base_quants/27B-${q}.gguf"
    DEST="/teamspace/studios/this_studio/hf_upload/27B-${q}.gguf"
    cat "$SOURCE" > "$DEST"
done

# 3. Move Mixed Precision Quants
mv /teamspace/studios/this_studio/Abliterated-27B-Mixed-*.gguf /teamspace/studios/this_studio/hf_upload/
```

---

## 7. Phase 7: Documentation & Release (CPU Mode)

**Hardware:** Switch to **CPU Mode**.

### 7.1 Create Model Card (README.md)
```bash
cat << 'EOF' > /teamspace/studios/this_studio/hf_upload/README.md
---
license: apache-2.0
language:
  - en
  - zh
  - ko
tags:
  - gguf
  - mamba
  - uncensored
  - reasoning
  - chain-of-thought
  - qwen3.5
base_model: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2
pipeline_tag: image-text-to-text
datasets:
  - nohurry/Opus-4.6-Reasoning-3000x-filtered
  - Jackrong/Qwen3.5-reasoning-700x
  - Roman1111111/claude-opus-4.6-10000x
---
# Title
Insert your content/description here
EOF
```

### 7.2 Upload to HuggingFace
```bash
huggingface-cli login

# 1. Create the Repo (if not exists)
huggingface-cli repo create [username]/[repo-name]

# 2. Upload sequentially to prevent CPU Instance Out-Of-Memory (OOM) crashes
cd /teamspace/studios/this_studio/hf_upload
huggingface-cli upload [username]/[repo-name] README.md README.md --commit-message "Initial Model Card"

for file in *.gguf; do
    huggingface-cli upload [username]/[repo-name] "$file" "$file" --commit-message "Upload $file"
done
```

### 7.3 Post-Upload Maintenance & Space Recovery
If the F16 "Blueprint" file was accidentally uploaded or is still consuming disk space:

```bash
# 1. Remove F16 from HuggingFace (Remotely)
huggingface-cli delete-file [username]/[repo-name] abliterated-27B-F16.gguf

# 2. Free up local persistent storage
rm /teamspace/studios/this_studio/abliterated-27B-F16.gguf
rm -rf /teamspace/studios/this_studio/base_quants
rm -rf /teamspace/studios/this_studio/hf_upload
rm -rf ~/gptq-gguf-toolkit/mapper/ep_database
```
