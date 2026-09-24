#!/usr/bin/env bash
# Download OPT weights into a Hugging Face cache under /home.
# Only the PyTorch .bin shards are fetched; the repos also carry Flax/TF copies we don't use.
set -euo pipefail

HF_HOME=/home/huggingface
MODELS=(facebook/opt-6.7b facebook/opt-30b meta-llama/Llama-2-7b-hf)
declare -A SIZE_GB=([facebook/opt-6.7b]=14 [facebook/opt-30b]=61 [meta-llama/Llama-2-7b-hf]=27)

export HF_HOME
# /home is root-owned, so create the cache with sudo if needed and hand it to the current user.
[[ -w $HF_HOME ]] || { sudo mkdir -p "$HF_HOME" && sudo chown "$(id -u):$(id -g)" "$HF_HOME"; }

# Persist HF_HOME for new shells so the benchmarks load from this cache instead of re-downloading.
sed -i '/^export HF_HOME=/d' ~/.bashrc
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

for model in "${MODELS[@]}"; do
    # Don't count the part of the model that is already cached against free space.
    have=$({ du -sLb "$HF_HOME/hub/models--${model//\//--}/snapshots" 2>/dev/null || true; } | cut -f1)
    need=$(( ${SIZE_GB[$model]} * 10**9 - ${have:-0} ))
    avail=$(df --output=avail -B1 "$HF_HOME" | tail -1)
    if (( need > avail )); then
        echo "Not enough space for $model: needs ~$(( need / 10**9 )) GB more, $(( avail / 10**9 )) GB free in $HF_HOME" >&2
        exit 1
    fi
    hf download "$model" --exclude "*.msgpack" --exclude "*.h5"
done

echo "Done. Run 'source ~/.bashrc' (or open a new terminal) before running the benchmarks."
