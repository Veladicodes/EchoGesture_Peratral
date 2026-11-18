#!/usr/bin/env bash
# Runs ON the GCP VM (uploaded in the bundle, driven by scripts/gcp_orchestrate.py over `gcloud compute ssh`).
#   remote_run.sh setup            install uv + Python 3.11 venv + requirements
#   remote_run.sh extract          unzip ~/archive.zip -> data/raw and run gcp_extract.py on all vCPUs
#   remote_run.sh train <args...>  start scripts/train_pipeline.py in the background (writes DONE/EXIT markers)
#   remote_run.sh poll             print the log tail and a __DONE__ marker when training has finished
set -e
cd ~/echogesture
export PATH="$HOME/.local/bin:$PATH"
case "$1" in
  setup)
    sudo apt-get update -y -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip libgl1 libglib2.0-0 unzip tar
    python3 -m pip install -q --user uv
    uv venv --python 3.11 .venv
    uv pip install -q --python .venv/bin/python -r requirements.txt
    .venv/bin/python -c 'import torch,onnx,onnxruntime,mediapipe,tensorflow;print("deps ok",torch.__version__,"cuda",torch.cuda.is_available())'
    ;;
  extract)
    mkdir -p data/raw
    unzip -q ~/archive.zip -d data/raw
    .venv/bin/python scripts/prepare_data.py
    .venv/bin/python scripts/gcp_extract.py --splits train val test --workers "$(nproc)"
    ;;
  train)
    shift
    rm -f DONE EXIT
    setsid nohup bash -c ".venv/bin/python scripts/train_pipeline.py $* > train_stdout.log 2>&1; echo \$? > EXIT; touch DONE" > /dev/null 2>&1 < /dev/null &
    echo started
    ;;
  poll)
    tail -n 2 train_stdout.log 2>/dev/null || echo "(no log yet)"
    if [ -f DONE ]; then echo "__DONE__ exit=$(cat EXIT)"; fi
    ;;
  *)
    echo "usage: remote_run.sh setup|extract|train|poll" >&2
    exit 2
    ;;
esac
