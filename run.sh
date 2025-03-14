source .venv/bin/activate

accelerate launch --config-file configs/zero3.yaml --num-processes 7 verifiers/examples/openbookqa_search.py