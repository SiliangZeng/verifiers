git clone https://github.com/willccbb/verifiers.git
cd verifiers
uv sync
uv pip install flash-attn --no-build-isolation
source .venv/bin/activate

#accelerate launch --config-file configs/zero3.yaml --num-processes 7 verifiers/examples/gsm8k_calculator.py

accelerate launch --config-file configs/zero3.yaml --num-processes 7 verifiers/examples/gsm8k_llm_judge.py