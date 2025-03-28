# Verifiers: Reinforcement Learning with LLMs in Verifiable Environments

This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments. 

**Note:** This repository in its current state should be viewed as "research code", and is not guaranteed to yield optimal training results. RL is delicate, expect that experimentation will be required. The examples are intended for illustrative purposes of usage patterns rather than stable training recipes. You are encouraged to write your own standalone training scripts, modifying environments/datasets/rewards/configs as needed for your use case.

## Updated Features

- [X] New Example / Dataset: Wiki-Search + TriviaQA (check verifiers/examples/triviaqa_search.py)

- [X] This example supports local search so we can avoid the rate limit issue when using duckduckgo and also the cost for calling api when using brave

![Dataset format (Question and Acceptable Answers)](Figures/triviaqa_dataset.png)

![Results of Multi-Step GRPO on TriviaQA + Wiki-Search](Figures/triviaqa_results.png)


## Installation

PyPI [coming soon](https://pypi.org/project/verifiers/) once a couple more features are added, just clone it for now and run:
```
git clone https://github.com/SiliangZeng/verifiers.git
cd verifiers
git checkout sz-dev-triviaqa
uv sync
uv pip install pyserini
uv pip install flash-attn --no-build-isolation
```

If you want to run the triviaqa-search example, please install java correctly:
```
# Install Java 21 (required for Pyserini-Search), if you want to run the TriviaQA search example
# First remove any old Java versions
apt-get remove --purge openjdk*

# Add Java 21 repository
apt-get update
apt-get install -y wget gpg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list

# Install Java 21
apt-get update
apt-get install -y temurin-21-jdk

# Set Java environment variables
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export JAVA_OPTS="--add-modules jdk.incubator.vector -Xms4g -Xmx12g -XX:+UseG1GC"
export JVM_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server/libjvm.so
export LD_LIBRARY_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server:$LD_LIBRARY_PATH

# Verify Java installation
java -version
```

Once requirement for both env requirement and java is satisfied, you can run the triviaqa-search example:
```
./quickstart.sh
```


Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`).

Tested with Python 3.11 and this [image](https://hub.docker.com/layers/pytorch/pytorch/2.5.1-cuda12.1-cudnn9-devel/images/sha256-e8e63dd7baca894ba11fe1ba48a52a550793c8974f89b533d697784dd20a4dc0). If you encounter version issues, please confirm that you are able to run basic TRL training in your environment before opening an issue. `flash-attn` and `liger-kernel` are used for performance reasons. Recommended usage is via `accelerate` with DeepSpeed ZeRO 3 ([example config](https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/deepspeed_zero3.yaml)) but `torchrun` works in my tests as well. You should really be using `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`). I don't have the bandwidth to help debug your version issues if you're using `pip`, sorry.

## Usage

```python
# script.py
import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.ToolEnv(
    dataset="gsm8k",
    few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[calculator],
    max_steps=3
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    reward_funcs=vf_env.get_rubric(),
    args=vf.get_default_grpo_config(run_name="gsm8k-calc", num_gpus=2),
    train_dataset=vf_env.get_dataset(),
)
trainer.train()
```
See `examples` for additional usage examples. 

To create your own multi-step environment, inherit from `MultiStepEnv` and implement:
```python
def get_dataset(self, **kwargs: Any) -> Dataset:
    pass

def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
    pass

def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
    pass

def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
    pass
```

### Launch Commands
Accelerate:
```bash
accelerate launch --config_file /path/to/deepspeed_zero3.yaml --num_processes [N-1] script.py
```
Torchrun:
```bash
torchrun --nproc_per_node=[N-1] script.py
```

## Features
- [X] Environments: `SimpleEnv`, `MathEnv`, `DoubleCheckEnv`, `CodeEnv`, `ToolEnv`
- [X] Multi-step execution in `CodeEnv` and `ToolEnv`
- [X] Dataset formatting + XML parsers
- [X] Basic ubrics for math/code correctness + formatting
- [X] Defaults for GRPO, model, tokenizer, etc.

## Roadmap

There are a number of features we're planning to support in the near future:
- [ ] Integrated evals
- [ ] TextArena games
- [ ] LLM judges
- [ ] Claude-generated rubrics
- [ ] A range of other environments (suggestions welcome!)
- [ ] PPO
- [ ] Potential interoperability with other RL libraries (veRL, OpenRLHF, open-instruct, oat, etc.)

Community contributions are appreciated and encouraged!

## Citation

If you use this code in your research, please cite:

```bibtex
@article{brown2025verifiers,
  title={Verifiers: Reinforcement Learning with LLMs in Verifiable Environments},
  author={Brown, William},
  year={2025}
}
```
