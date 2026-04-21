# On the Reliability of Computer Use Agents

Code for reproducing the experiments in [*"On the Reliability of Computer Use Agents"*](https://arxiv.org/abs/2604.17849)

This repository studies the sources of unreliability in computer-use agents through three factors: stochasticity during execution, ambiguity in task specification, and variability in agent behavior. All experiments are conducted on [OSWorld](https://github.com/xlang-ai/OSWorld) using repeated executions (N=3) of the same task with paired statistical tests.

## Overview

- **`gui_agents/`** — Agent S3 framework: agent classes, LLM engines (OpenAI, Anthropic, Azure, vLLM, OpenRouter, etc.), system prompts, and behavior narrator.
- **`osworld/`** — Forked OSWorld evaluation environment: VM interface, task evaluators, and model-specific agent implementations (Claude, Kimi, OpenCUA, UI-TARS).
- **`agents/`** — Run scripts for each model: `run_s3.py` (GPT-5, Qwen), `run_claude.py`, `run_kimi.py`, `run_opencua.py`, `run_uitars.py`.
- **`experiments/`** — Experiment-specific scripts: instruction clarification, plan generation, fact caption generation, plan feedback extraction, user simulator, and environment perturbation definitions.
- **`compute_metrics.py`** — Compute reliability metrics (Pass^k, McNemar, Wilcoxon) from result directories.
- **`evaluation_examples/`** — Task configurations with original and clarified instructions.

## Setup

### 1. OSWorld Environment

Follow the [OSWorld setup guide](https://github.com/xlang-ai/OSWorld) to configure your VM provider. We use AWS in our experiments, but VMware, Azure, and other providers are also supported. See `osworld/README.md` and `osworld/PUBLIC_EVALUATION_GUIDELINE.md` for detailed instructions.

### 2. Install Dependencies

```bash
pip install -r osworld/requirements.txt  # OSWorld dependencies
```

Also install `tesseract` for OCR support (`brew install tesseract` on macOS, `apt install tesseract-ocr` on Ubuntu).

### 3. API Keys

Copy `.envrc.template` to `.envrc` and fill in your API keys, then run `source .envrc`:

- **Azure OpenAI** (GPT-5): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- **Anthropic** (Claude Sonnet 4.6): `ANTHROPIC_API_KEY`
- **OpenRouter** (Kimi K2.5): `OPENROUTER_API_KEY`
- **Open-source models**: Self-hosted via vLLM (use `--model_url`)

### 4. Open-Source Model Hosting (vLLM)

For deterministic experiments with open-source models, serve them via vLLM with batch-invariant mode:

```bash
# UI-TARS-1.5-7B
VLLM_BATCH_INVARIANT=1 vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
  --served-model-name uitars15-7b \
  --host 127.0.0.1 --port 8000 \
  --data-parallel-size 4 --tensor-parallel-size 1 \
  --attention-backend FLASH_ATTN

# OpenCUA-7B
VLLM_BATCH_INVARIANT=1 vllm serve xlangai/OpenCUA-7B \
  --served-model-name opencua-7b \
  --host 127.0.0.1 --port 8001 \
  --trust-remote-code \
  --data-parallel-size 4 --tensor-parallel-size 1 \
  --attention-backend FLASH_ATTN

# Qwen3-VL-8B-Instruct
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --host 127.0.0.1 --port 8002 \
  --data-parallel-size 4 --tensor-parallel-size 1 \
  --attention-backend FLASH_ATTN
```

`VLLM_BATCH_INVARIANT=1` ensures deterministic outputs at temperature 0 regardless of batching, which is required for the deterministic decoding experiments.

## Running Experiments

Agent S3 runs (GPT-5, Qwen) use `agents/run_s3.py`. Other models have dedicated scripts in `agents/`. Agent S3 requires a separate grounding model — we used UI-TARS-1.5-7B for grounding throughout our experiments.

### Stochastic Decoding and Execution Noise

#### Deterministic Agent Execution

**Baseline (stochastic decoding):** Run with default temperature.

```bash
# Example: Qwen (S3) baseline
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider vllm --model qwen3-vl-8b \
  --model_url http://127.0.0.1:8002/v1 \
  --model_temperature 0.7 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --test_all_meta_path evaluation_examples/test_all.json \
  --result_dir results/baseline/run1
```

**Deterministic decoding:** Set temperature to 0 and serve models with `VLLM_BATCH_INVARIANT=1`.

```bash
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider vllm --model qwen3-vl-8b \
  --model_url http://127.0.0.1:8002/v1 \
  --model_temperature 0 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --test_all_meta_path evaluation_examples/test_all.json \
  --result_dir results/deterministic/run1
```

**Strategy determinism:** First generate plans, then run with plan injection.

```bash
# Step 1: Generate plans from initial screenshot
python experiments/generate_plans.py \
  --provider_name aws --headless --num_envs 5 \
  --model_provider vllm --model qwen3-vl-8b \
  --model_url http://127.0.0.1:8002/v1 \
  --model_temperature 0 \
  --result_dir ./plan_cache/qwen \
  --test_all_meta_path evaluation_examples/test_all.json

# Step 2: Run with pre-generated plans
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider vllm --model qwen3-vl-8b \
  --model_url http://127.0.0.1:8002/v1 \
  --model_temperature 0 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --enable_plan --plan_cache_dir ./plan_cache/qwen \
  --test_all_meta_path evaluation_examples/test_all.json \
  --result_dir results/strategy_determinism/run1
```

#### Sensitivity to Environment Noise

Environment perturbations are defined in `experiments/perturbations.py` and applied at the VM level via gsettings commands before each task. Two perturbation sets modify wallpaper, cursor size, dock position, icon theme, and timezone. The `--perturbations` flag is supported on `run_s3.py`, `run_claude.py`, and `run_kimi.py`.

```bash
# Baseline: 3 runs in unperturbed environment
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider azure --model gpt-5 \
  --model_temperature 1 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir results/env_baseline/run1

# Perturbed: runs with perturbation set 1 or 2
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider azure --model gpt-5 \
  --model_temperature 1 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --perturbations 1 \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir results/env_perturbed/perturb_set1
```

The cross-environment comparison uses 1 baseline run paired with the 2 perturbed runs to compute reliability metrics.

### Instruction Ambiguity

#### Clarification Before Execution

Clarified instructions are provided in `evaluation_examples/examples_clarified/`. To regenerate them:

```bash
python experiments/clarify_instructions.py \
  evaluation_examples/test_nogdrive.json \
  --model gpt-5 \
  --output-dir evaluation_examples/examples_clarified \
  --max-workers 5
```

To run with clarified instructions, use `--examples_dir examples_clarified`:

```bash
# GPT-5 (S3) with clarified instructions
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider azure --model gpt-5 \
  --model_temperature 1 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --examples_dir examples_clarified \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir results/clarified/run1

# Claude with clarified instructions
python agents/run_claude.py \
  --headless --observation_type screenshot \
  --model claude-sonnet-4-6 \
  --examples_dir examples_clarified \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --max_steps 100 --num_envs 30 \
  --result_dir results/clarified_claude/run1

# Kimi with clarified instructions
python agents/run_kimi.py \
  --headless --observation_type screenshot \
  --model kimi-k2.5 --thinking \
  --examples_dir examples_clarified \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --max_steps 100 --num_envs 30 \
  --result_dir results/clarified_kimi/run1
```

#### Clarification During User Interaction

The user simulator (`experiments/user_simulator.py`) provides targeted feedback on failed executions based on the agent's trajectory, task instruction, and evaluation signals. Two methods are supported via `--method`:

- **`binary_retry`**: Agent receives a generic failure signal and retries.
- **`clarify_upon_retry`**: Agent receives targeted feedback from the user simulator describing what went wrong.

```bash
# Binary retry baseline (no user simulator needed)
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider azure --model gpt-5 \
  --model_temperature 1 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --method binary_retry --max_retries 5 --max_total_steps 100 \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir results/binary_retry/run1

# Clarify upon retry (user simulator generates feedback after failures)
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider azure --model gpt-5 \
  --model_temperature 1 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --method clarify_upon_retry --max_retries 5 --max_total_steps 100 \
  --user_sim_model gpt-5 \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir results/clarify_upon_retry/run1
```

The `--max_total_steps` budget is shared across all retries. The user simulator model is configured via `--user_sim_model`.

### Planning Variability

#### Plan Extraction (Iteration 0 -> Iteration 1)

Starting from Iteration 0 (runs with clarified instructions), perform multiple rollouts and extract feedback:

```bash
# Step 1: Run Iteration 0 (3 runs with clarified instructions)
# (same as clarification commands above)

# Step 2: Generate fact captions from rollout screenshots (for tasks with variance)
# Results should be organized as: <base_dir>/trial0_runs/run1, run2, run3
python experiments/generate_facts.py \
  --results-dirs \
    results/clarified_trials/trial0_runs/run1/<action_space>/<observation_type>/<model> \
    results/clarified_trials/trial0_runs/run2/<action_space>/<observation_type>/<model> \
    results/clarified_trials/trial0_runs/run3/<action_space>/<observation_type>/<model> \
  --model gpt-5 --engine-type azure --temperature 1

# Step 3: Extract plan feedback from Iteration 0 rollouts
python experiments/extract_plan_feedback.py \
  --trial-number 0 \
  --base-results-dir results/clarified_trials \
  --judge-model gpt-5 \
  --judge-engine azure \
  --temperature 1

# Step 4: Run Iteration 1 with plan feedback
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider azure --model gpt-5 \
  --model_temperature 1 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --examples_dir examples_clarified \
  --enable_plan_feedback \
  --plan_feedback_file results/clarified_trials/trial0_feedback.jsonl \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir results/clarified_trials/trial1_runs/run1
```

#### Iterative Plan Refinement (Iteration 1 -> Iteration 2)

```bash
# Step 1: Generate fact captions from Iteration 1 rollouts
python experiments/generate_facts.py \
  --results-dirs \
    results/clarified_trials/trial1_runs/run1/<action_space>/<observation_type>/<model> \
    results/clarified_trials/trial1_runs/run2/<action_space>/<observation_type>/<model> \
    results/clarified_trials/trial1_runs/run3/<action_space>/<observation_type>/<model> \
  --model gpt-5 --engine-type azure --temperature 1

# Step 2: Extract updated feedback from Iteration 1 rollouts
python experiments/extract_plan_feedback.py \
  --trial-number 1 \
  --base-results-dir results/clarified_trials \
  --judge-model gpt-5 \
  --judge-engine azure \
  --temperature 1

# Step 3: Run Iteration 2 with refined feedback
python agents/run_s3.py \
  --provider_name aws --headless --num_envs 30 \
  --model_provider azure --model gpt-5 \
  --model_temperature 1 \
  --ground_provider <YOUR_GROUNDING_PROVIDER> \
  --ground_url <YOUR_GROUNDING_URL> \
  --ground_model <YOUR_GROUNDING_MODEL> \
  --grounding_width 1920 --grounding_height 1080 \
  --max_steps 100 \
  --examples_dir examples_clarified \
  --enable_plan_feedback \
  --plan_feedback_file results/clarified_trials/trial1_feedback.jsonl \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir results/clarified_trials/trial2_runs/run1
```

### Pre-computed Artifacts

The following pre-computed artifacts are required for some experiments and will be made available for download.

- **Plan caches**: Pre-generated plans for strategy determinism and plan extraction experiments
- **Plan feedback**: JSONL files with extracted feedback for Iterations 1 and 2
- **Environment perturbation VM snapshots**: Pre-configured VM images with cosmetic modifications

## Computing Metrics

Results from each run are stored in the result directory with the following structure:

```
results/<action_space>/<observation_type>/<model>/<domain>/<example_id>/
├── traj.jsonl          # Trajectory (actions, observations per step)
├── result.txt          # Final score (0.0 or 1.0)
├── instruction.txt     # Task instruction used
├── step_0.png          # Initial screenshot
├── step_N_*.png        # Screenshots after each action
├── plan_info.json      # Plan metadata (if --enable_plan)
└── plan_feedback_info.json # Plan feedback metadata (if --enable_plan_feedback)
```

### Reliability Metrics

Use `compute_metrics.py` to compute Pass^1, Pass^3, and paired statistical tests (McNemar, Wilcoxon):

```bash
# Single setting
python compute_metrics.py \
  --run-dirs results/experiment/run1 results/experiment/run2 results/experiment/run3 \
  --example-json evaluation_examples/test_nogdrive.json

# Compare two settings (baseline vs new)
python compute_metrics.py \
  --run-dirs results/baseline/run1 results/baseline/run2 results/baseline/run3 \
  --compare-dirs results/new/run1 results/new/run2 results/new/run3 \
  --example-json evaluation_examples/test_nogdrive.json
```

Output:
```
Setting     | Pass^1  | Pass^3  | b-c     | Δcx
------------+---------+---------+---------+--------
Baseline    | ...     | ...     |         |
New         | ...     | ...     | ...     | ...
```

An asterisk (`*`) denotes statistical significance at p < 0.05. Use `--verbose` for detailed per-run loading output.

## Citation

<!-- TODO: Update with final citation details -->
```bibtex
@misc{gonzalezpumariega2026reliabilitycomputeruseagents,
      title={On the Reliability of Computer Use Agents}, 
      author={Gonzalo Gonzalez-Pumariega and Saaket Agashe and Jiachen Yang and Ang Li and Xin Eric Wang},
      year={2026},
      eprint={2604.17849},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.17849}, 
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
