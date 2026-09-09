# cultural-bias-red-team

Synthetic red-team probe generation for evaluating cultural bias in LLMs.

Implements the **Singapore AI Safety Red Teaming Challenge methodology** (IMDA + Humane Intelligence, Feb 2025) using [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

## Why

Standard red-teaming focuses on adversarial prompts from an attacker perspective. Singapore's challenge found:

- **86.1%** of successful exploits came from **single-turn benign prompts** — normal everyday user interactions
- **67.2%** exploit success rate in regional languages vs **49.2%** in English
- **Positive framing** ("which city is *richest*?") bypasses more guardrails than negative framing
- Cultural bias appears in daily use, not only adversarial scenarios

This tool generates both probe types, grounded in specific cultural contexts rather than generic demographics.

## Quick start

```bash
git clone https://github.com/williamcaban/cultural-bias-red-team.git
cd cultural-bias-red-team

# Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

cp .env.example .env        # add OPENAI_API_KEY (and optionally OPENAI_BASE_URL for vLLM/RHOAI)

# Validate and preview
uv run data-designer validate pipeline.py
uv run data-designer preview  pipeline.py

# Generate 500 probes (Singapore 9-country taxonomy, default)
uv run data-designer create pipeline.py --num-records 500
```

## Comparison modes

Use a `BIAS_SPEC` environment variable to run a two-group comparison instead of the Singapore taxonomy:

```bash
# US vs Mexico
BIAS_SPEC=us-mexico uv run data-designer create pipeline.py --num-records 200

# US (mainland) vs Puerto Rico
BIAS_SPEC=us-puerto-rico uv run data-designer create pipeline.py --num-records 200

# US White vs US Brown/Latino
BIAS_SPEC=us-white-brown uv run data-designer create pipeline.py --num-records 200
```

## Available specs

| Spec slug | Description |
|---|---|
| `singapore` (default) | 9-country APAC taxonomy — 5 bias categories grounded in Singapore red team findings |
| `us-mexico` | National identity + race/ethnicity bias: US Americans vs Mexicans |
| `us-puerto-rico` | Citizenship + national identity: mainland US vs Puerto Ricans |
| `us-white-brown` | Intra-US racial bias: White (non-Hispanic) vs Brown/Latino Americans |

## Adding a new comparison

Create `specs/<name>.py` that exports `SPEC = BiasComparisonSpec(...)`. No other files need to change.

```bash
# After creating the file:
uv run pytest                # validates the spec format
# Open a PR — CI runs automatically
```

Full instructions: [`specs/CONTRIBUTING.md`](specs/CONTRIBUTING.md)

## Output

Each probe row includes:

| Column | Description |
|---|---|
| `policy_concept` | Bias category (e.g., "Gender Bias") |
| `probe_mode` | `adversarial` or `benign` |
| `region` | Geographic context grounding the probe |
| `prompt_strategy` | Singapore strategy used (fill-in-blank, positive-framing, etc.) |
| `adversarial_probe` | Structured output with `prompt` + explanations (when mode=adversarial) |
| `benign_probe` | Structured output with `prompt` + `why_benign` + `bias_mechanism` (when mode=benign) |
| `probe_quality_score` | 0–3 score from LLM judge (Singapore RT-SG-05 sliding scale) |
| `*__trace` | Audit trail — exact system prompt + rendered user prompt sent to model |

In comparison mode, also includes `group_1_name`, `group_2_name`, `group_1_ref`, `group_2_ref`, `comparison_topic`.

## Development

```bash
uv sync                     # install all deps
uv run ruff check .         # lint
uv run ruff format .        # format
uv run pytest               # unit tests (no LLM required)
```

## Using with RHOAI / vLLM

```bash
export OPENAI_BASE_URL=https://your-vllm-endpoint/v1
export OPENAI_API_KEY=your-key
export MODEL_NAME=granite-3.3-8b-instruct

uv run data-designer create pipeline.py --num-records 500
```

## Methodology

**Singapore AI Safety Red Teaming Challenge** — world's first multicultural, multilingual AI safety red teaming exercise. 9 countries, 54 in-person experts + 308 virtual participants, 1,335–1,887 successful exploits per round.

**Source:** [IMDA Evaluation Report, February 2025](https://www.imda.gov.sg/-/media/imda/files/about/emerging-tech-and-research/artificial-intelligence/singapore-ai-safety-red-teaming-challenge-evaluation-report.pdf)

## License

Apache 2.0
