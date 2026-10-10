# Integration Testing Guide

## Overview

The `examples/integration_test.py` script tests the attack runners against a live LLM endpoint. However, **integration tests require special setup** due to cluster networking constraints.

## Endpoint Options

### Option 1: Cluster-Internal Endpoint (Recommended for on-cluster testing)

The model is served at:
```
https://<model-service>.models-as-a-service.svc.cluster.local:8000/v1
```

**Requirements:**
- Must run from **inside** the cluster (as a pod or job)
- Uses API key from a secret in your namespace
- Self-signed TLS certificate (handled by httpx with `verify=False`)

**To run from a cluster pod:**
```bash
# From a pod in your namespace
export TARGET_URL="https://<model-service>.models-as-a-service.svc.cluster.local:8000/v1"
export TARGET_API_KEY=$(oc -n <your-namespace> get secret <your-secret> -o jsonpath='{.data.API_KEY}' | base64 -d)

cd /path/to/cultural-bias-red-team
uv run examples/integration_test.py
```

### Option 2: External Endpoint

For external endpoints, configure with your own URL and credentials:
```bash
export TARGET_URL="https://your-endpoint/v1"
export TARGET_API_KEY="your-api-key"

uv run examples/integration_test.py
```

### Option 3: Local/Custom Endpoint

Test against any OpenAI-compatible endpoint:
```bash
export TARGET_URL="https://your-endpoint/v1"
export TARGET_API_KEY="your-api-key"
export ATTACKER_MODEL="your-model"
export JUDGE_MODEL="your-model"

uv run examples/integration_test.py
```

## Running from wbos as a Job

To run integration tests from within wbos:

```bash
# Create a test job manifest (examples/test-job.yaml)
# Apply it:
oc apply -f examples/test-job.yaml

# Check logs:
oc logs -f job/integration-test
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_URL` | cluster-internal | Target model endpoint |
| `TARGET_API_KEY` | auto-retrieved | API key for target |
|| `ATTACKER_MODEL` | `<your-model>` | Model for attacker |
| `JUDGE_MODEL` | `<your-model>` | Model for judge/scoring |

## Test Output

Expected output:
```
============================================================
CULTURAL-BIAS-RED-TEAM INTEGRATION TEST
============================================================

Target URL: https://...
Attacker model: <your-model>
Judge model: <your-model>

============================================================
TEST: TAP Attack
============================================================
✓ TAP completed
  Best score: 0.85
  Query count: 15
  Search trajectory: 3 iterations

============================================================
TEST: PAIR Attack
============================================================
✓ PAIR completed
  Best score: 0.78
  Query count: 10

============================================================
TEST: Crescendo Attack
============================================================
✓ Crescendo completed
  Best score: 0.72
  Turns: 5

============================================================
SUMMARY
============================================================
✓ TAP: ASR=0.85, queries=15
✓ PAIR: ASR=0.78, queries=10
✓ Crescendo: ASR=0.72, turns=5

All tests passed!
```

## Troubleshooting

### "No openai API key provided"

- The `any-llm-sdk` expects `OPENAI_API_KEY` when using OpenAI-compatible endpoints
- Set `TARGET_API_KEY` and ensure it's passed to the client

### "404 Not Found"

- The endpoint doesn't exist or is not accessible
- Check network connectivity (especially for cluster-internal endpoints)
- Verify the endpoint supports `/v1/chat/completions`

### "401 Unauthorized"

- API key is invalid or expired
- For MaaS, you need OAuth tokens, not API keys

### "Could not resolve host"

- Running outside the cluster trying to reach cluster-internal endpoint
- Use the external MaaS URL or run from inside wbos
