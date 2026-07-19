#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

export LLM_PROVIDER=custom
export CUSTOM_BASE_URL="${CUSTOM_BASE_URL:-https://api-hub.hellorobotaxi.top/v1}"
export CUSTOM_PROCESS_MODEL="${CUSTOM_PROCESS_MODEL:-qwen3.7-max}"
export CUSTOM_REFINE_MODEL="${CUSTOM_REFINE_MODEL:-qwen3.7-max}"
export LLM_CONCURRENCY="${LLM_CONCURRENCY:-5}"
export LLM_REQUEST_TIMEOUT="${LLM_REQUEST_TIMEOUT:-180}"

if [[ -z "${CUSTOM_API_KEY:-}" ]]; then
    read -r -s -p "Custom API key: " CUSTOM_API_KEY
    printf '\n'
    export CUSTOM_API_KEY
fi

if [[ -z "${GITHUB_TOKEN:-}" ]] && command -v gh >/dev/null 2>&1; then
    GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
    export GITHUB_TOKEN
fi

cd "$PROJECT_DIR"

printf 'Provider: %s\nEndpoint: %s\nModel: %s\nLLM concurrency: %s\n' \
    "$LLM_PROVIDER" \
    "$CUSTOM_BASE_URL" \
    "$CUSTOM_PROCESS_MODEL" \
    "$LLM_CONCURRENCY"
printf 'Checkpoint resume is enabled; rerun this script after any interruption.\n\n'

exec uv run howtocook -v detect --force --concurrency 5 "$@"
