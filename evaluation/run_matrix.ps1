# Runs the comparison matrix, one configuration at a time (never in parallel,
# so measurements don't disturb each other). Every run: 3 warm-up passes
# discarded, 10 timed. Results land in evaluation_results/.
#
#   powershell -ExecutionPolicy Bypass -File evaluation\run_matrix.ps1 -Baseline <baseline json>
#
# "Tuned" Ollama is a second instance on port 11435 started by this script with
# OLLAMA_KV_CACHE_TYPE=q8_0 (8-bit KV cache; flash attention, which it needs,
# is already on by default in Ollama 0.40). The user's own Ollama service on
# port 11434 is never reconfigured or restarted.
param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [string]$Only = ""
)
# Not "Stop": Windows PowerShell 5.1 turns any line a native program writes to
# stderr (e.g. a Hugging Face warning) into a terminating error.
$ErrorActionPreference = "Continue"
# Every model is already cached; never reach out to the Hugging Face Hub.
$env:HF_HUB_OFFLINE = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = ".\.venv\Scripts\python.exe"
$default = "http://127.0.0.1:11434"
$tuned = "http://127.0.0.1:11435"

function Start-TunedOllama {
    if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) { return }
    $env:OLLAMA_HOST = "127.0.0.1:11435"
    $env:OLLAMA_KV_CACHE_TYPE = "q8_0"
    $env:OLLAMA_FLASH_ATTENTION = "1"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardError "data\eval\ollama-tuned.log" -RedirectStandardOutput "data\eval\ollama-tuned.out"
    Remove-Item Env:OLLAMA_HOST, Env:OLLAMA_KV_CACHE_TYPE, Env:OLLAMA_FLASH_ATTENTION
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) { return }
    }
    throw "tuned Ollama did not start"
}

function Run($label, [string[]]$arguments) {
    if ($Only -and ($label -notmatch $Only)) { return }
    Write-Host "=== $label ($(Get-Date -Format HH:mm:ss))"
    & $python -u -m @arguments --label $label 2>&1 |
        Where-Object { $_ -match '^\s*(embed_query|rerank|retrieval_total|first_token_ms|total_ms|prompt_tokens_per_s|output_tokens_per_s)\s|wrote |residency|Traceback|Error' }
}

# --- 1. Where should the embedding model live? (no reranker, no answering model)
Run "embed-cuda-fp32" @("evaluation.bench_pipeline", "--reranker", "none", "--llm", "none")
Run "embed-cuda-fp16" @("evaluation.bench_pipeline", "--reranker", "none", "--llm", "none", "--embed-dtype", "fp16")
Run "embed-cpu-fp32"  @("evaluation.bench_pipeline", "--reranker", "none", "--llm", "none", "--embed-device", "cpu")

# --- 2. Answering model: same prompts, fresh every request
$common = @("--baseline", $Baseline)
Run "answer-qwen3-8b-default"    (@("evaluation.bench_answering", "--model", "qwen3:8b",   "--resident-embedder", "cuda") + $common)
Run "answer-qwen3.5-4b-default"  (@("evaluation.bench_answering", "--model", "qwen3.5:4b", "--resident-embedder", "cuda") + $common)

# Ablations for Qwen3 8B: each tuning factor on its own.
Run "answer-qwen3-8b-embed-fp16" (@("evaluation.bench_answering", "--model", "qwen3:8b", "--resident-embedder", "cuda-fp16") + $common)
Run "answer-qwen3-8b-embed-cpu"  (@("evaluation.bench_answering", "--model", "qwen3:8b", "--resident-embedder", "cpu") + $common)
Start-TunedOllama
Run "answer-qwen3-8b-kv-q8"      (@("evaluation.bench_answering", "--model", "qwen3:8b", "--resident-embedder", "cuda",
                                    "--ollama-url", $tuned, "--other-ollama-url", $default,
                                    "--ollama-env", "OLLAMA_KV_CACHE_TYPE=q8_0") + $common)

# Tuned: 8-bit KV cache and the embedding model kept off the GPU.
Run "answer-qwen3-8b-tuned"      (@("evaluation.bench_answering", "--model", "qwen3:8b",   "--resident-embedder", "cpu",
                                    "--ollama-url", $tuned, "--other-ollama-url", $default,
                                    "--ollama-env", "OLLAMA_KV_CACHE_TYPE=q8_0") + $common)
Run "answer-qwen3.5-4b-tuned"    (@("evaluation.bench_answering", "--model", "qwen3.5:4b", "--resident-embedder", "cpu",
                                    "--ollama-url", $tuned, "--other-ollama-url", $default,
                                    "--ollama-env", "OLLAMA_KV_CACHE_TYPE=q8_0") + $common)
Write-Host "=== done ($(Get-Date -Format HH:mm:ss))"
