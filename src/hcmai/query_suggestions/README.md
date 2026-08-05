# Query suggestions

`hcmai.query_suggestions` owns operator-requested alternative search queries.
External callers use `SuggestionService` from `pipeline.py`; provider-specific
HTTP and GPU behavior stays behind adapters.

```text
query_suggestions/
├── pipeline.py              # SuggestionService and configured builder
├── prompting.py             # Prompt construction and bounded parsing
├── models/contracts.py      # SuggestionAdapter contract
└── adapters/
    ├── gpu.py               # Private GPU inference provider
    └── openai.py            # OpenAI-compatible provider
```

The active provider, timeout, model, and generation settings come from
`llm/config.yaml`. API keys are read only from the environment variable named
by `api_key_env`; never put credential values in YAML, logs, or frontend code.

Query suggestions are an explicit operator action, not part of the default
retrieval critical path. `SearchService.suggest()` delegates to the configured
`SuggestionService`; it does not change retrieval state or frame identity.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_query_suggestions.py
pyright src/hcmai/query_suggestions
```
