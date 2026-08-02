OpenAI LLM Plugin for Fess
==========================

## Overview

This plugin provides OpenAI integration for Fess's RAG (Retrieval-Augmented Generation) features. It enables Fess to use OpenAI's reasoning models (GPT-5 and the o1/o3/o4 series) for AI-powered search capabilities including intent detection, answer generation, document summarization, and FAQ handling.

## Download

See [Maven Repository](https://repo1.maven.org/maven2/org/codelibs/fess/fess-llm-openai/).

## Requirements

- Fess 15.x or later
- Java 21 or later
- OpenAI API key

## Installation

1. Download the plugin JAR from the Maven Repository
2. Place it in your Fess plugin directory
3. Restart Fess

For detailed instructions, see the [Plugin Administration Guide](https://fess.codelibs.org/14.19/admin/plugin-guide.html).

## Configuration

This plugin has two independently-configured clients that read from different files - see
[Content Chunk Embedding](#content-chunk-embedding) below for the second one. Set
`rag.llm.name=openai` in `conf/system.properties` (or as a `-Dfess.system.rag.llm.name` JVM
argument) to select this plugin as the RAG chat client - unlike every other property in the
table below, it is not read from `fess_config.properties`. The rest of the RAG chat client
(`OpenAiLlmClient`) properties in this section are configured in `fess_config.properties`:

| Property | Default | Description |
|----------|---------|-------------|
| `rag.chat.enabled` | `false` | Enable RAG chat feature |
| `rag.llm.openai.api.key` | - | OpenAI API key (required) |
| `rag.llm.openai.api.url` | `https://api.openai.com/v1` | OpenAI API endpoint URL |
| `rag.llm.openai.model` | `gpt-5-mini` | Model name (e.g., `gpt-5-nano`, `gpt-5`, `o3-mini`). See [Model Support](#model-support). |
| `rag.llm.openai.timeout` | `60000` | HTTP request timeout in milliseconds |
| `rag.llm.openai.availability.check.interval` | `60` | Interval (seconds) for checking API availability |
| `rag.llm.openai.chat.context.max.chars` | `4000` | Maximum characters for context in chat |
| `rag.llm.openai.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |

### Per-Prompt-Type Parameters

You can configure reasoning effort for each prompt type (applies to reasoning models: o1, o3, o4, gpt-5):

| Property | Description |
|----------|-------------|
| `rag.llm.openai.<promptType>.reasoning.effort` | Reasoning effort level (`low`, `medium`, `high`) |

### Content Chunk Embedding

When Fess's content-chunking RAG feature (`content_chunker.enabled=true`) is configured to use
this plugin as its embedding provider (`content_chunker.embedding.name=openai`), the following
properties configure `OpenAiEmbeddingClient`, which calls OpenAI's `POST /embeddings` endpoint.

**Unlike the `rag.llm.openai.*` properties above, every `content_chunker.embedding.openai.*`
property is configured in `conf/system.properties` (or passed as a `-Dfess.system.<key>` JVM
argument) - never in `fess_config.properties`.** This matches how Fess core reads every other
`content_chunker.*` property. Every value except `api.key` is visible read-only under admin
System Info > Config Info > App Properties (`api.key` is masked there). `timeout` and
`availability.check.interval` are read once, when the client is initialized at startup, so
changing either requires a restart; every other property below is re-read on each call and
takes effect immediately:

| Property | Default | Description |
|----------|---------|-------------|
| `content_chunker.embedding.openai.api.key` | - | OpenAI API key (required) |
| `content_chunker.embedding.openai.api.url` | `https://api.openai.com/v1` | OpenAI API endpoint URL |
| `content_chunker.embedding.openai.model` | `text-embedding-3-small` | Embedding model name. Only the `text-embedding-3-*` family sends the `dimensions` truncation parameter. |
| `content_chunker.embedding.openai.dimensions.enabled` | `auto` | Whether to send the `dimensions` truncation parameter. `auto` infers it from the model name (`text-embedding-3-*` only); `true` / `false` force it on or off. Needed on Azure OpenAI, where `model` is the operator-chosen deployment name rather than an OpenAI model id, so the name-based inference does not apply. |
| `content_chunker.embedding.openai.timeout` | `120000` | HTTP request timeout in milliseconds |
| `content_chunker.embedding.openai.availability.check.interval` | `60` | Interval (seconds) for checking API availability |
| `content_chunker.embedding.openai.retry.max` | `10` | Maximum HTTP retry attempts on `429` / `500` / `502` / `503` / `504` |
| `content_chunker.embedding.openai.retry.base.delay.ms` | `2000` | Base delay (ms) for exponential backoff between retries (overridden by a positive server `Retry-After` header when present; a `Retry-After: 0` falls back to exponential backoff) |
| `content_chunker.embedding.openai.retry.max.delay.ms` | `60000` | Hard cap (ms) on any single backoff sleep, bounding both exponential backoff and an honored `Retry-After` so a persistently rate-limited endpoint cannot stall content-chunk indexing |

Also requires the shared `content_chunker.embedding.dimension` property (embedding vector
dimension) to be set, independent of this plugin.

Unlike the sibling Ollama (text prefixes) and Gemini (`task_type`) embedding clients, OpenAI's
`/v1/embeddings` API has no document/query distinction mechanism, so `embedDocuments()` and
`embedQuery()` on this client are intentionally identical, both delegating to the same
request/response handling.

#### Request batching

`/v1/embeddings` caps a single request at 2048 inputs and 300,000 tokens across all of them, and
Fess core hands this client every chunk of a batch of documents at once - which for one large
document can exceed either cap on its own. The client therefore splits the input list into
sub-batches before calling the API. The split is invisible to callers and preserves input order.

Token counts are estimated from character classes rather than by running a real tokenizer:
characters that tokenize at CJK density count for 1.5 tokens each, everything else for a quarter
token. Japanese, Chinese and Korean text is roughly four times denser per character than English,
so a single Latin-calibrated ratio would under-count CJK content badly enough to push a full
sub-batch past the real limit.

The estimate is deliberately conservative, but it is still an estimate. If the provider rejects a
sub-batch as too large anyway, the client halves it and retries rather than failing the document:

```
WARN [Embedding:OPENAI] Estimated token count was too low for this batch;
     splitting 416 inputs into 208+208 and retrying.
```

Seeing this warning repeatedly means the estimate is mis-calibrated for your content - it costs
one wasted request per split. A single chunk that exceeds the per-request limit on its own cannot
be split any further and fails with a message pointing at
`content_chunker.length.chunk_size`; the per-input ceiling is 8192 tokens, so keep the chunk size
well below that.

### Model Support

The supported models are the reasoning families - `gpt-5*` and `o1*` / `o3*` / `o4*`.
`gpt-5-nano` is the cheapest of them and the one this plugin is exercised against;
`gpt-5-mini` is the default.

Any other model name still works if you configure it - `gpt-4o`, an Azure deployment name, or an
OpenAI-compatible gateway all take the second column below. They are simply not part of what this
plugin is tested against.

The plugin adapts the request to the model, keyed off the model name:

| | Reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`) | Any other model name |
|---|---|---|
| Token limit parameter | `max_completion_tokens` | `max_tokens` |
| `temperature` | Not sent - the API rejects any non-default value | Sent |
| `top_p`, `frequency_penalty`, `presence_penalty` | Not sent - the API rejects them outright | Sent |
| `reasoning_effort` | Supported | Not sent |
| Default `max_tokens` | Per-prompt-type default x `rag.llm.openai.reasoning.token.multiplier` (default `4`) | Per-prompt-type default |

Configuring `rag.llm.openai.<promptType>.top.p` (or either penalty) on a reasoning model logs a
warning and omits the parameter, rather than sending it and failing the whole chat with
`400 Unsupported parameter`.

The second column is keyed off the model name not matching a reasoning-family prefix, so it covers
both the older OpenAI families and any name the plugin has no knowledge of. Nothing rejects such a
model; it just receives the classic parameter set.

> **Note on `reasoning.effort`.** Raising it for the `intent` or `evaluation` prompt types without
> also raising the token budget can consume the entire allowance on internal reasoning and return
> empty content. The client logs `Chat finished abnormally ... reasoningTokens=N, contentLength=0`
> and Fess falls back to a plain search. The per-prompt-type defaults (`low` for these short
> classification prompts) exist for this reason.

## Features

- **Intent Detection** - Determines user intent (search, summary, FAQ, unclear) and generates Lucene queries
- **Answer Generation** - Generates answers based on search results with citation support
- **Document Summarization** - Summarizes specific documents
- **FAQ Handling** - Provides direct, concise answers to FAQ-type questions
- **Relevance Evaluation** - Identifies the most relevant documents for answer generation
- **Streaming Support** - Real-time response streaming via Server-Sent Events (SSE)
- **Availability Checking** - Validates API availability at configurable intervals
- **Reasoning Model Support** - Adaptive parameter handling for gpt-5 and o1/o3/o4 reasoning models

## OpenAI API Endpoints Used

- `GET /v1/models` - Lists available models for availability checking
- `POST /v1/chat/completions` - Performs chat completion (supports both standard and streaming modes)

## Development

### Building from Source

```bash
mvn clean package
```

### Running Tests

```bash
mvn test
```

## License

Apache License 2.0
