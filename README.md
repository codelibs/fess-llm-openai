OpenAI LLM Plugin for Fess
==========================

## Overview

This plugin provides OpenAI integration for Fess's RAG (Retrieval-Augmented Generation) features. It enables Fess to use OpenAI's reasoning models (GPT-5 and the o1/o3/o4 series) for AI-powered search capabilities including intent detection, answer generation, document summarization, and FAQ handling.

It also drives any model served behind an OpenAI-compatible Chat Completions API - LiteLLM, vLLM, RamaLama, Ollama's OpenAI shim, Azure OpenAI - because every decision the client makes from the model name can be overridden explicitly. See [OpenAI-compatible endpoints](#openai-compatible-endpoints).

## Download

See [Maven Repository](https://repo1.maven.org/maven2/org/codelibs/fess/fess-llm-openai/).

## Requirements

- Fess 15.x or later
- Java 21 or later
- An OpenAI API key, or an OpenAI-compatible endpoint (see [OpenAI-compatible endpoints](#openai-compatible-endpoints); a placeholder key is still required there)

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
| `rag.llm.openai.api.key` | - | OpenAI API key (required; set any non-empty value for a keyless local endpoint, see [OpenAI-compatible endpoints](#openai-compatible-endpoints)) |
| `rag.llm.openai.api.url` | `https://api.openai.com/v1` | OpenAI API endpoint URL |
| `rag.llm.openai.model` | `gpt-5-mini` | Model name (e.g., `gpt-5-nano`, `gpt-5`, `o3-mini`). See [Model Support](#model-support). |
| `rag.llm.openai.timeout` | `120000` | HTTP request timeout in milliseconds |
| `rag.llm.openai.availability.check.interval` | `60` | Interval (seconds) for checking API availability |
| `rag.llm.openai.retry.max` | `10` | Maximum HTTP attempts (initial call plus retries) for a retryable status |
| `rag.llm.openai.retry.base.delay.ms` | `2000` | Base delay for exponential backoff between retries |
| `rag.llm.openai.max.concurrent.requests` | `5` | Maximum chat requests in flight to the API at once; further requests wait for a permit |
| `rag.llm.openai.concurrency.wait.timeout` | `30000` | Milliseconds a request waits for a concurrency permit before failing with `Too many concurrent requests` |
| `rag.llm.openai.stream.include.usage` | `true` | Request `stream_options.include_usage` on streaming calls. Set `false` for backends that reject the field. |
| `rag.llm.openai.reasoning.model.enabled` | `auto` | Whether the model is a reasoning model. `auto` infers it from the model name (`gpt-5*`, `o1*`, `o3*`, `o4*`); `true` / `false` force it. See [Model Support](#model-support). |
| `rag.llm.openai.temperature.enabled` | `auto` | Whether to send `temperature`. `auto` follows `reasoning.model.enabled` inverted; `true` / `false` force it. |
| `rag.llm.openai.sampling.params.enabled` | `auto` | Whether to send `top_p`, `frequency_penalty` and `presence_penalty`. `auto` follows `reasoning.model.enabled` inverted; `true` / `false` force it. |
| `rag.llm.openai.max.completion.tokens.enabled` | `auto` | Whether to send the token limit as `max_completion_tokens` instead of `max_tokens`. `auto` follows `reasoning.model.enabled`; `true` / `false` force it. |
| `rag.llm.openai.reasoning.effort.enabled` | `auto` | Whether to send `reasoning_effort`. `auto` follows `reasoning.model.enabled`; `true` / `false` force it. |
| `rag.llm.openai.reasoning.token.multiplier` | `4` | Multiplier applied to the per-prompt-type default `max_tokens` for reasoning models |
| `rag.llm.openai.<promptType>.context.max.chars` | `16000` (`answer`, `summary`), `10000` (others) | Maximum characters of retrieved context passed to the prompt |
| `rag.llm.openai.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |
| `rag.llm.openai.chat.evaluation.description.max.chars` | `500` | Maximum characters of each document description during evaluation |
| `rag.llm.openai.history.max.chars` | `8000` | Maximum characters of conversation history |
| `rag.llm.openai.history.assistant.max.chars` | `800` | Maximum characters kept from each assistant turn |
| `rag.llm.openai.history.assistant.summary.max.chars` | `800` | Maximum characters kept from each assistant summary |
| `rag.llm.openai.intent.history.max.messages` | `8` | Maximum history messages passed to the intent prompt |
| `rag.llm.openai.intent.history.max.chars` | `4000` | Maximum history characters passed to the intent prompt |

### Per-Prompt-Type Parameters

You can configure these OpenAI request parameters for each prompt type; `reasoning.effort`
applies to reasoning models by default, and `top.p`, `frequency.penalty` and `presence.penalty`
are suppressed for them by default - every capability is independently overridable, see
[Model Support](#model-support):

| Property | Description |
|----------|-------------|
| `rag.llm.openai.<promptType>.temperature` | `temperature` for this prompt type, replacing the built-in per-prompt-type default. Not sent at all when the model does not accept `temperature`, and the drop is reported - see [Dropped parameters](#dropped-parameters). |
| `rag.llm.openai.<promptType>.max.tokens` | Token limit for this prompt type, replacing the built-in per-prompt-type default. Setting it explicitly also disables the reasoning token multiplier (`rag.llm.openai.reasoning.token.multiplier`) for this prompt type - the value you write is the value sent, as `max_tokens` or `max_completion_tokens`. This is the key to raise when a reasoning model exhausts its budget on internal reasoning. |
| `rag.llm.openai.<promptType>.reasoning.effort` | Reasoning effort level (`low`, `medium`, `high`) |
| `rag.llm.openai.<promptType>.top.p` | Nucleus sampling `top_p` (`0.0`-`1.0`) |
| `rag.llm.openai.<promptType>.frequency.penalty` | `frequency_penalty` (`-2.0`-`2.0`) |
| `rag.llm.openai.<promptType>.presence.penalty` | `presence_penalty` (`-2.0`-`2.0`) |
| `rag.llm.openai.<promptType>.thinking.budget` | **Read but ignored by this plugin.** Fess core parses it as an integer for every LLM plugin, but OpenAI's Chat Completions API has no thinking-budget field and this client never sends one. Setting it changes nothing; setting it to a non-integer value fails the request while core parses it. Use `<promptType>.max.tokens` together with `<promptType>.reasoning.effort` to control reasoning spend instead. |

There is no prompt-type-less form of these keys: `rag.llm.openai.reasoning.effort` (without a
`<promptType>` segment) is read by nothing and is silently ignored. It is also not an abbreviation
of `rag.llm.openai.reasoning.effort.enabled`, which is the `auto`/`true`/`false` capability
override in the [Configuration](#configuration) table and takes entirely different values.

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
`embedQuery()` on this client issue an identical request and share all request/response handling.

They differ in what they send. `embedQuery()` strips Fess/Lucene query syntax - `+required`
terms, `(a OR b)` groups, `title:"x"^2` field boosts, quoted phrases - before embedding, because
on the RAG path the string it receives is a Fess query built by the LLM's intent step and those
operators are markup rather than words. `embedDocuments()` strips nothing: document text is
prose whose punctuation is content. A query left empty by the removals is embedded unchanged.

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

By default the plugin adapts the request to the model by matching the model name:

| | Reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`) | Any other model name | Override |
|---|---|---|---|
| Token limit parameter | `max_completion_tokens` | `max_tokens` | `rag.llm.openai.max.completion.tokens.enabled` |
| `temperature` | Not sent - the API rejects any non-default value | Sent | `rag.llm.openai.temperature.enabled` |
| `top_p`, `frequency_penalty`, `presence_penalty` | Not sent - the API rejects them outright | Sent | `rag.llm.openai.sampling.params.enabled` |
| `reasoning_effort` | Supported | Not sent | `rag.llm.openai.reasoning.effort.enabled` |
| Default `max_tokens` | Per-prompt-type default x `rag.llm.openai.reasoning.token.multiplier` (default `4`) | Per-prompt-type default | `rag.llm.openai.reasoning.model.enabled` |

Each override takes `auto` (the default), `true` or `false`. `auto` is the model-name inference
described above: `rag.llm.openai.reasoning.model.enabled` infers the family from the name, and the other
four follow it. `true` and `false` force the decision whatever the model is called.

Because the five are independent, they can also be combined into a state OpenAI itself never
produces. The one worth knowing about: `rag.llm.openai.max.completion.tokens.enabled=true` with
`rag.llm.openai.reasoning.model.enabled` left on `auto` (or set to `false`) sends the token limit
under the reasoning **field** name, `max_completion_tokens`, but does not apply the reasoning token
**budget** - the `rag.llm.openai.reasoning.token.multiplier` follows `reasoning.model.enabled`
alone. On a model that really does reason, that hands the whole unmultiplied per-prompt-type
default (256 tokens for `intent`) to internal reasoning and returns empty content. Set
`reasoning.model.enabled=true` whenever the model reasons, and use the other four keys only to
correct what its endpoint accepts.

#### Dropped parameters

Configuring `rag.llm.openai.<promptType>.temperature`, `.top.p`, `.frequency.penalty`,
`.presence.penalty` or `.reasoning.effort` on a model that does not accept it logs a warning and
omits the parameter, rather than sending it and failing the whole chat with
`400 Unsupported parameter`:

```
WARN [LLM:OPENAI] top_p is not supported by model gpt-5-mini and was not sent.
     Remove the rag.llm.openai.<promptType>.top.p setting for this model.
```

The warning names the OpenAI field, the resolved model, and the configuration key to edit - note
that the key spells the field with dots (`top.p`, `frequency.penalty`, `presence.penalty`), so grep
for that form rather than for the wire name. It is emitted **once per parameter and model**, not
once per request: one RAG search issues several LLM calls, so repeating it would cost several log
lines per user search for as long as the misconfiguration lasts. Changing the model reports the
drop afresh.

Only a value *you* configured is reported. The client's own per-prompt-type default `temperature`
is withdrawn quietly when the model does not accept temperature, because that is the client's
decision rather than a misconfiguration. Either way the request sent is identical.

### OpenAI-compatible endpoints

Behind LiteLLM, vLLM, RamaLama, Ollama's OpenAI shim or an Azure deployment, the model name
carries no OpenAI semantics, so the name-based inference above cannot classify it. Set the
overrides explicitly instead. A reasoning model such as Qwen3 served by vLLM is reasoning, yet it
accepts `temperature` and `top_p`, expects `max_tokens`, and has no `reasoning_effort` field:

```properties
rag.llm.openai.api.url=http://localhost:8000/v1
rag.llm.openai.api.key=dummy
rag.llm.openai.model=qwen3-32b
rag.llm.openai.reasoning.model.enabled=true
rag.llm.openai.temperature.enabled=true
rag.llm.openai.sampling.params.enabled=true
rag.llm.openai.max.completion.tokens.enabled=false
rag.llm.openai.reasoning.effort.enabled=false
```

Setting `rag.llm.openai.reasoning.model.enabled` alone is enough when the endpoint follows OpenAI's own
parameter rules - the other four keys default to following it.

Notes for these deployments:

- The properties above live in `fess_config.properties`. They are not read at startup: each key is
  resolved the first time something asks for it, and a resolved value is then cached for the
  lifetime of the running instance (a key that is absent is not cached, and is re-resolved on every
  call). Either way, restart Fess after editing.
  `-Dfess.config.rag.llm.openai.<key>=<value>` works as a JVM-level override.
- `rag.llm.openai.api.key` must be non-empty even when the endpoint needs no credential: a blank
  key makes the availability check report the client unavailable. Any placeholder works.
- `rag.llm.openai.api.url` must not end with a trailing slash.
- Set `rag.llm.openai.stream.include.usage=false` if the backend rejects `stream_options`.
- The availability check calls `GET <api.url>/models`; the endpoint must implement it.
- Servers that end a completion with a `finish_reason` other than `stop` (for example `eos` or
  `end_turn`) produce a `Chat finished abnormally` warning on every call. It is harmless.

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
- **Reasoning Model Support** - Adaptive parameter handling for gpt-5 and o1/o3/o4 reasoning models, with per-capability overrides for models served through OpenAI-compatible APIs

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
