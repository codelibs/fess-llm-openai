# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI LLM plugin for [Fess](https://github.com/codelibs/fess) enterprise search server. Implements the `AbstractLlmClient` interface to integrate OpenAI models into Fess's RAG (Retrieval-Augmented Generation) pipeline. Single-class plugin with `OpenAiLlmClient` as the sole production class.

## Build Commands

```bash
# Build (requires fess-parent installed locally)
mvn clean package

# Run tests
mvn test

# Run a single test method
mvn test -Dtest=OpenAiLlmClientTest#testMethodName
```

**Important**: This project depends on `fess-parent` (15.6.0-SNAPSHOT) which must be installed locally first. CI checks out and installs it from `codelibs/fess-parent` main branch. For local development:
```bash
cd /path/to/fess-parent && mvn install -Dgpg.skip=true
```

## Architecture

- **`OpenAiLlmClient`** (`src/main/java/.../openai/OpenAiLlmClient.java`) - Extends `AbstractLlmClient` from fess core. Handles:
  - Synchronous and streaming (SSE) chat completions via `/v1/chat/completions`
  - Availability checking via `/v1/models`
  - Model-aware parameter handling: reasoning models (o1/o3/o4/gpt-5) use `max_completion_tokens` instead of `max_tokens`, don't support `temperature`, and accept `reasoning_effort`
  - Per-prompt-type default parameters (temperature, max_tokens) via `applyDefaultParams()`
  - Configuration read from `fess_config.properties` via `ComponentUtil.getFessConfig()` with prefix `rag.llm.openai`

### Logging keys

`streamChat` emits a single `[LLM:OPENAI] Stream completed.` INFO line per call carrying:
`chunkCount`, `objectCount`, `firstChunkMs`, `elapsedTime`, `id`, `systemFingerprint`,
`finishReason`, `promptTokens`, `cachedTokens`, `completionTokens`, `reasoningTokens`,
`totalTokens`.

`chat()` emits a single `[LLM:OPENAI] Chat response received.` INFO line carrying the
same fields plus `contentLength`.

When `finishReason` is anything other than `stop`, both `chat()` and `streamChat()` emit
an extra WARN line (`Chat finished abnormally` / `Stream finished abnormally`) so
truncation (`length`), moderation (`content_filter`), unexpected tool-calling
(`tool_calls`, `function_call`), and any future unknown values can be alerted on without
DEBUG. The WARN line carries `id`, `finishReason`, `completionTokens`, `reasoningTokens`,
`model`, plus `contentLength` (chat only). Use these fields to mine logs for
`max_tokens` tuning, content-filter audits, and misconfigured `extra_params`.

Streaming additionally emits `[LLM:OPENAI] Stream refusal.` WARN whenever
`delta.refusal` is set (structured-output refusals), carrying `id`, `refusal`, `model`.

Enable `org.codelibs.fess.llm.openai` at DEBUG to additionally log:
- the JSON request body (`requestBody=`),
- HTTP status + `Content-Type` of the streaming response,
- each parsed JSON object from the stream (`streamObject#N json=`),
- the response body for non-streaming calls (`responseBody=`).

All URL log fields run through a credential-mask helper that strips
`api_key`, `apikey`, `api-key`, `key`, `token`, `access_token`, `access-token`
query parameters (case-insensitive) so URLs stored in `rag.llm.openai.api.url` for
credentialed proxies do not leak keys to logs. The helper also masks a URL's
`user:password@` userinfo, but only defensively: HttpClient rejects a userinfo-bearing
request URI outright, so such a URL can never issue a request.

A malformed `api.url` is rejected while the request is built, and `URI.create` quotes the
whole URI in the `IllegalArgumentException` it raises. Requests are therefore built through
`HttpRequestFactory`, which replaces that exception with one naming only the configuration
key plus the parser's reason and index, with no URL and no cause. Do not "improve" it by
echoing a masked URL: a URL is unparseable precisely because it holds a character the
masking patterns exclude, so the pattern that should have covered it is the one that stops
matching.

### Auth & retries

The OpenAI API key is sent as the `Authorization: Bearer <key>` header per the OpenAI
spec — keys never appear in the canonical OpenAI URL. For credentialed proxies, see the
URL-masking note above.

Retries: HTTP `429`, `500`, `502`, `503`, `504` are retried up to
`rag.llm.openai.retry.max` times (default `10`) with exponential backoff starting at
`rag.llm.openai.retry.base.delay.ms` (default `2000`) and ±20% jitter. When the server
returns `Retry-After` (integer seconds, max `600`), it overrides the computed backoff
for that attempt — honoring this is OpenAI's official guidance. HTTP-date format on
`Retry-After` is unsupported and falls back to backoff.

`IOException` (connect timeout, socket reset, DNS failure) is **not** retried —
mirrors the Gemini client; if the request reached the server, retrying may double-bill.

Streaming retries only the initial connect — once the response body starts flowing,
partial-stream errors propagate immediately as `LlmException` so consumers never see
the same chunk twice.

**Worst-case retry budget:** at default settings (10 attempts, 2 s base) the sum of
backoff sleeps is `2 + 4 + 8 + … + 512 ≈ 1022 s ≈ 17 min` across the 9 retries before
the 10th attempt. With every attempt receiving a `Retry-After` (capped at 600 s), the
worst case approaches `9 × 600 s = 90 min`. Tune `rag.llm.openai.retry.max` down for
tighter latency bounds.

### Streaming usage

By default the client requests `stream_options.include_usage=true` so the final SSE
chunk carries the full `usage` object (including `completion_tokens_details.reasoning_tokens`
for reasoning models and `prompt_tokens_details.cached_tokens` for prompt-cache hits).
The read-loop deliberately continues past the `finish_reason` chunk so this terminal
usage chunk can be captured before `[DONE]`.

Set `rag.llm.openai.stream.include.usage=false` for OpenAI-compatible backends (some
older proxies, custom vLLM deployments) that reject this field.

### Reasoning models

Reasoning models (`o1*`, `o3*`, `o4*`, `gpt-5*`) are the supported set; `gpt-5-nano` is
what the plugin is exercised against. `isReasoningModel()` is the single predicate for the
family - `useMaxCompletionTokens()` and `supportsTemperature()` delegate to it, so the three
cannot drift apart. It gates `max_completion_tokens` (in place of `max_tokens`),
`reasoning_effort`, and the suppression of `temperature`, `top_p`, `frequency_penalty` and
`presence_penalty`, all four of which these models reject with
`400 Unsupported parameter`. A suppressed parameter WARNs rather than being dropped
silently, since it came from an explicit `rag.llm.openai.<promptType>.*` setting.

For these models the per-prompt-type default `max_tokens` is multiplied by
`rag.llm.openai.reasoning.token.multiplier` (default `4`) so internal reasoning-token spend
does not crowd out visible output. Raising `reasoning.effort` on the short `intent` /
`evaluation` prompts can still exhaust the budget on reasoning alone and return empty
content - the `Chat finished abnormally ... contentLength=0` WARN is the signal.

## Testing

Tests use `UnitFessTestCase` (extends LastaFlute's `WebContainerTestCase`) with OkHttp `MockWebServer` for HTTP mocking. The test creates a `TestableOpenAiLlmClient` inner subclass that overrides config methods to avoid needing a running Fess instance.

## Coding Conventions

- Java 21, Maven build with `formatter-maven-plugin` and `license-maven-plugin` from fess-parent
- Use `final` on local variables and method parameters
- Log with `logger.debug`/`logger.warn` using `[LLM:OPENAI]` prefix for debug messages
- Guard debug logging with `if (logger.isDebugEnabled())`
- Configuration properties accessed via `ComponentUtil.getFessConfig().getOrDefault(key, default)`
- Prompts are injected via setter methods (configured externally, not hardcoded)
