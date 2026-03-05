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

## Testing

Tests use `UnitFessTestCase` (extends LastaFlute's `WebContainerTestCase`) with OkHttp `MockWebServer` for HTTP mocking. The test creates a `TestableOpenAiLlmClient` inner subclass that overrides config methods to avoid needing a running Fess instance.

## Coding Conventions

- Java 21, Maven build with `formatter-maven-plugin` and `license-maven-plugin` from fess-parent
- Use `final` on local variables and method parameters
- Log with `logger.debug`/`logger.warn` using `[LLM:OPENAI]` prefix for debug messages
- Guard debug logging with `if (logger.isDebugEnabled())`
- Configuration properties accessed via `ComponentUtil.getFessConfig().getOrDefault(key, default)`
- Prompts are injected via setter methods (configured externally, not hardcoded)
