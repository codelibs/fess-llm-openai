OpenAI LLM Plugin for Fess
==========================

## Overview

This plugin provides OpenAI integration for Fess's RAG (Retrieval-Augmented Generation) features. It enables Fess to use OpenAI models (GPT-4, GPT-4o, o1, o3, o4, GPT-5, etc.) for AI-powered search capabilities including intent detection, answer generation, document summarization, and FAQ handling.

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

Configure the following properties in `fess_config.properties`:

| Property | Default | Description |
|----------|---------|-------------|
| `rag.llm.name` | - | Set to `openai` to use this plugin |
| `rag.chat.enabled` | `false` | Enable RAG chat feature |
| `rag.llm.openai.api.key` | - | OpenAI API key (required) |
| `rag.llm.openai.api.url` | `https://api.openai.com/v1` | OpenAI API endpoint URL |
| `rag.llm.openai.model` | `gpt-5-mini` | Model name (e.g., `gpt-4o`, `o3-mini`, `gpt-5`) |
| `rag.llm.openai.timeout` | `60000` | HTTP request timeout in milliseconds |
| `rag.llm.openai.availability.check.interval` | `60` | Interval (seconds) for checking API availability |
| `rag.llm.openai.chat.context.max.chars` | `4000` | Maximum characters for context in chat |
| `rag.llm.openai.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |

### Per-Prompt-Type Parameters

You can configure reasoning effort for each prompt type (applies to reasoning models: o1, o3, o4, gpt-5):

| Property | Description |
|----------|-------------|
| `rag.llm.openai.<promptType>.reasoning.effort` | Reasoning effort level (`low`, `medium`, `high`) |

### Model Support

The plugin automatically adapts its behavior based on the model:

| Model Family | `max_tokens` Parameter | Temperature | Reasoning Effort |
|---|---|---|---|
| gpt-3.5, gpt-4, gpt-4o | `max_tokens` | Supported | N/A |
| o1, o3, o4, gpt-5 | `max_completion_tokens` | Not supported | Supported |

## Features

- **Intent Detection** - Determines user intent (search, summary, FAQ, unclear) and generates Lucene queries
- **Answer Generation** - Generates answers based on search results with citation support
- **Document Summarization** - Summarizes specific documents
- **FAQ Handling** - Provides direct, concise answers to FAQ-type questions
- **Relevance Evaluation** - Identifies the most relevant documents for answer generation
- **Streaming Support** - Real-time response streaming via Server-Sent Events (SSE)
- **Availability Checking** - Validates API availability at configurable intervals
- **Reasoning Model Support** - Adaptive parameter handling for o1/o3/o4/gpt-5 reasoning models

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
