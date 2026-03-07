/*
 * Copyright 2012-2025 CodeLibs Project and the Others.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 */
package org.codelibs.fess.llm.openai;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.codelibs.core.lang.StringUtil;
import org.codelibs.fess.llm.AbstractLlmClient;
import org.codelibs.fess.llm.LlmChatRequest;
import org.codelibs.fess.llm.LlmChatResponse;
import org.codelibs.fess.llm.LlmException;
import org.codelibs.fess.llm.LlmMessage;
import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.util.ComponentUtil;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * LLM client implementation for OpenAI API.
 *
 * OpenAI provides cloud-based LLM services including GPT-4 and other models.
 * This client supports both synchronous and streaming chat completions.
 *
 * @author FessProject
 * @see <a href="https://platform.openai.com/docs/api-reference">OpenAI API Reference</a>
 */
public class OpenAiLlmClient extends AbstractLlmClient {

    private static final Logger logger = LogManager.getLogger(OpenAiLlmClient.class);
    /** The name identifier for the OpenAI LLM client. */
    protected static final String NAME = "openai";
    private static final String SSE_DATA_PREFIX = "data: ";
    private static final String SSE_DONE_MARKER = "[DONE]";

    /** The system prompt for LLM interactions. */
    protected String systemPrompt;
    /** The prompt for detecting user intent. */
    protected String intentDetectionPrompt;
    /** The system prompt for handling unclear intents. */
    protected String unclearIntentSystemPrompt;
    /** The system prompt for handling no results. */
    protected String noResultsSystemPrompt;
    /** The system prompt for handling document not found. */
    protected String documentNotFoundSystemPrompt;
    /** The prompt for evaluating responses. */
    protected String evaluationPrompt;
    /** The system prompt for answer generation. */
    protected String answerGenerationSystemPrompt;
    /** The system prompt for summary generation. */
    protected String summarySystemPrompt;
    /** The system prompt for FAQ answer generation. */
    protected String faqAnswerSystemPrompt;
    /** The system prompt for direct answer generation. */
    protected String directAnswerSystemPrompt;

    /**
     * Default constructor.
     */
    public OpenAiLlmClient() {
        // Default constructor
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected boolean checkAvailabilityNow() {
        final String apiKey = getApiKey();
        if (StringUtil.isBlank(apiKey)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] OpenAI is not available. apiKey is blank");
            }
            return false;
        }
        final String apiUrl = getApiUrl();
        if (StringUtil.isBlank(apiUrl)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] OpenAI is not available. apiUrl is blank");
            }
            return false;
        }
        try {
            final HttpGet request = new HttpGet(apiUrl + "/models");
            request.addHeader("Authorization", "Bearer " + apiKey);
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                final boolean available = statusCode >= 200 && statusCode < 300;
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OPENAI] OpenAI availability check. url={}, statusCode={}, available={}", apiUrl, statusCode,
                            available);
                }
                return available;
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] OpenAI is not available. url={}, error={}", apiUrl, e.getMessage());
            }
            return false;
        }
    }

    @Override
    public LlmChatResponse chat(final LlmChatRequest request) {
        final String url = getApiUrl() + "/chat/completions";
        final Map<String, Object> requestBody = buildRequestBody(request, false);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OPENAI] Sending chat request to OpenAI. url={}, model={}, messageCount={}", url, requestBody.get("model"),
                    request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] requestBody={}", json);
            }
            final HttpPost httpRequest = new HttpPost(url);
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));
            httpRequest.addHeader("Authorization", "Bearer " + getApiKey());

            try (var response = getHttpClient().execute(httpRequest)) {
                final int statusCode = response.getCode();
                if (statusCode < 200 || statusCode >= 300) {
                    String errorBody = "";
                    if (response.getEntity() != null) {
                        try {
                            errorBody = EntityUtils.toString(response.getEntity());
                        } catch (final IOException e) {
                            // ignore
                        }
                    }
                    logger.warn("OpenAI API error. url={}, statusCode={}, message={}, body={}", url, statusCode, response.getReasonPhrase(),
                            errorBody);
                    throw new LlmException("OpenAI API error: " + statusCode + " " + response.getReasonPhrase(),
                            resolveErrorCode(statusCode));
                }

                final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OPENAI] responseBody={}", responseBody);
                }
                final JsonNode jsonNode = objectMapper.readTree(responseBody);

                final LlmChatResponse chatResponse = new LlmChatResponse();
                if (jsonNode.has("choices") && jsonNode.get("choices").isArray() && jsonNode.get("choices").size() > 0) {
                    final JsonNode firstChoice = jsonNode.get("choices").get(0);
                    if (firstChoice.has("message") && firstChoice.get("message").has("content")) {
                        chatResponse.setContent(firstChoice.get("message").get("content").asText());
                    }
                    if (firstChoice.has("finish_reason") && !firstChoice.get("finish_reason").isNull()) {
                        chatResponse.setFinishReason(firstChoice.get("finish_reason").asText());
                    }
                }
                if (jsonNode.has("model")) {
                    chatResponse.setModel(jsonNode.get("model").asText());
                }
                if (jsonNode.has("usage")) {
                    final JsonNode usage = jsonNode.get("usage");
                    if (usage.has("prompt_tokens")) {
                        chatResponse.setPromptTokens(usage.get("prompt_tokens").asInt());
                    }
                    if (usage.has("completion_tokens")) {
                        chatResponse.setCompletionTokens(usage.get("completion_tokens").asInt());
                    }
                    if (usage.has("total_tokens")) {
                        chatResponse.setTotalTokens(usage.get("total_tokens").asInt());
                    }
                }

                if (logger.isDebugEnabled()) {
                    logger.debug(
                            "Received chat response from OpenAI. model={}, promptTokens={}, completionTokens={}, totalTokens={}, contentLength={}, elapsedTime={}ms",
                            chatResponse.getModel(), chatResponse.getPromptTokens(), chatResponse.getCompletionTokens(),
                            chatResponse.getTotalTokens(), chatResponse.getContent() != null ? chatResponse.getContent().length() : 0,
                            System.currentTimeMillis() - startTime);
                }

                return chatResponse;
            }
        } catch (final LlmException e) {
            throw e;
        } catch (final Exception e) {
            logger.warn("Failed to call OpenAI API. url={}, error={}", url, e.getMessage(), e);
            throw new LlmException("Failed to call OpenAI API", LlmException.ERROR_CONNECTION, e);
        }
    }

    @Override
    public void streamChat(final LlmChatRequest request, final LlmStreamCallback callback) {
        final String url = getApiUrl() + "/chat/completions";
        final Map<String, Object> requestBody = buildRequestBody(request, true);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OPENAI] Starting streaming chat request to OpenAI. url={}, model={}, messageCount={}", url,
                    requestBody.get("model"), request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] requestBody={}", json);
            }
            final HttpPost httpRequest = new HttpPost(url);
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));
            httpRequest.addHeader("Authorization", "Bearer " + getApiKey());

            try (var response = getHttpClient().execute(httpRequest)) {
                final int statusCode = response.getCode();
                if (statusCode < 200 || statusCode >= 300) {
                    String errorBody = "";
                    if (response.getEntity() != null) {
                        try {
                            errorBody = EntityUtils.toString(response.getEntity());
                        } catch (final IOException | ParseException e) {
                            // ignore
                        }
                    }
                    logger.warn("OpenAI streaming API error. url={}, statusCode={}, message={}, body={}", url, statusCode,
                            response.getReasonPhrase(), errorBody);
                    throw new LlmException("OpenAI API error: " + statusCode + " " + response.getReasonPhrase(),
                            resolveErrorCode(statusCode));
                }

                if (response.getEntity() == null) {
                    logger.warn("Empty response from OpenAI streaming API. url={}", url);
                    throw new LlmException("Empty response from OpenAI");
                }

                int chunkCount = 0;
                try (BufferedReader reader =
                        new BufferedReader(new InputStreamReader(response.getEntity().getContent(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (StringUtil.isBlank(line)) {
                            continue;
                        }

                        if (!line.startsWith(SSE_DATA_PREFIX)) {
                            continue;
                        }

                        final String data = line.substring(SSE_DATA_PREFIX.length()).trim();
                        if (SSE_DONE_MARKER.equals(data)) {
                            callback.onChunk("", true);
                            break;
                        }

                        try {
                            final JsonNode jsonNode = objectMapper.readTree(data);
                            if (jsonNode.has("choices") && jsonNode.get("choices").isArray() && jsonNode.get("choices").size() > 0) {
                                final JsonNode firstChoice = jsonNode.get("choices").get(0);
                                final boolean done = firstChoice.has("finish_reason") && !firstChoice.get("finish_reason").isNull();

                                if (firstChoice.has("delta") && firstChoice.get("delta").has("content")) {
                                    final String content = firstChoice.get("delta").get("content").asText();
                                    callback.onChunk(content, done);
                                    chunkCount++;
                                } else if (done) {
                                    callback.onChunk("", true);
                                }

                                if (done) {
                                    break;
                                }
                            }
                        } catch (final JsonProcessingException e) {
                            logger.warn("Failed to parse OpenAI streaming response. line={}", line, e);
                        }
                    }
                }

                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OPENAI] Completed streaming chat from OpenAI. url={}, chunkCount={}, elapsedTime={}ms", url,
                            chunkCount, System.currentTimeMillis() - startTime);
                }
            }
        } catch (final LlmException e) {
            callback.onError(e);
            throw e;
        } catch (final IOException e) {
            logger.warn("Failed to stream from OpenAI API. url={}, error={}", url, e.getMessage(), e);
            final LlmException llmException = new LlmException("Failed to stream from OpenAI API", LlmException.ERROR_CONNECTION, e);
            callback.onError(llmException);
            throw llmException;
        }
    }

    /**
     * Builds the request body for the OpenAI API.
     *
     * @param request the chat request
     * @param stream whether to enable streaming
     * @return the request body as a map
     */
    protected Map<String, Object> buildRequestBody(final LlmChatRequest request, final boolean stream) {
        final Map<String, Object> body = new HashMap<>();

        String model = request.getModel();
        if (StringUtil.isBlank(model)) {
            model = getModel();
        }
        body.put("model", model);

        final List<Map<String, String>> messages = request.getMessages().stream().map(this::convertMessage).collect(Collectors.toList());
        body.put("messages", messages);

        body.put("stream", stream);

        if (supportsTemperature(model) && request.getTemperature() != null) {
            body.put("temperature", request.getTemperature());
        }

        final String maxTokensKey = useMaxCompletionTokens(model) ? "max_completion_tokens" : "max_tokens";
        if (request.getMaxTokens() != null) {
            body.put(maxTokensKey, request.getMaxTokens());
        }

        if (isReasoningModel(model)) {
            final String reasoningEffort = request.getExtraParam("reasoning_effort");
            if (reasoningEffort != null) {
                body.put("reasoning_effort", reasoningEffort);
            }
        }

        final String topP = request.getExtraParam("top_p");
        if (topP != null) {
            try {
                body.put("top_p", Double.parseDouble(topP));
            } catch (final NumberFormatException e) {
                logger.warn("[LLM:OPENAI] Invalid top_p value: {}", topP);
            }
        }

        final String frequencyPenalty = request.getExtraParam("frequency_penalty");
        if (frequencyPenalty != null) {
            try {
                body.put("frequency_penalty", Double.parseDouble(frequencyPenalty));
            } catch (final NumberFormatException e) {
                logger.warn("[LLM:OPENAI] Invalid frequency_penalty value: {}", frequencyPenalty);
            }
        }

        final String presencePenalty = request.getExtraParam("presence_penalty");
        if (presencePenalty != null) {
            try {
                body.put("presence_penalty", Double.parseDouble(presencePenalty));
            } catch (final NumberFormatException e) {
                logger.warn("[LLM:OPENAI] Invalid presence_penalty value: {}", presencePenalty);
            }
        }

        return body;
    }

    /**
     * Determines whether the given model requires the "max_completion_tokens" parameter
     * instead of the legacy "max_tokens" parameter.
     *
     * @param model the model name
     * @return true if the model uses max_completion_tokens
     */
    protected boolean useMaxCompletionTokens(final String model) {
        if (StringUtil.isBlank(model)) {
            return false;
        }
        if (model.startsWith("o1") || model.startsWith("o3") || model.startsWith("o4")) {
            return true;
        }
        if (model.startsWith("gpt-5")) {
            return true;
        }
        return false;
    }

    /**
     * Determines whether the given model supports the "temperature" parameter.
     * Reasoning models (o1, o3, o4, gpt-5 series) do not support custom temperature values.
     * Only the default value (1) is accepted by these models.
     *
     * @param model the model name
     * @return true if the model supports custom temperature values
     */
    protected boolean supportsTemperature(final String model) {
        if (StringUtil.isBlank(model)) {
            return true;
        }
        if (model.startsWith("o1") || model.startsWith("o3") || model.startsWith("o4")) {
            return false;
        }
        if (model.startsWith("gpt-5")) {
            return false;
        }
        return true;
    }

    /**
     * Determines whether the given model is a reasoning model that uses internal
     * reasoning tokens (e.g., o1, o3, o4, gpt-5 series).
     *
     * @param model the model name
     * @return true if the model is a reasoning model
     */
    protected boolean isReasoningModel(final String model) {
        if (StringUtil.isBlank(model)) {
            return false;
        }
        if (model.startsWith("o1") || model.startsWith("o3") || model.startsWith("o4")) {
            return true;
        }
        if (model.startsWith("gpt-5")) {
            return true;
        }
        return false;
    }

    /**
     * Converts an LlmMessage to a map for the API request.
     *
     * @param message the message to convert
     * @return the message as a map
     */
    protected Map<String, String> convertMessage(final LlmMessage message) {
        final Map<String, String> map = new HashMap<>();
        map.put("role", message.getRole());
        map.put("content", message.getContent());
        return map;
    }

    /** Sets the system prompt for LLM interactions.
     * @param systemPrompt the system prompt */
    public void setSystemPrompt(final String systemPrompt) {
        this.systemPrompt = systemPrompt;
    }

    /** Sets the prompt for detecting user intent.
     * @param intentDetectionPrompt the intent detection prompt */
    public void setIntentDetectionPrompt(final String intentDetectionPrompt) {
        this.intentDetectionPrompt = intentDetectionPrompt;
    }

    /** Sets the system prompt for handling unclear intents.
     * @param unclearIntentSystemPrompt the unclear intent system prompt */
    public void setUnclearIntentSystemPrompt(final String unclearIntentSystemPrompt) {
        this.unclearIntentSystemPrompt = unclearIntentSystemPrompt;
    }

    /** Sets the system prompt for handling no results.
     * @param noResultsSystemPrompt the no results system prompt */
    public void setNoResultsSystemPrompt(final String noResultsSystemPrompt) {
        this.noResultsSystemPrompt = noResultsSystemPrompt;
    }

    /** Sets the system prompt for handling document not found.
     * @param documentNotFoundSystemPrompt the document not found system prompt */
    public void setDocumentNotFoundSystemPrompt(final String documentNotFoundSystemPrompt) {
        this.documentNotFoundSystemPrompt = documentNotFoundSystemPrompt;
    }

    /** Sets the prompt for evaluating responses.
     * @param evaluationPrompt the evaluation prompt */
    public void setEvaluationPrompt(final String evaluationPrompt) {
        this.evaluationPrompt = evaluationPrompt;
    }

    /** Sets the system prompt for answer generation.
     * @param answerGenerationSystemPrompt the answer generation system prompt */
    public void setAnswerGenerationSystemPrompt(final String answerGenerationSystemPrompt) {
        this.answerGenerationSystemPrompt = answerGenerationSystemPrompt;
    }

    /** Sets the system prompt for summary generation.
     * @param summarySystemPrompt the summary system prompt */
    public void setSummarySystemPrompt(final String summarySystemPrompt) {
        this.summarySystemPrompt = summarySystemPrompt;
    }

    /** Sets the system prompt for FAQ answer generation.
     * @param faqAnswerSystemPrompt the FAQ answer system prompt */
    public void setFaqAnswerSystemPrompt(final String faqAnswerSystemPrompt) {
        this.faqAnswerSystemPrompt = faqAnswerSystemPrompt;
    }

    /** Sets the system prompt for direct answer generation.
     * @param directAnswerSystemPrompt the direct answer system prompt */
    public void setDirectAnswerSystemPrompt(final String directAnswerSystemPrompt) {
        this.directAnswerSystemPrompt = directAnswerSystemPrompt;
    }

    /**
     * Gets the OpenAI API key.
     *
     * @return the API key
     */
    protected String getApiKey() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.api.key", "");
    }

    /**
     * Gets the OpenAI API URL.
     *
     * @return the API URL
     */
    protected String getApiUrl() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.api.url", "https://api.openai.com/v1");
    }

    @Override
    protected String getModel() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.model", "gpt-5-mini");
    }

    @Override
    protected int getTimeout() {
        return Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.timeout", "120000"));
    }

    @Override
    protected String getConfigPrefix() {
        return "rag.llm.openai";
    }

    @Override
    protected void applyPromptTypeParams(final LlmChatRequest request, final String promptType) {
        super.applyPromptTypeParams(request, promptType);
        final String configPrefix = getConfigPrefix();
        final String reasoningEffort =
                ComponentUtil.getFessConfig().getOrDefault(configPrefix + "." + promptType + ".reasoning.effort", null);
        if (reasoningEffort != null) {
            request.putExtraParam("reasoning_effort", reasoningEffort);
        }
        final String topP = ComponentUtil.getFessConfig().getOrDefault(configPrefix + "." + promptType + ".top.p", null);
        if (topP != null) {
            request.putExtraParam("top_p", topP);
        }
        final String frequencyPenalty =
                ComponentUtil.getFessConfig().getOrDefault(configPrefix + "." + promptType + ".frequency.penalty", null);
        if (frequencyPenalty != null) {
            request.putExtraParam("frequency_penalty", frequencyPenalty);
        }
        final String presencePenalty =
                ComponentUtil.getFessConfig().getOrDefault(configPrefix + "." + promptType + ".presence.penalty", null);
        if (presencePenalty != null) {
            request.putExtraParam("presence_penalty", presencePenalty);
        }
        applyDefaultParams(request, promptType);
    }

    /**
     * Applies default generation parameters based on prompt type.
     * Only sets defaults when user has not configured the parameter.
     *
     * @param request the LLM chat request
     * @param promptType the prompt type (e.g. "intent", "evaluation", "answer")
     */
    protected void applyDefaultParams(final LlmChatRequest request, final String promptType) {
        switch (promptType) {
        case "intent":
        case "evaluation":
            if (request.getTemperature() == null) {
                request.setTemperature(0.1);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
            }
            break;
        case "unclear":
        case "noresults":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(512);
            }
            break;
        case "docnotfound":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
            }
            break;
        case "direct":
        case "faq":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(1024);
            }
            break;
        case "answer":
            if (request.getTemperature() == null) {
                request.setTemperature(0.5);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(2048);
            }
            break;
        case "summary":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(2048);
            }
            break;
        default:
            break;
        }
    }

    @Override
    protected int getAvailabilityCheckInterval() {
        return Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.availability.check.interval", "60"));
    }

    @Override
    protected boolean isRagChatEnabled() {
        return Boolean.parseBoolean(ComponentUtil.getFessConfig().getOrDefault("rag.chat.enabled", "false"));
    }

    @Override
    protected String getLlmType() {
        return ComponentUtil.getFessConfig().getSystemProperty("rag.llm.name", "ollama");
    }

    @Override
    protected int getContextMaxChars(final String promptType) {
        final String key = "rag.llm.openai." + promptType + ".context.max.chars";
        final String configValue = ComponentUtil.getFessConfig().getOrDefault(key, null);
        if (configValue != null) {
            final int value = Integer.parseInt(configValue);
            if (value > 0) {
                return value;
            }
            logger.warn("Invalid context max chars for promptType={}: {}. Using default.", promptType, value);
        }
        switch (promptType) {
        case "answer":
            return 16000;
        case "summary":
            return 16000;
        case "faq":
            return 10000;
        default:
            return 10000;
        }
    }

    @Override
    protected int getEvaluationMaxRelevantDocs() {
        final int value =
                Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.chat.evaluation.max.relevant.docs", "3"));
        if (value <= 0) {
            logger.warn("Invalid evaluation max relevant docs: {}. Using default: 3", value);
            return 3;
        }
        return value;
    }

    @Override
    protected int getEvaluationDescriptionMaxChars() {
        final int value =
                Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.chat.evaluation.description.max.chars", "500"));
        if (value <= 0) {
            logger.warn("Invalid evaluation description max chars: {}. Using default: 500", value);
            return 500;
        }
        return value;
    }

    @Override
    protected String getSystemPrompt() {
        if (systemPrompt == null) {
            throw new LlmException("systemPrompt is not configured for " + getName());
        }
        return systemPrompt;
    }

    @Override
    protected String getIntentDetectionPrompt() {
        if (intentDetectionPrompt == null) {
            throw new LlmException("intentDetectionPrompt is not configured for " + getName());
        }
        return intentDetectionPrompt;
    }

    @Override
    protected String getUnclearIntentSystemPrompt() {
        if (unclearIntentSystemPrompt == null) {
            throw new LlmException("unclearIntentSystemPrompt is not configured for " + getName());
        }
        return unclearIntentSystemPrompt;
    }

    @Override
    protected String getNoResultsSystemPrompt() {
        if (noResultsSystemPrompt == null) {
            throw new LlmException("noResultsSystemPrompt is not configured for " + getName());
        }
        return noResultsSystemPrompt;
    }

    @Override
    protected String getDocumentNotFoundSystemPrompt() {
        if (documentNotFoundSystemPrompt == null) {
            throw new LlmException("documentNotFoundSystemPrompt is not configured for " + getName());
        }
        return documentNotFoundSystemPrompt;
    }

    @Override
    protected String getEvaluationPrompt() {
        if (evaluationPrompt == null) {
            throw new LlmException("evaluationPrompt is not configured for " + getName());
        }
        return evaluationPrompt;
    }

    @Override
    protected String getAnswerGenerationSystemPrompt() {
        if (answerGenerationSystemPrompt == null) {
            throw new LlmException("answerGenerationSystemPrompt is not configured for " + getName());
        }
        return answerGenerationSystemPrompt;
    }

    @Override
    protected String getSummarySystemPrompt() {
        if (summarySystemPrompt == null) {
            throw new LlmException("summarySystemPrompt is not configured for " + getName());
        }
        return summarySystemPrompt;
    }

    @Override
    protected String getFaqAnswerSystemPrompt() {
        if (faqAnswerSystemPrompt == null) {
            throw new LlmException("faqAnswerSystemPrompt is not configured for " + getName());
        }
        return faqAnswerSystemPrompt;
    }

    @Override
    protected String getDirectAnswerSystemPrompt() {
        if (directAnswerSystemPrompt == null) {
            throw new LlmException("directAnswerSystemPrompt is not configured for " + getName());
        }
        return directAnswerSystemPrompt;
    }
}
