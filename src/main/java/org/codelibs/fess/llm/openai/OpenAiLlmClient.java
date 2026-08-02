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
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
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
import org.codelibs.fess.openai.util.HttpRequestFactory;
import org.codelibs.fess.openai.util.OpenAiErrorBody;
import org.codelibs.fess.openai.util.OpenAiRetry;
import org.codelibs.fess.util.ComponentUtil;
import org.codelibs.fess.util.CredentialUrlUtil;

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

    /**
     * Default constructor.
     */
    public OpenAiLlmClient() {
        // Default constructor
    }

    /**
     * Returns whether the given {@code finish_reason} indicates an abnormal completion.
     *
     * <p>For OpenAI, only {@code "stop"} (and absent / blank values) is nominal. All other
     * values WARN, including:
     * <ul>
     *   <li>{@code "length"} - output was truncated by {@code max_tokens}.</li>
     *   <li>{@code "content_filter"} - moderation removed content; raise to operators.</li>
     *   <li>{@code "tool_calls"} / {@code "function_call"} - Fess RAG never enables tool /
     *       function calling, so seeing these means a misconfiguration in
     *       {@code extra_params}; surfacing them lets operators correct it.</li>
     *   <li>Any future unknown value - surface so operators can update this client.</li>
     * </ul>
     */
    static boolean isAbnormalFinishReason(final String reason) {
        if (reason == null) {
            return false;
        }
        final String trimmed = reason.trim();
        if (trimmed.isEmpty() || "stop".equals(trimmed)) {
            return false;
        }
        return true;
    }

    /**
     * Masks credentials in a URL before it is logged, covering both credential-bearing query
     * parameters and the authority's userinfo component. See
     * {@link CredentialUrlUtil#maskCredentialInUrl(String)} for the exact rules.
     *
     * <p>OpenAI uses header authentication - the canonical {@code https://api.openai.com}
     * URL does not contain credentials - but {@code rag.llm.openai.api.url} may point at a
     * gateway (Azure, vLLM, custom) that takes its credential as a query parameter, so all log
     * lines that include a URL route through this helper. The userinfo rule is defensive only:
     * HttpClient rejects a userinfo-bearing request URI outright, and this client now refuses such
     * an {@code api.url} before any call site that masks a URL is reached, so no production path
     * feeds userinfo to the masking rules at all.
     *
     * @param url the URL to mask (may be {@code null}).
     * @return the URL with credential values replaced by {@code ***}, or {@code null} when input is null.
     */
    static String maskCredentialInUrl(final String url) {
        return CredentialUrlUtil.maskCredentialInUrl(url);
    }

    /**
     * Returns the maximum number of attempts (initial + retries) for a single HTTP call.
     * Configured via {@code rag.llm.openai.retry.max} (default {@code 10}).
     *
     * <p>Worst-case sleep budget at default settings (base=2000ms, +/-20% jitter, no
     * {@code Retry-After} server hint): {@code 2 + 4 + 8 + ... + 512} approx {@code 1022s}
     * approx {@code 17 min} across 9 sleeps before the 10th attempt. With
     * {@code Retry-After} hints honored (each capped at 600s), the worst case approaches
     * {@code 9 * 600s = 90 min}. Tune down via this property when tighter latency bounds
     * are required.
     *
     * @return the maximum number of HTTP attempts (initial call plus retries).
     */
    protected int getRetryMaxAttempts() {
        return getConfigInt("retry.max", 10);
    }

    /**
     * Returns the base delay in milliseconds for exponential backoff between retries.
     * Configured via {@code rag.llm.openai.retry.base.delay.ms} (default {@code 2000}).
     *
     * @return the base backoff delay in milliseconds.
     */
    protected long getRetryBaseDelayMs() {
        return Long.parseLong(ComponentUtil.getFessConfig().getOrDefault(getConfigPrefix() + ".retry.base.delay.ms", "2000"));
    }

    /**
     * Whether to opt in to {@code stream_options.include_usage=true} on streaming requests so
     * the final SSE chunk carries token usage. Default {@code true}; set to {@code false} for
     * OpenAI-compatible backends that reject the field. Configured via
     * {@code rag.llm.openai.stream.include.usage}.
     *
     * @return {@code true} when stream usage reporting should be requested.
     */
    protected boolean isStreamUsageEnabled() {
        return Boolean.parseBoolean(ComponentUtil.getFessConfig().getOrDefault(getConfigPrefix() + ".stream.include.usage", "true"));
    }

    /**
     * Executes {@code call} with retry on {@link OpenAiRetry.RetryableHttpException}. {@link IOException},
     * {@link ParseException}, and {@link LlmException} (RuntimeException, NOT caught here -
     * matches Gemini's contract: connect-failures and parse-failures are not retried because
     * we cannot tell whether the request reached the server or not) all propagate immediately.
     *
     * <p>Backoff is exponential ({@code base * 2^(attempt-1)}) with +/-20% jitter, but a
     * server-provided {@code Retry-After} (in seconds) takes precedence per OpenAI guidance.
     *
     * @param operation log label (e.g. {@code "chat"}).
     * @param call the HTTP call body.
     */
    <T> T executeWithRetry(final String operation, final OpenAiRetry.HttpCall<T> call) throws IOException, ParseException {
        return executeWithRetry(operation, call, null);
    }

    /**
     * Same as {@link #executeWithRetry(String, HttpCall)} but additionally notifies the
     * given {@link LlmStreamCallback} (when non-{@code null}) between attempts via
     * {@link LlmStreamCallback#onRetry(String, int, int, long, Throwable)}.
     *
     * @param operation log label, e.g. {@code "chat"} or {@code "streamChat"}.
     * @param call the HTTP call body.
     * @param callback optional callback to notify on retry; may be {@code null}.
     * @param <T> the call result type.
     * @return the call result on success.
     */
    <T> T executeWithRetry(final String operation, final OpenAiRetry.HttpCall<T> call, final LlmStreamCallback callback)
            throws IOException, ParseException {
        final int maxAttempts = Math.max(1, getRetryMaxAttempts());
        final long baseDelay = Math.max(0L, getRetryBaseDelayMs());
        IOException lastIo = null;
        ParseException lastParse = null;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return call.call();
            } catch (final OpenAiRetry.RetryableHttpException e) {
                if (attempt == maxAttempts) {
                    logger.warn("[LLM:OPENAI] {} retry exhausted. attempts={}, lastStatus={}, retryAfter={}s", operation, attempt,
                            e.statusCode, e.retryAfterSeconds);
                    // Preserve the status-driven error code (429 -> rate_limit, 502/503 -> service_unavailable)
                    // across retry exhaustion; otherwise the outer catch in chat()/streamChat() would degrade
                    // every retryable failure to ERROR_CONNECTION and break downstream classification.
                    throw new LlmException("OpenAI API retryable error: " + e.statusCode + " " + e.reason, resolveErrorCode(e.statusCode),
                            e);
                }
                sleepBackoff(operation, attempt, maxAttempts, baseDelay, e, callback);
            } catch (final IOException e) {
                lastIo = e;
                break;
            } catch (final ParseException e) {
                lastParse = e;
                break;
            }
        }
        if (lastIo != null) {
            throw lastIo;
        }
        throw lastParse;
    }

    /**
     * Sleeps the computed backoff interval and notifies the supplied callback before
     * the actual sleep. When the {@link OpenAiRetry.RetryableHttpException} carries a non-negative
     * {@code Retry-After} hint (in seconds), it overrides the exponential-backoff +
     * jitter computation. Restores interrupt status if interrupted. Exceptions thrown
     * by the callback are swallowed (logged at DEBUG) so retry behavior is never
     * affected by callback bugs.
     *
     * @param operation log label.
     * @param attempt 1-based current attempt index.
     * @param maxAttempts total attempts including the first.
     * @param baseDelay base delay in milliseconds (already clamped to {@code >= 0}).
     * @param cause the {@link OpenAiRetry.RetryableHttpException} that triggered the retry.
     * @param callback optional callback to notify; may be {@code null}.
     * @throws IOException if the sleep is interrupted.
     */
    private void sleepBackoff(final String operation, final int attempt, final int maxAttempts, final long baseDelay,
            final OpenAiRetry.RetryableHttpException cause, final LlmStreamCallback callback) throws IOException {
        final long delayMs;
        if (cause.retryAfterSeconds >= 0L) {
            delayMs = cause.retryAfterSeconds * 1000L;
        } else {
            final long jitter = (long) (baseDelay * 0.2 * ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
            delayMs = (long) (baseDelay * Math.pow(2, attempt - 1)) + jitter;
        }
        final long sleepMs = Math.max(0, delayMs);
        logger.info("[LLM:OPENAI] {} retrying. attempt={}/{}, status={}, retryAfter={}s, sleepMs={}", operation, attempt, maxAttempts,
                cause.statusCode, cause.retryAfterSeconds, sleepMs);
        if (callback != null) {
            try {
                callback.onRetry(operation, attempt, maxAttempts, sleepMs, cause);
            } catch (final Exception cbEx) {
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OPENAI] onRetry callback threw. error={}", cbEx.getMessage());
                }
            }
        }
        try {
            Thread.sleep(sleepMs);
        } catch (final InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IOException("Retry interrupted", ie);
        }
    }

    /**
     * Renders an OpenAI error response body as a single-line diagnostic. Returns
     * {@code "type=...,code=...,param=...,message=..."} when the body parses as the
     * documented {@code {"error":{...}}} envelope; otherwise returns the body trimmed
     * so non-JSON gateway pages remain readable
     * in logs. See {@link OpenAiErrorBody#render(String)} for the exact rendering rules.
     *
     * @param errorBody the raw HTTP response body from a failed OpenAI API call.
     * @return a single-line diagnostic suitable for logging.
     */
    protected String extractErrorDetails(final String errorBody) {
        return OpenAiErrorBody.render(errorBody);
    }

    @Override
    public String getName() {
        return NAME;
    }

    /**
     * Guards the one-shot ERROR emitted by {@link #isUserInfoApiUrlRefused(String)}. The
     * availability check runs on a timer, so reporting the refusal on every pass would flood the
     * log for as long as the misconfiguration lasts. Cleared again as soon as a check sees a URL
     * without userinfo, so a re-broken configuration is reported afresh.
     */
    private final AtomicBoolean userInfoRefusalReported = new AtomicBoolean();

    /**
     * Returns whether the configured {@code api.url} must be refused because its authority carries
     * a userinfo credential, reporting the reason and the remedy at ERROR the first time.
     *
     * <p>This <em>fails closed</em> - it reports the client unavailable rather than throwing.
     * {@link #checkAvailabilityNow()} is reached from {@code init()}, which the DI container runs
     * as a {@code postConstruct} init method during eager assembly
     * ({@code init} -&gt; {@code startAvailabilityCheck} -&gt; {@code updateAvailability} -&gt;
     * {@code checkAvailabilityNow}); an exception escaping there would abort container startup, so
     * a bad configuration value would stop the whole application from starting rather than
     * disabling one optional client. The request paths ({@link #chat} / {@link #streamChat}) are
     * never reached during initialization and do throw, because they have no other way to report
     * failure.
     *
     * <p>Nothing about the URL is logged: the URL is what holds the credential, and the userinfo
     * masking rule does not cover a credential containing whitespace.
     *
     * @param apiUrl the configured API URL.
     * @return true when the URL carries userinfo and no request may be attempted.
     */
    private boolean isUserInfoApiUrlRefused(final String apiUrl) {
        if (!CredentialUrlUtil.hasUserInfo(apiUrl)) {
            userInfoRefusalReported.set(false);
            return false;
        }
        if (userInfoRefusalReported.compareAndSet(false, true)) {
            logger.error("[LLM:OPENAI] OpenAI is not available. {}", HttpRequestFactory.userInfoRejectedMessage(userInfoConfigKey()));
        }
        return true;
    }

    /** The configuration key named by a userinfo refusal. */
    private String userInfoConfigKey() {
        return getConfigPrefix() + ".api.url";
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
        if (isUserInfoApiUrlRefused(apiUrl)) {
            return false;
        }
        final String maskedUrl = maskCredentialInUrl(apiUrl);
        try {
            final HttpGet request = HttpRequestFactory.createGet(apiUrl + "/models", userInfoConfigKey());
            request.addHeader("Authorization", "Bearer " + apiKey);
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                final boolean available = statusCode >= 200 && statusCode < 300;
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OPENAI] OpenAI availability check. url={}, statusCode={}, available={}", maskedUrl, statusCode,
                            available);
                }
                return available;
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] OpenAI is not available. url={}, error={}", maskedUrl, e.getMessage());
            }
            return false;
        }
    }

    @Override
    public LlmChatResponse chat(final LlmChatRequest request) {
        final String apiUrl = getApiUrl();
        if (CredentialUrlUtil.hasUserInfo(apiUrl)) {
            // Refused before the masked URL is computed, let alone logged: the masking rules do
            // not cover a userinfo credential containing whitespace, so the url={} field of the
            // failure log would otherwise hand that credential back verbatim.
            throw new LlmException(HttpRequestFactory.userInfoRejectedMessage(userInfoConfigKey()), LlmException.ERROR_CONNECTION);
        }
        final String url = apiUrl + "/chat/completions";
        final String maskedUrl = maskCredentialInUrl(url);
        final Map<String, Object> requestBody = buildRequestBody(request, false);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OPENAI] Sending chat request to OpenAI. url={}, model={}, messageCount={}", maskedUrl,
                    requestBody.get("model"), request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] requestBody={}", json);
            }
            final HttpPost httpRequest = HttpRequestFactory.createPost(url, userInfoConfigKey());
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON)); // repeatable per HttpClient 5 contract
            httpRequest.addHeader("Authorization", "Bearer " + getApiKey());

            return executeWithRetry("chat", () -> {
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (statusCode < 200 || statusCode >= 300) {
                        String errorBody = "";
                        if (response.getEntity() != null) {
                            try {
                                errorBody = EntityUtils.toString(response.getEntity());
                            } catch (final IOException e) { /* ignore */ }
                        }
                        final String errorDetails = extractErrorDetails(errorBody);
                        logger.warn("[LLM:OPENAI] API error. url={}, statusCode={}, message={}, error={}", maskedUrl, statusCode,
                                response.getReasonPhrase(), errorDetails);
                        if (OpenAiRetry.isRetryableStatus(statusCode)) {
                            final var ra = response.getFirstHeader("Retry-After");
                            final long retryAfter = OpenAiRetry.parseRetryAfterSeconds(ra != null ? ra.getValue() : null);
                            throw new OpenAiRetry.RetryableHttpException(statusCode, response.getReasonPhrase(), retryAfter);
                        }
                        throw new LlmException("OpenAI API error: " + statusCode + " " + response.getReasonPhrase(),
                                resolveErrorCode(statusCode));
                    }

                    final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                    if (logger.isDebugEnabled()) {
                        logger.debug("[LLM:OPENAI] responseBody={}", responseBody);
                    }
                    final JsonNode jsonNode = objectMapper.readTree(responseBody);

                    final LlmChatResponse chatResponse = new LlmChatResponse();
                    String refusal = null;
                    if (jsonNode.has("choices") && jsonNode.get("choices").isArray() && jsonNode.get("choices").size() > 0) {
                        final JsonNode firstChoice = jsonNode.get("choices").get(0);
                        if (firstChoice.has("message")) {
                            final JsonNode message = firstChoice.get("message");
                            if (message.has("content") && !message.get("content").isNull()) {
                                chatResponse.setContent(message.get("content").asText());
                            }
                            if (message.has("refusal") && !message.get("refusal").isNull()) {
                                refusal = message.get("refusal").asText();
                            }
                        }
                        if (firstChoice.has("finish_reason") && !firstChoice.get("finish_reason").isNull()) {
                            chatResponse.setFinishReason(firstChoice.get("finish_reason").asText());
                        }
                    }
                    if (jsonNode.has("model")) {
                        chatResponse.setModel(jsonNode.get("model").asText());
                    }
                    final String responseId = jsonNode.has("id") && !jsonNode.get("id").isNull() ? jsonNode.get("id").asText() : null;
                    final String systemFingerprint = jsonNode.has("system_fingerprint") && !jsonNode.get("system_fingerprint").isNull()
                            ? jsonNode.get("system_fingerprint").asText()
                            : null;
                    Integer reasoningTokens = null;
                    Integer cachedTokens = null;
                    if (jsonNode.has("usage")) {
                        final JsonNode usage = jsonNode.get("usage");
                        if (usage.has("prompt_tokens"))
                            chatResponse.setPromptTokens(usage.get("prompt_tokens").asInt());
                        if (usage.has("completion_tokens"))
                            chatResponse.setCompletionTokens(usage.get("completion_tokens").asInt());
                        if (usage.has("total_tokens"))
                            chatResponse.setTotalTokens(usage.get("total_tokens").asInt());
                        if (usage.has("completion_tokens_details") && usage.get("completion_tokens_details").has("reasoning_tokens")) {
                            reasoningTokens = usage.get("completion_tokens_details").get("reasoning_tokens").asInt();
                        }
                        if (usage.has("prompt_tokens_details") && usage.get("prompt_tokens_details").has("cached_tokens")) {
                            cachedTokens = usage.get("prompt_tokens_details").get("cached_tokens").asInt();
                        }
                    }

                    logger.info(
                            "[LLM:OPENAI] Chat response received. model={}, id={}, systemFingerprint={}, "
                                    + "promptTokens={}, cachedTokens={}, completionTokens={}, reasoningTokens={}, "
                                    + "totalTokens={}, finishReason={}, contentLength={}, elapsedTime={}ms",
                            chatResponse.getModel(), responseId, systemFingerprint, chatResponse.getPromptTokens(), cachedTokens,
                            chatResponse.getCompletionTokens(), reasoningTokens, chatResponse.getTotalTokens(),
                            chatResponse.getFinishReason(), chatResponse.getContent() != null ? chatResponse.getContent().length() : 0,
                            System.currentTimeMillis() - startTime);
                    if (isAbnormalFinishReason(chatResponse.getFinishReason())) {
                        logger.warn(
                                "[LLM:OPENAI] Chat finished abnormally. id={}, finishReason={}, "
                                        + "completionTokens={}, reasoningTokens={}, contentLength={}, model={}",
                                responseId, chatResponse.getFinishReason(), chatResponse.getCompletionTokens(), reasoningTokens,
                                chatResponse.getContent() != null ? chatResponse.getContent().length() : 0, chatResponse.getModel());
                    }
                    if (refusal != null) {
                        // Mirror streamChat's WARN on delta.refusal: structured-output refusals can pair with
                        // finish_reason=stop and null content, which would otherwise be silently logged as a
                        // normal empty success.
                        logger.warn("[LLM:OPENAI] Chat refusal. id={}, refusal={}, model={}", responseId, refusal, chatResponse.getModel());
                    }
                    return chatResponse;
                }
            });
        } catch (final LlmException e) {
            throw e;
        } catch (final OpenAiRetry.RetryableHttpException e) {
            // Defensive: executeWithRetry consumes OpenAiRetry.RetryableHttpException; this should never fire.
            throw new LlmException("OpenAI API retryable exhausted", LlmException.ERROR_CONNECTION, e);
        } catch (final Exception e) {
            logger.warn("[LLM:OPENAI] Failed to call OpenAI API. url={}, error={}", maskedUrl, e.getMessage(), e);
            throw new LlmException("Failed to call OpenAI API", LlmException.ERROR_CONNECTION, e);
        }
    }

    /**
     * Summary of a single streamChat invocation. Exposed for diagnostics, not part of the LLM SPI.
     */
    public static final class StreamSummary {
        /** Number of SSE {@code data:} lines received (including the terminal usage chunk). */
        public final int chunkCount;
        /** Number of parsed JSON chunk objects (excludes {@code [DONE]} and SSE comments). */
        public final int objectCount;
        /** Final {@code finish_reason} reported by the server (e.g. {@code stop}, {@code length}). */
        public final String finishReason;
        /** OpenAI response id ({@code id} field) for log correlation; {@code null} if absent. */
        public final String responseId;
        /** OpenAI {@code system_fingerprint} for backend-version pinning; {@code null} if absent. */
        public final String systemFingerprint;
        /** Prompt tokens reported by the terminal usage chunk; {@code null} if usage is disabled or omitted. */
        public final Integer promptTokens;
        /** Cached prompt tokens ({@code prompt_tokens_details.cached_tokens}); {@code null} when absent. */
        public final Integer cachedTokens;
        /** Completion tokens reported by the terminal usage chunk; {@code null} when absent. */
        public final Integer completionTokens;
        /** Reasoning tokens ({@code completion_tokens_details.reasoning_tokens}); {@code null} for non-reasoning models. */
        public final Integer reasoningTokens;
        /** Total tokens reported by the terminal usage chunk; {@code null} when absent. */
        public final Integer totalTokens;
        /** Wall-clock milliseconds from request start to the first chunk arriving. */
        public final long firstChunkMs;
        /** Wall-clock milliseconds for the full streaming call (request start to stream close). */
        public final long elapsedMs;

        StreamSummary(final int chunkCount, final int objectCount, final String finishReason, final String responseId,
                final String systemFingerprint, final Integer promptTokens, final Integer cachedTokens, final Integer completionTokens,
                final Integer reasoningTokens, final Integer totalTokens, final long firstChunkMs, final long elapsedMs) {
            this.chunkCount = chunkCount;
            this.objectCount = objectCount;
            this.finishReason = finishReason;
            this.responseId = responseId;
            this.systemFingerprint = systemFingerprint;
            this.promptTokens = promptTokens;
            this.cachedTokens = cachedTokens;
            this.completionTokens = completionTokens;
            this.reasoningTokens = reasoningTokens;
            this.totalTokens = totalTokens;
            this.firstChunkMs = firstChunkMs;
            this.elapsedMs = elapsedMs;
        }
    }

    /** Test hook; not thread-safe. Set once before invoking streamChat from a single thread. */
    private java.util.function.Consumer<StreamSummary> streamSummaryConsumer;

    /** Test hook: receives the per-call {@link StreamSummary} right after the completion log line. */
    void setStreamSummaryConsumer(final java.util.function.Consumer<StreamSummary> consumer) {
        this.streamSummaryConsumer = consumer;
    }

    @Override
    public void streamChat(final LlmChatRequest request, final LlmStreamCallback callback) {
        final String apiUrl = getApiUrl();
        if (CredentialUrlUtil.hasUserInfo(apiUrl)) {
            // Refused before the masked URL is computed, let alone logged - see chat(). The
            // callback is notified first so onError stays symmetric with every other failure.
            final LlmException e =
                    new LlmException(HttpRequestFactory.userInfoRejectedMessage(userInfoConfigKey()), LlmException.ERROR_CONNECTION);
            callback.onError(e);
            throw e;
        }
        final String url = apiUrl + "/chat/completions";
        final String maskedUrl = maskCredentialInUrl(url);
        final Map<String, Object> requestBody = buildRequestBody(request, true);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OPENAI] Starting streaming chat request to OpenAI. url={}, model={}, messageCount={}", maskedUrl,
                    requestBody.get("model"), request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OPENAI] requestBody={}", json);
            }
            final HttpPost httpRequest = HttpRequestFactory.createPost(url, userInfoConfigKey());
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));
            httpRequest.addHeader("Authorization", "Bearer " + getApiKey());

            executeWithRetry("streamChat", () -> {
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (logger.isDebugEnabled()) {
                        final var ctHeader = response.getFirstHeader("Content-Type");
                        logger.debug("[LLM:OPENAI] /chat/completions stream http response. statusCode={}, contentType={}", statusCode,
                                ctHeader != null ? ctHeader.getValue() : null);
                    }
                    if (statusCode < 200 || statusCode >= 300) {
                        String errorBody = "";
                        if (response.getEntity() != null) {
                            try {
                                errorBody = EntityUtils.toString(response.getEntity());
                            } catch (final IOException | ParseException e) { /* ignore */ }
                        }
                        final String errorDetails = extractErrorDetails(errorBody);
                        logger.warn("[LLM:OPENAI] Streaming API error. url={}, statusCode={}, message={}, error={}", maskedUrl, statusCode,
                                response.getReasonPhrase(), errorDetails);
                        if (OpenAiRetry.isRetryableStatus(statusCode)) {
                            final var ra = response.getFirstHeader("Retry-After");
                            final long retryAfter = OpenAiRetry.parseRetryAfterSeconds(ra != null ? ra.getValue() : null);
                            throw new OpenAiRetry.RetryableHttpException(statusCode, response.getReasonPhrase(), retryAfter);
                        }
                        throw new LlmException("OpenAI API error: " + statusCode + " " + response.getReasonPhrase(),
                                resolveErrorCode(statusCode));
                    }
                    if (response.getEntity() == null) {
                        logger.warn("[LLM:OPENAI] Empty response from OpenAI streaming API. url={}", maskedUrl);
                        throw new LlmException("Empty response from OpenAI");
                    }

                    int chunkCount = 0;
                    int objectCount = 0;
                    long firstChunkTime = 0;
                    String lastResponseId = null;
                    String lastSystemFingerprint = null;
                    String lastFinishReason = null;
                    String lastRefusal = null;
                    Integer promptTokens = null;
                    Integer cachedTokens = null;
                    Integer completionTokens = null;
                    Integer reasoningTokens = null;
                    Integer totalTokens = null;
                    boolean terminalCallbackSent = false;

                    try (BufferedReader reader =
                            new BufferedReader(new InputStreamReader(response.getEntity().getContent(), StandardCharsets.UTF_8))) {
                        String line;
                        boolean streamDone = false;
                        while (!streamDone && (line = reader.readLine()) != null) {
                            if (StringUtil.isBlank(line)) {
                                continue;
                            }
                            if (line.charAt(0) == ':') {
                                continue; // SSE comment
                            }
                            if (!line.startsWith(SSE_DATA_PREFIX)) {
                                continue;
                            }
                            final String data = line.substring(SSE_DATA_PREFIX.length()).trim();
                            if (SSE_DONE_MARKER.equals(data)) {
                                if (!terminalCallbackSent) {
                                    callback.onChunk("", true);
                                    terminalCallbackSent = true;
                                }
                                streamDone = true;
                                continue;
                            }

                            try {
                                final JsonNode jsonNode = objectMapper.readTree(data);
                                objectCount++;
                                if (logger.isDebugEnabled()) {
                                    logger.debug("[LLM:OPENAI] streamObject#{} json={}", objectCount, data);
                                }
                                if (lastResponseId == null && jsonNode.has("id") && !jsonNode.get("id").isNull()) {
                                    lastResponseId = jsonNode.get("id").asText();
                                }
                                if (lastSystemFingerprint == null && jsonNode.has("system_fingerprint")
                                        && !jsonNode.get("system_fingerprint").isNull()) {
                                    lastSystemFingerprint = jsonNode.get("system_fingerprint").asText();
                                }
                                if (jsonNode.has("usage") && !jsonNode.get("usage").isNull()) {
                                    final JsonNode u = jsonNode.get("usage");
                                    if (u.has("prompt_tokens")) {
                                        promptTokens = u.get("prompt_tokens").asInt();
                                    }
                                    if (u.has("completion_tokens")) {
                                        completionTokens = u.get("completion_tokens").asInt();
                                    }
                                    if (u.has("total_tokens")) {
                                        totalTokens = u.get("total_tokens").asInt();
                                    }
                                    if (u.has("completion_tokens_details") && u.get("completion_tokens_details").has("reasoning_tokens")) {
                                        reasoningTokens = u.get("completion_tokens_details").get("reasoning_tokens").asInt();
                                    }
                                    if (u.has("prompt_tokens_details") && u.get("prompt_tokens_details").has("cached_tokens")) {
                                        cachedTokens = u.get("prompt_tokens_details").get("cached_tokens").asInt();
                                    }
                                }
                                if (jsonNode.has("choices") && jsonNode.get("choices").isArray() && jsonNode.get("choices").size() > 0) {
                                    final JsonNode firstChoice = jsonNode.get("choices").get(0);
                                    final boolean done = firstChoice.has("finish_reason") && !firstChoice.get("finish_reason").isNull();
                                    if (done) {
                                        lastFinishReason = firstChoice.get("finish_reason").asText();
                                    }
                                    final JsonNode delta = firstChoice.has("delta") ? firstChoice.get("delta") : null;
                                    if (delta != null && delta.has("refusal") && !delta.get("refusal").isNull()) {
                                        final String r = delta.get("refusal").asText();
                                        lastRefusal = (lastRefusal == null) ? r : lastRefusal + r;
                                    }
                                    if (delta != null && delta.has("content") && !delta.get("content").isNull()) {
                                        final String content = delta.get("content").asText();
                                        callback.onChunk(content, done);
                                        if (done) {
                                            terminalCallbackSent = true;
                                        }
                                        if (chunkCount == 0) {
                                            firstChunkTime = System.currentTimeMillis() - startTime;
                                        }
                                        chunkCount++;
                                    } else if (done && !terminalCallbackSent) {
                                        callback.onChunk("", true);
                                        terminalCallbackSent = true;
                                    }
                                    // Continue reading even after done — usage chunk arrives next.
                                }
                                // Usage-only chunk has empty choices[]; size>0 guard above keeps callback silent.
                            } catch (final JsonProcessingException e) {
                                logger.warn("[LLM:OPENAI] Failed to parse streaming response. line={}", line, e);
                            }
                        }
                    }

                    final long elapsed = System.currentTimeMillis() - startTime;
                    logger.info(
                            "[LLM:OPENAI] Stream completed. chunkCount={}, objectCount={}, firstChunkMs={}, elapsedTime={}ms, "
                                    + "id={}, systemFingerprint={}, finishReason={}, promptTokens={}, cachedTokens={}, "
                                    + "completionTokens={}, reasoningTokens={}, totalTokens={}",
                            chunkCount, objectCount, firstChunkTime, elapsed, lastResponseId, lastSystemFingerprint, lastFinishReason,
                            promptTokens, cachedTokens, completionTokens, reasoningTokens, totalTokens);
                    if (isAbnormalFinishReason(lastFinishReason)) {
                        logger.warn(
                                "[LLM:OPENAI] Stream finished abnormally. id={}, finishReason={}, chunkCount={}, "
                                        + "completionTokens={}, reasoningTokens={}, model={}",
                                lastResponseId, lastFinishReason, chunkCount, completionTokens, reasoningTokens, requestBody.get("model"));
                    }
                    if (lastRefusal != null) {
                        logger.warn("[LLM:OPENAI] Stream refusal. id={}, refusal={}, model={}", lastResponseId, lastRefusal,
                                requestBody.get("model"));
                    }
                    if (streamSummaryConsumer != null) {
                        streamSummaryConsumer.accept(new StreamSummary(chunkCount, objectCount, lastFinishReason, lastResponseId,
                                lastSystemFingerprint, promptTokens, cachedTokens, completionTokens, reasoningTokens, totalTokens,
                                firstChunkTime, elapsed));
                    }
                    return null;
                }
            }, callback);
        } catch (final LlmException e) {
            callback.onError(e);
            throw e;
        } catch (final OpenAiRetry.RetryableHttpException e) {
            // Defensive: executeWithRetry consumes OpenAiRetry.RetryableHttpException; this should never fire.
            final LlmException llm = new LlmException("OpenAI API retryable exhausted", LlmException.ERROR_CONNECTION, e);
            callback.onError(llm);
            throw llm;
        } catch (final IOException | ParseException | RuntimeException e) {
            // RuntimeException covers consumer onChunk failures and unexpected JSON / runtime
            // errors; the callback is always notified before propagating so onError stays symmetric with chat().
            logger.warn("[LLM:OPENAI] Failed to stream from OpenAI API. url={}, error={}", maskedUrl, e.getMessage(), e);
            final LlmException llm = new LlmException("Failed to stream from OpenAI API", LlmException.ERROR_CONNECTION, e);
            callback.onError(llm);
            throw llm;
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

        if (stream && isStreamUsageEnabled()) {
            final Map<String, Object> streamOptions = new HashMap<>();
            streamOptions.put("include_usage", Boolean.TRUE);
            body.put("stream_options", streamOptions);
        }

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

        // Reasoning models reject these outright (HTTP 400 "Unsupported parameter"), exactly as
        // they reject a non-default temperature. Sending them would fail the whole chat rather
        // than degrade it, so they are dropped with a warning naming the model.
        final boolean sampling = supportsSamplingParams(model);
        putDoubleParam(body, request, "top_p", sampling, model);
        putDoubleParam(body, request, "frequency_penalty", sampling, model);
        putDoubleParam(body, request, "presence_penalty", sampling, model);

        return body;
    }

    /**
     * Copies a numeric extra param into the request body, unless the model does not accept it.
     *
     * @param body the request body being built.
     * @param request the chat request carrying the configured extra params.
     * @param name the OpenAI request-body field name, which is also the extra-param key.
     * @param supported whether this model accepts the parameter at all.
     * @param model the model name, for the log line when the parameter is dropped.
     */
    protected void putDoubleParam(final Map<String, Object> body, final LlmChatRequest request, final String name, final boolean supported,
            final String model) {
        final String value = request.getExtraParam(name);
        if (value == null) {
            return;
        }
        if (!supported) {
            // Warn rather than drop silently: the value came from an explicit
            // rag.llm.openai.<promptType>.<name> setting, and silence would read as "ignored".
            logger.warn("[LLM:OPENAI] {} is not supported by model {} and was not sent. Remove the "
                    + "rag.llm.openai.*.{} setting for this model.", name, model, name);
            return;
        }
        try {
            body.put(name, Double.parseDouble(value));
        } catch (final NumberFormatException e) {
            logger.warn("[LLM:OPENAI] Invalid {} value: {}", name, value);
        }
    }

    /**
     * Determines whether the given model accepts the sampling parameters {@code top_p},
     * {@code frequency_penalty} and {@code presence_penalty}.
     *
     * <p>Reasoning models accept none of them - verified against the live API, which answers
     * {@code 400 Unsupported parameter: 'top_p' is not supported with this model.} and the
     * equivalent for both penalties. This mirrors {@link #supportsTemperature(String)}, which
     * already guards the fourth member of the same group.
     *
     * @param model the model name.
     * @return true if the model supports the sampling parameters.
     */
    protected boolean supportsSamplingParams(final String model) {
        return !isReasoningModel(model);
    }

    /**
     * Determines whether the given model requires the "max_completion_tokens" parameter
     * instead of the legacy "max_tokens" parameter.
     *
     * @param model the model name
     * @return true if the model uses max_completion_tokens
     */
    protected boolean useMaxCompletionTokens(final String model) {
        return isReasoningModel(model);
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
        return !isReasoningModel(model);
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
        return getConfigInt("timeout", 120000);
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
     * For reasoning models, multiplies the default max tokens to account for
     * internal reasoning token consumption, and sets reasoning_effort to "low"
     * for simple classification tasks.
     *
     * @param request the LLM chat request
     * @param promptType the prompt type (e.g. "intent", "evaluation", "answer")
     */
    protected void applyDefaultParams(final LlmChatRequest request, final String promptType) {
        final boolean maxTokensSetByUser = request.getMaxTokens() != null;
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
        case "queryregeneration":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
            }
            break;
        default:
            break;
        }

        // For reasoning models, apply token multiplier and default reasoning_effort
        final String model = getModel();
        if (isReasoningModel(model)) {
            // Multiply default max tokens if not explicitly set by user
            if (!maxTokensSetByUser && request.getMaxTokens() != null) {
                final int multiplier = getReasoningTokenMultiplier();
                request.setMaxTokens(request.getMaxTokens() * multiplier);
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OPENAI] Applied reasoning token multiplier. promptType={}, maxTokens={}, multiplier={}", promptType,
                            request.getMaxTokens(), multiplier);
                }
            }

            // Set default reasoning_effort for simple tasks
            if (request.getExtraParam("reasoning_effort") == null) {
                switch (promptType) {
                case "intent":
                case "evaluation":
                case "docnotfound":
                case "unclear":
                case "noresults":
                case "queryregeneration":
                    request.putExtraParam("reasoning_effort", "low");
                    if (logger.isDebugEnabled()) {
                        logger.debug("[LLM:OPENAI] Applied default reasoning_effort=low. promptType={}", promptType);
                    }
                    break;
                default:
                    break;
                }
            }
        }
    }

    /**
     * Gets the reasoning token multiplier for reasoning models.
     * Reasoning models consume part of max_completion_tokens for internal reasoning,
     * so default token limits need to be increased to ensure sufficient output tokens.
     *
     * @return the multiplier (default: 4)
     */
    protected int getReasoningTokenMultiplier() {
        return Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.openai.reasoning.token.multiplier", "4"));
    }

    @Override
    protected int getAvailabilityCheckInterval() {
        return getConfigInt("availability.check.interval", 60);
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
        return getConfigInt("chat.evaluation.max.relevant.docs", 3);
    }

    @Override
    protected int getEvaluationDescriptionMaxChars() {
        return getConfigInt("chat.evaluation.description.max.chars", 500);
    }

    @Override
    protected int getHistoryMaxChars() {
        return getConfigInt("history.max.chars", 8000);
    }

    @Override
    protected int getIntentHistoryMaxMessages() {
        return getConfigInt("intent.history.max.messages", 8);
    }

    @Override
    protected int getIntentHistoryMaxChars() {
        return getConfigInt("intent.history.max.chars", 4000);
    }

    @Override
    public int getHistoryAssistantMaxChars() {
        return getConfigInt("history.assistant.max.chars", 800);
    }

    @Override
    public int getHistoryAssistantSummaryMaxChars() {
        return getConfigInt("history.assistant.summary.max.chars", 800);
    }

}
