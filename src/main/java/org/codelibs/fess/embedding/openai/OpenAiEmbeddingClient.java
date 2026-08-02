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
package org.codelibs.fess.embedding.openai;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.codelibs.core.lang.StringUtil;
import org.codelibs.fess.embedding.AbstractEmbeddingClient;
import org.codelibs.fess.embedding.EmbeddingException;
import org.codelibs.fess.openai.util.HttpRequestFactory;
import org.codelibs.fess.util.CredentialUrlUtil;
import org.codelibs.fess.openai.util.OpenAiErrorBody;
import org.codelibs.fess.openai.util.OpenAiRetry;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Embedding client implementation for OpenAI's Embeddings API.
 * Calls OpenAI's {@code POST /embeddings} endpoint.
 *
 * <p>Unlike {@code OllamaEmbeddingClient}, this client's retry/timeout/error-handling
 * conventions mirror {@link org.codelibs.fess.llm.openai.OpenAiLlmClient} from this same
 * repository: a server-provided {@code Retry-After} header overrides the computed
 * exponential backoff, {@link IOException} and {@link ParseException} are never retried
 * (the request may already have reached the server), and the default timeout is
 * {@code 120000}ms.
 *
 * <p>Unlike {@code OllamaEmbeddingClient} (text prefixes) or {@code GeminiEmbeddingClient}
 * ({@code task_type}), this client has no mechanism to distinguish a document/passage
 * embedding call from a query embedding call: OpenAI's {@code /v1/embeddings} API takes no
 * such parameter, and its models are designed to be used symmetrically for both. {@link
 * #embedDocuments(List)} and {@link #embedQuery(List)} are therefore intentionally
 * behaviorally identical, both delegating to the same request/response handling.
 *
 * @see <a href="https://developers.openai.com/api/docs/guides/embeddings">OpenAI Embeddings API</a>
 */
public class OpenAiEmbeddingClient extends AbstractEmbeddingClient {

    private static final Logger logger = LogManager.getLogger(OpenAiEmbeddingClient.class);

    /** Shared ObjectMapper instance for JSON processing. */
    protected static final ObjectMapper objectMapper = new ObjectMapper();

    /** The name identifier for the OpenAI embedding client. */
    protected static final String NAME = "openai";

    /** Model name prefix that supports the {@code dimensions} truncation parameter. */
    private static final String DIMENSIONS_SUPPORTED_MODEL_PREFIX = "text-embedding-3";

    /**
     * Config key suffix (under {@link #getConfigPrefix()}) overriding whether the {@code dimensions}
     * request parameter is sent. See {@link #supportsDimensionsParam(String)}.
     */
    private static final String CONFIG_DIMENSIONS_ENABLED = "dimensions.enabled";

    /** {@link #CONFIG_DIMENSIONS_ENABLED} value selecting model-name inference (the default). */
    private static final String DIMENSIONS_ENABLED_AUTO = "auto";

    /** {@link #CONFIG_DIMENSIONS_ENABLED} value forcing the {@code dimensions} parameter on. */
    private static final String DIMENSIONS_ENABLED_TRUE = "true";

    /** {@link #CONFIG_DIMENSIONS_ENABLED} value forcing the {@code dimensions} parameter off. */
    private static final String DIMENSIONS_ENABLED_FALSE = "false";

    /** Config key suffix (under {@link #getConfigPrefix()}) for the OpenAI API key. */
    private static final String CONFIG_API_KEY = "api.key";

    /** Config key suffix (under {@link #getConfigPrefix()}) for the OpenAI API base URL. */
    private static final String CONFIG_API_URL = "api.url";

    /** Config key suffix (under {@link #getConfigPrefix()}) for the embedding model name. */
    private static final String CONFIG_MODEL = "model";

    /**
     * Config key suffix (under {@link #getConfigPrefix()}) for the exponential-backoff base delay.
     * See {@link #getRetryBaseDelayMs()}.
     */
    private static final String CONFIG_RETRY_BASE_DELAY_MS = "retry.base.delay.ms";

    /**
     * Config key suffix (under {@link #getConfigPrefix()}) for the per-sleep backoff cap.
     * See {@link #getRetryMaxDelayMs()}.
     */
    private static final String CONFIG_RETRY_MAX_DELAY_MS = "retry.max.delay.ms";

    /**
     * OpenAI's documented maximum number of inputs in a single {@code /v1/embeddings}
     * request array. A request exceeding this is rejected with a non-retryable {@code 400},
     * so the input list is split into sub-batches of at most this many items.
     *
     * @see <a href="https://developers.openai.com/api/docs/guides/embeddings">OpenAI Embeddings API</a>
     */
    private static final int MAX_BATCH_ITEMS = 2048;

    /**
     * OpenAI's documented maximum total token count summed across all inputs of a single
     * {@code /v1/embeddings} request. Like the array-size cap, exceeding it yields a
     * non-retryable {@code 400}, so sub-batches are also split to keep their estimated
     * cumulative token count under this ceiling.
     *
     * @see <a href="https://developers.openai.com/api/docs/guides/embeddings">OpenAI Embeddings API</a>
     */
    private static final int MAX_BATCH_TOKENS_ESTIMATE = 300_000;

    /**
     * Token budget used when deciding sub-batch boundaries, held below
     * {@link #MAX_BATCH_TOKENS_ESTIMATE} so the deliberately-approximate token estimate
     * ({@link #estimateTokens(String)}) has headroom before the real, exact limit.
     */
    private static final long BATCH_TOKEN_BUDGET = (long) (MAX_BATCH_TOKENS_ESTIMATE * 0.95);

    /**
     * Approximate characters-per-token for non-CJK text, per OpenAI's own rule of thumb
     * (~4 characters per token for English). Used only to place sub-batch boundaries, never
     * for anything exact.
     */
    private static final int APPROX_CHARS_PER_TOKEN = 4;

    /**
     * Numerator of the tokens-per-character weight applied to {@link #isCjk(int)} characters,
     * over {@link #CJK_TOKENS_PER_CHAR_DENOMINATOR}.
     *
     * <p>One token per character is <em>not</em> conservative enough. Measured against
     * {@code text-embedding-3-small}'s reported {@code usage.prompt_tokens} over Japanese prose,
     * a one-token-per-CJK-character estimate came in at 1.24-1.35x <em>under</em> the real count,
     * and {@link #BATCH_TOKEN_BUDGET} only holds 5% of headroom - so a sub-batch packed to the
     * budget still exceeded {@link #MAX_BATCH_TOKENS_ESTIMATE} and drew a non-retryable
     * {@code 400}. At 3/2 the same corpus estimates 1.15-1.27x <em>over</em> the real count,
     * which keeps a full sub-batch inside the real limit. Latin text is unaffected: it has no
     * CJK characters, so its estimate is unchanged.
     *
     * <p>This only moves sub-batch boundaries. It is a heuristic, and
     * {@link #doEmbed(List)} still splits and retries if the real tokenizer disagrees.
     */
    private static final int CJK_TOKENS_PER_CHAR_NUMERATOR = 3;

    /** Denominator of the CJK tokens-per-character weight. See {@link #CJK_TOKENS_PER_CHAR_NUMERATOR}. */
    private static final int CJK_TOKENS_PER_CHAR_DENOMINATOR = 2;

    /**
     * Default constructor.
     */
    public OpenAiEmbeddingClient() {
        // Default constructor
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
     * disabling one optional client. The embed path ({@link #callEmbedApi(List)}) is never reached
     * during initialization and does throw, because it has no other way to report failure.
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
            logger.error("[Embedding:OPENAI] OpenAI is not available. {}", HttpRequestFactory.userInfoRejectedMessage(userInfoConfigKey()));
        }
        return true;
    }

    /** The configuration key named by a userinfo refusal. */
    private String userInfoConfigKey() {
        return getConfigPrefix() + "." + CONFIG_API_URL;
    }

    @Override
    protected boolean checkAvailabilityNow() {
        final String apiKey = getApiKey();
        if (StringUtil.isBlank(apiKey)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[Embedding:OPENAI] OpenAI is not available. apiKey is blank");
            }
            return false;
        }
        final String apiUrl = getApiUrl();
        if (StringUtil.isBlank(apiUrl)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[Embedding:OPENAI] OpenAI is not available. apiUrl is blank");
            }
            return false;
        }
        if (isUserInfoApiUrlRefused(apiUrl)) {
            return false;
        }
        final String maskedUrl = maskCredentialInUrl(apiUrl);
        try {
            final HttpGet request = HttpRequestFactory.createGet(appendPath(apiUrl, "/models"), userInfoConfigKey());
            request.addHeader("Authorization", "Bearer " + apiKey);
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                final boolean available = statusCode >= 200 && statusCode < 300;
                if (logger.isDebugEnabled()) {
                    logger.debug("[Embedding:OPENAI] OpenAI availability check. url={}, statusCode={}, available={}", maskedUrl, statusCode,
                            available);
                }
                return available;
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                logger.debug("[Embedding:OPENAI] OpenAI is not available. url={}, error={}", maskedUrl, e.getMessage());
            }
            return false;
        }
    }

    /**
     * Generates embedding vectors for the given texts.
     *
     * <p>Delegated to identically by both {@link #embedDocuments(List)} and
     * {@link #embedQuery(List)}: OpenAI's {@code /v1/embeddings} API has no
     * document/query distinction mechanism (unlike {@code OllamaEmbeddingClient}'s
     * prefixes or {@code GeminiEmbeddingClient}'s {@code task_type}), so there is
     * nothing provider-specific to vary between the two call sites.
     *
     * @param texts the texts to embed, in order
     * @return the list of vectors, one per input text, in the same order
     * @throws EmbeddingException if the provider call fails or returns an unusable response
     */
    @Override
    public List<float[]> embedDocuments(final List<String> texts) {
        return doEmbed(texts);
    }

    /**
     * Generates embedding vectors for the given texts.
     *
     * <p>Delegated to identically by both {@link #embedDocuments(List)} and
     * {@link #embedQuery(List)}: OpenAI's {@code /v1/embeddings} API has no
     * document/query distinction mechanism (unlike {@code OllamaEmbeddingClient}'s
     * prefixes or {@code GeminiEmbeddingClient}'s {@code task_type}), so there is
     * nothing provider-specific to vary between the two call sites.
     *
     * @param texts the texts to embed, in order
     * @return the list of vectors, one per input text, in the same order
     * @throws EmbeddingException if the provider call fails or returns an unusable response
     */
    @Override
    public List<float[]> embedQuery(final List<String> texts) {
        return doEmbed(texts);
    }

    /**
     * Embeds the given texts, transparently splitting them into sub-batches that each respect
     * OpenAI's per-request limits, and concatenating the per-sub-batch results in input order.
     *
     * <p>OpenAI's {@code /v1/embeddings} endpoint rejects a request with a non-retryable
     * {@code 400} when either its input array exceeds {@link #MAX_BATCH_ITEMS} entries or the
     * summed token count across all inputs exceeds {@link #MAX_BATCH_TOKENS_ESTIMATE}. Because
     * callers (e.g. fess core's {@code ChunkVectorHelper}, which flattens every chunk of a batch
     * of documents into one list) can pass a list larger than either limit - including a single
     * very large document whose own chunk list alone exceeds them - this method partitions
     * {@code texts} into contiguous sub-batches under both caps before calling the API, so the
     * split is invisible to callers and never a hard failure.
     *
     * <p>Failure is all-or-nothing: if any sub-batch call fails after its retries are exhausted,
     * an {@link EmbeddingException} propagates and no partial result is returned. Result order is
     * preserved exactly, as callers slice the returned list by contiguous per-document ranges.
     *
     * <p>Shared by {@link #embedDocuments(List)} and {@link #embedQuery(List)}, which are
     * otherwise identical since OpenAI's API has no query/document distinction mechanism.
     *
     * @param texts the texts to embed, in order
     * @return the list of vectors, one per input text, in the same order
     * @throws EmbeddingException if the provider call fails or returns an unusable response
     */
    private List<float[]> doEmbed(final List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Collections.emptyList();
        }
        final List<float[]> vectors = new ArrayList<>(texts.size());
        int start = 0;
        while (start < texts.size()) {
            final int end = nextBatchEnd(texts, start);
            // subList is a view; callEmbedApi only reads it. Any sub-batch failure throws out of
            // this loop, so partially-accumulated vectors are discarded (all-or-nothing).
            vectors.addAll(embedSubBatch(texts, start, end));
            start = end;
        }
        return vectors;
    }

    /**
     * Embeds {@code texts[from, to)} in one call, halving and retrying if the provider rejects it
     * for carrying too many tokens.
     *
     * <p>{@link #estimateTokens(String)} places the boundaries, but it is a character-class
     * heuristic and the real tokenizer is free to disagree - by script, by domain vocabulary, or
     * because OpenAI changed the tokenizer. When it disagrees in the unsafe direction the request
     * fails with a non-retryable {@code 400}, and without this the whole document is marked
     * {@code fail} and never recovers on its own, because every subsequent run rebuilds the same
     * sub-batch and sends the identical payload.
     *
     * <p>Halving is the recovery: each half carries roughly half the tokens, so a single split
     * clears any overshoot below 2x and deeper recursion covers the rest, at a cost of one wasted
     * request per level. Recursion terminates at a single input, which cannot be divided further
     * and is reported as an {@link EmbeddingException} - that case means one chunk alone exceeds
     * the per-request limit, which only a smaller {@code content_chunker.length.chunk_size} fixes.
     *
     * @param texts the full input list.
     * @param from inclusive start index of this sub-batch.
     * @param to exclusive end index of this sub-batch.
     * @return the vectors for {@code texts[from, to)}, in input order.
     * @throws EmbeddingException if the call fails for any reason a smaller batch cannot fix.
     */
    private List<float[]> embedSubBatch(final List<String> texts, final int from, final int to) {
        try {
            return callEmbedApi(texts.subList(from, to));
        } catch (final BatchTooLargeException e) {
            final int size = to - from;
            if (size <= 1) {
                throw new EmbeddingException("OpenAI rejected a single input as too large for one request;"
                        + " reduce content_chunker.length.chunk_size. " + e.getMessage(), e);
            }
            final int mid = from + size / 2;
            logger.warn("[Embedding:OPENAI] Estimated token count was too low for this batch; splitting {} inputs into {}+{}"
                    + " and retrying. detail={}", size, mid - from, to - mid, e.getMessage());
            final List<float[]> split = new ArrayList<>(size);
            split.addAll(embedSubBatch(texts, from, mid));
            split.addAll(embedSubBatch(texts, mid, to));
            return split;
        }
    }

    /**
     * Computes the exclusive end index of the sub-batch beginning at {@code start}, growing it
     * while it stays within both {@link #MAX_BATCH_ITEMS} and {@link #BATCH_TOKEN_BUDGET}. The
     * item at {@code start} is always included even if it alone exceeds a limit, so the batch is
     * never empty and partitioning always terminates (an over-limit lone text is sent as its own
     * sub-batch, which is the most this list-level split can do about it).
     *
     * @param texts the full input list
     * @param start the inclusive start index of this sub-batch
     * @return the exclusive end index ({@code > start})
     */
    private static int nextBatchEnd(final List<String> texts, final int start) {
        final int size = texts.size();
        long tokenSum = estimateTokens(texts.get(start));
        int end = start + 1;
        while (end < size) {
            final int currentItems = end - start;
            final long candidateTokens = tokenSum + estimateTokens(texts.get(end));
            if (currentItems >= MAX_BATCH_ITEMS || candidateTokens > BATCH_TOKEN_BUDGET) {
                break;
            }
            tokenSum = candidateTokens;
            end++;
        }
        return end;
    }

    /**
     * Conservatively approximates the token count of a single input, used only to place
     * sub-batch boundaries - never as an exact tokenizer. Characters that tokenize at CJK
     * density (see {@link #isCjk(int)}) are weighted by
     * {@link #CJK_TOKENS_PER_CHAR_NUMERATOR}/{@link #CJK_TOKENS_PER_CHAR_DENOMINATOR}, while
     * other text uses OpenAI's ~4-characters-per-token rule of thumb; both are rounded up. This
     * over-counts so a sub-batch is split a little early rather than risk exceeding the real
     * {@link #MAX_BATCH_TOKENS_ESTIMATE} limit, whose breach is a non-retryable {@code 400}.
     *
     * <p>Being an approximation, it can still be wrong for a script it was not calibrated
     * against; {@link #doEmbed(List)} treats an over-limit {@code 400} as recoverable and
     * re-splits, so a bad estimate costs an extra round trip rather than the batch.
     *
     * @param text the input text (may be {@code null} or empty)
     * @return the estimated token count, never negative
     */
    private static long estimateTokens(final String text) {
        if (text == null || text.isEmpty()) {
            return 0L;
        }
        long cjk = 0L;
        long other = 0L;
        final int length = text.length();
        for (int i = 0; i < length;) {
            final int codePoint = text.codePointAt(i);
            if (isCjk(codePoint)) {
                cjk++;
            } else {
                other++;
            }
            i += Character.charCount(codePoint);
        }
        final long cjkTokens =
                (cjk * CJK_TOKENS_PER_CHAR_NUMERATOR + CJK_TOKENS_PER_CHAR_DENOMINATOR - 1) / CJK_TOKENS_PER_CHAR_DENOMINATOR;
        return cjkTokens + (other + APPROX_CHARS_PER_TOKEN - 1) / APPROX_CHARS_PER_TOKEN;
    }

    /**
     * Returns whether the code point tokenizes at CJK density - far denser than the
     * ~4-characters-per-token of Latin text. Used only by {@link #estimateTokens(String)}.
     *
     * <p>Script membership alone is not enough. {@link Character.UnicodeScript#of} reports
     * {@code COMMON} for the punctuation that CJK text is actually written with - the ideographic
     * comma and full stop, corner brackets and the ideographic space of
     * {@code U+3000..U+303F}, and the fullwidth forms of {@code U+FF00..U+FFEF} - so a
     * script-only test counts those characters as Latin and estimates them at a quarter token
     * each. Measured on Japanese prose they are about a third of all non-Han/kana characters,
     * which is a large enough share to push a sub-batch over the real limit on its own.
     *
     * @param codePoint the Unicode code point
     * @return true if the code point should be counted at CJK density
     */
    private static boolean isCjk(final int codePoint) {
        // U+1100 (Hangul Jamo) is the lowest code point in any matched block or script, so every
        // Latin/Greek/Cyrillic character exits here instead of paying for the binary search in
        // Character.UnicodeScript.of - which this method would otherwise run once per character of
        // every chunk being sized.
        if (codePoint < 0x1100) {
            return false;
        }
        // CJK Symbols and Punctuation, Kanbun, and Halfwidth/Fullwidth Forms: written as part of
        // CJK text but reported as COMMON script, so they must be matched by block.
        if (codePoint >= 0x3000 && codePoint <= 0x303F || codePoint >= 0x3190 && codePoint <= 0x319F
                || codePoint >= 0xFF00 && codePoint <= 0xFFEF) {
            return true;
        }
        final Character.UnicodeScript script = Character.UnicodeScript.of(codePoint);
        return script == Character.UnicodeScript.HAN || script == Character.UnicodeScript.HIRAGANA
                || script == Character.UnicodeScript.KATAKANA || script == Character.UnicodeScript.HANGUL;
    }

    /**
     * Calls OpenAI's {@code POST /embeddings} endpoint for a single sub-batch of texts already
     * sized to respect the per-request limits by {@link #doEmbed(List)}. Retry, backoff,
     * {@code Retry-After} handling and timeout apply independently to this one call.
     *
     * @param texts the sub-batch of texts to embed, in order
     * @return the list of vectors, one per input text, in the same order
     * @throws EmbeddingException if the provider call fails or returns an unusable response
     */
    private List<float[]> callEmbedApi(final List<String> texts) {
        final String apiUrl = getApiUrl();
        if (CredentialUrlUtil.hasUserInfo(apiUrl)) {
            // Refused before the masked URL is computed, let alone logged: the masking rules do
            // not cover a userinfo credential containing whitespace, so the url={} field of the
            // failure log would otherwise hand that credential back verbatim.
            throw new EmbeddingException(HttpRequestFactory.userInfoRejectedMessage(userInfoConfigKey()));
        }
        final String url = appendPath(apiUrl, "/embeddings");
        final String maskedUrl = maskCredentialInUrl(url);
        final String model = getModel();
        final Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("input", texts);
        requestBody.put("model", model);
        if (supportsDimensionsParam(model)) {
            requestBody.put("dimensions", getDimension());
        }
        final long startTime = System.currentTimeMillis();
        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            return executeWithRetry("embed", () -> {
                final HttpPost httpRequest = HttpRequestFactory.createPost(url, userInfoConfigKey());
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
                                // Reading the error body must never disrupt status-based retry classification below.
                            }
                        }
                        final String errorDetails = extractErrorDetails(errorBody);
                        logger.warn("[Embedding:OPENAI] API error. url={}, statusCode={}, message={}, error={}", maskedUrl, statusCode,
                                response.getReasonPhrase(), errorDetails);
                        if (OpenAiRetry.isRetryableStatus(statusCode)) {
                            final var ra = response.getFirstHeader("Retry-After");
                            final long retryAfter = OpenAiRetry.parseRetryAfterSeconds(ra != null ? ra.getValue() : null);
                            throw new OpenAiRetry.RetryableHttpException(statusCode, response.getReasonPhrase(), retryAfter);
                        }
                        if (isRequestTokenLimitError(statusCode, errorDetails)) {
                            // Recoverable by sending fewer inputs, so it is reported separately
                            // from a flat failure - see doEmbed, which re-splits and retries.
                            throw new BatchTooLargeException(statusCode + " " + response.getReasonPhrase()
                                    + (errorDetails.isEmpty() ? "" : " (" + errorDetails + ")"));
                        }
                        throw new EmbeddingException("OpenAI API error: " + statusCode + " " + response.getReasonPhrase()
                                + (errorDetails.isEmpty() ? "" : " (" + errorDetails + ")"));
                    }
                    final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                    final List<float[]> vectors = parseEmbedResponse(responseBody, texts.size());
                    logger.info("[Embedding:OPENAI] Embed response received. count={}, elapsedTime={}ms", vectors.size(),
                            System.currentTimeMillis() - startTime);
                    return vectors;
                }
            });
        } catch (final EmbeddingException | BatchTooLargeException e) {
            // BatchTooLargeException must reach doEmbed intact: wrapping it here would turn a
            // re-splittable batch into a hard failure.
            throw e;
        } catch (final Exception e) {
            logger.warn("[Embedding:OPENAI] Failed to call OpenAI embed API. url={}, error={}", maskedUrl, e.getMessage(), e);
            throw new EmbeddingException("Failed to call OpenAI embed API", e);
        }
    }

    /**
     * Determines whether the {@code dimensions} truncation parameter should be sent for the given
     * model, honoring the {@code content_chunker.embedding.openai.dimensions.enabled} override.
     *
     * <p>Accepted values (case-insensitive):
     * <ul>
     * <li>{@code auto} (default) - infer from the model name: only the
     * {@code text-embedding-3-*} family documents support for the parameter, and sending it to
     * {@code text-embedding-ada-002} or any other model would error.</li>
     * <li>{@code true} - always send it, whatever the model name is.</li>
     * <li>{@code false} - never send it.</li>
     * </ul>
     *
     * <p>The override exists because the name-based inference is only valid when the {@code model}
     * field really is an OpenAI model id. On Azure OpenAI it is the operator-chosen <em>deployment
     * name</em>, so a deployment called e.g. {@code embeddings-prod} backed by
     * {@code text-embedding-3-small} would fail the prefix test, omit {@code dimensions}, receive a
     * native-length vector and then trip {@link #parseEmbedResponse(String, int)}'s dimension guard
     * on every chunk indefinitely. The inverse case - an OpenAI-compatible gateway that rejects the
     * parameter for a {@code text-embedding-3-*} model - is covered by {@code false}.
     *
     * <p>An unrecognized value is logged and degrades to {@code auto} rather than silently
     * disabling the parameter.
     *
     * @param model the model name (on Azure, the deployment name)
     * @return true if a {@code dimensions} request parameter should be sent
     */
    protected boolean supportsDimensionsParam(final String model) {
        final String mode = getConfigString(CONFIG_DIMENSIONS_ENABLED, DIMENSIONS_ENABLED_AUTO);
        if (DIMENSIONS_ENABLED_TRUE.equalsIgnoreCase(mode)) {
            return true;
        }
        if (DIMENSIONS_ENABLED_FALSE.equalsIgnoreCase(mode)) {
            return false;
        }
        if (!DIMENSIONS_ENABLED_AUTO.equalsIgnoreCase(mode)) {
            logger.warn("[Embedding:OPENAI] Invalid {}.{} value: {}. Using {}.", getConfigPrefix(), CONFIG_DIMENSIONS_ENABLED, mode,
                    DIMENSIONS_ENABLED_AUTO);
        }
        if (StringUtil.isBlank(model)) {
            return false;
        }
        return model.startsWith(DIMENSIONS_SUPPORTED_MODEL_PREFIX);
    }

    /**
     * Parses the {@code /embeddings} response body into a list of vectors, validating
     * that the returned vector count matches {@code expectedCount} and that every
     * vector's length matches {@link #getDimension()}.
     *
     * <p>OpenAI does not guarantee {@code data} is returned in request order; each
     * entry carries an explicit {@code index} field for exactly this reason. Every
     * vector is placed at its own {@code index} position rather than its array
     * position, and every index in {@code [0, expectedCount)} must be covered exactly
     * once, or the response is treated as malformed.
     *
     * @param responseBody the raw JSON response body
     * @param expectedCount the expected number of vectors (= number of input texts)
     * @return the parsed vectors, reordered by each entry's {@code index} field
     * @throws EmbeddingException if the response is malformed or a count/dimension mismatch is detected
     */
    protected List<float[]> parseEmbedResponse(final String responseBody, final int expectedCount) {
        final JsonNode jsonNode;
        try {
            jsonNode = objectMapper.readTree(responseBody);
        } catch (final IOException e) {
            throw new EmbeddingException("Failed to parse OpenAI embed response", e);
        }
        final JsonNode dataNode = jsonNode.path("data");
        if (!dataNode.isArray()) {
            throw new EmbeddingException("OpenAI embed response missing 'data' array");
        }
        if (dataNode.size() != expectedCount) {
            throw new EmbeddingException("OpenAI embed response count mismatch: expected=" + expectedCount + ", actual=" + dataNode.size());
        }
        final int dimension = getDimension();
        final float[][] slots = new float[expectedCount][];
        final boolean[] filled = new boolean[expectedCount];
        for (final JsonNode entry : dataNode) {
            if (!entry.has("index")) {
                throw new EmbeddingException("OpenAI embed response entry missing 'index' field");
            }
            final int index = entry.get("index").asInt(-1);
            if (index < 0 || index >= expectedCount) {
                throw new EmbeddingException("OpenAI embed response index out of range: " + index);
            }
            if (filled[index]) {
                throw new EmbeddingException("OpenAI embed response duplicate index: " + index);
            }
            final JsonNode vectorNode = entry.path("embedding");
            if (!vectorNode.isArray() || vectorNode.size() != dimension) {
                throw new EmbeddingException("OpenAI embed vector dimension mismatch: expected=" + dimension + ", actual="
                        + (vectorNode.isArray() ? vectorNode.size() : -1));
            }
            final float[] vector = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                final JsonNode componentNode = vectorNode.get(i);
                if (componentNode == null || !componentNode.isNumber()) {
                    throw new EmbeddingException("OpenAI embed vector component is not numeric: index=" + index + ", position=" + i);
                }
                // isNumber() is true for a JSON literal like 1e999, which Jackson parses to
                // Double.POSITIVE_INFINITY; guard against non-finite values (Infinity/NaN) so a
                // corrupt vector never reaches the kNN index.
                final float component = (float) componentNode.asDouble();
                if (!Float.isFinite(component)) {
                    throw new EmbeddingException("OpenAI embed vector component is not finite: index=" + index + ", position=" + i);
                }
                vector[i] = component;
            }
            slots[index] = vector;
            filled[index] = true;
        }
        for (int i = 0; i < expectedCount; i++) {
            if (!filled[i]) {
                throw new EmbeddingException("OpenAI embed response missing index: " + i);
            }
        }
        final List<float[]> vectors = new ArrayList<>(expectedCount);
        for (int i = 0; i < expectedCount; i++) {
            vectors.add(slots[i]);
        }
        return vectors;
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

    /**
     * Gets the OpenAI API key.
     *
     * @return the API key
     */
    protected String getApiKey() {
        return getConfigString(CONFIG_API_KEY, "");
    }

    /**
     * Gets the OpenAI API URL.
     *
     * @return the API URL
     */
    protected String getApiUrl() {
        return getConfigString(CONFIG_API_URL, "https://api.openai.com/v1");
    }

    /**
     * Appends a fixed sub-path (leading {@code /}) to a configured base URL, stripping a single
     * trailing {@code /} from the base first so a base such as {@code https://host/v1/} yields
     * {@code https://host/v1/embeddings} rather than {@code https://host/v1//embeddings}. Unlike
     * {@code OllamaEmbeddingClient.normalizeApiUrl}, only one trailing slash is removed, which
     * covers the ordinary misconfiguration without altering intentional path structure.
     *
     * <p>Any query string on the base URL is split off before the sub-path is appended and
     * re-attached afterwards. {@code content_chunker.embedding.openai.api.url} routinely points at
     * non-OpenAI, OpenAI-compatible endpoints (see {@link #maskCredentialInUrl(String)}), and the
     * classic Azure OpenAI form carries the API version as a query parameter
     * ({@code https://host/openai/deployments/d?api-version=2024-02-01}). Blind concatenation would
     * bury the sub-path inside that query value ({@code ...?api-version=2024-02-01/embeddings}),
     * producing a request the gateway rejects.
     *
     * @param baseUrl the configured base URL (may be {@code null}).
     * @param path the sub-path to append, beginning with {@code /}.
     * @return the composed URL, or {@code path} when {@code baseUrl} is {@code null}.
     */
    static String appendPath(final String baseUrl, final String path) {
        if (baseUrl == null) {
            return path;
        }
        final int queryStart = baseUrl.indexOf('?');
        final String base = queryStart < 0 ? baseUrl : baseUrl.substring(0, queryStart);
        final String query = queryStart < 0 ? "" : baseUrl.substring(queryStart);
        final String trimmed = base.endsWith("/") ? base.substring(0, base.length() - 1) : base;
        return trimmed + path + query;
    }

    /**
     * Gets the configured OpenAI embedding model name.
     *
     * @return the model name (default {@code text-embedding-3-small})
     */
    protected String getModel() {
        return getConfigString(CONFIG_MODEL, "text-embedding-3-small");
    }

    @Override
    protected int getTimeout() {
        return getConfigInt("timeout", 120000);
    }

    @Override
    protected String getConfigPrefix() {
        return "content_chunker.embedding.openai";
    }

    /**
     * Returns the maximum number of attempts (initial + retries) for a single HTTP call.
     * Configured via {@code content_chunker.embedding.openai.retry.max} (default {@code 10}).
     *
     * @return the maximum number of HTTP attempts (initial call plus retries).
     */
    protected int getRetryMaxAttempts() {
        return getConfigInt("retry.max", 10);
    }

    /**
     * Returns the base delay in milliseconds for exponential backoff between retries.
     * Configured via {@code content_chunker.embedding.openai.retry.base.delay.ms} (default {@code 2000}).
     *
     * @return the base backoff delay in milliseconds.
     */
    protected long getRetryBaseDelayMs() {
        return getConfigLong(CONFIG_RETRY_BASE_DELAY_MS, 2000L);
    }

    /** Default hard cap on a single backoff sleep, mirroring {@code OllamaEmbeddingClient}'s per-sleep cap. */
    static final long DEFAULT_MAX_BACKOFF_MS = 60_000L;

    /**
     * Returns the hard cap (in milliseconds) on any single backoff sleep, bounding both the
     * exponential-backoff computation and an honored {@code Retry-After} hint. Configured via
     * {@code content_chunker.embedding.openai.retry.max.delay.ms} (default {@code 60000}).
     *
     * <p>Because the {@code ChunkVectorJob} embeds sub-batches sequentially, an uncapped sleep
     * lets a persistently rate-limited endpoint stall indexing for the full
     * {@code (retry.max - 1) x Retry-After} budget (up to ~90 min at defaults). Capping each
     * sleep bounds the worst-case aggregate to about {@code (retry.max - 1) x} this value. A
     * non-positive or unparseable value falls back to {@link #DEFAULT_MAX_BACKOFF_MS}.
     *
     * @return the per-sleep backoff cap in milliseconds (always {@code > 0}).
     */
    protected long getRetryMaxDelayMs() {
        // Unlike a base delay, a cap of 0 (or negative) would clamp every sleep to nothing and
        // silently disable backoff, so this getter keeps a positivity rule getConfigLong does not.
        final long ms = getConfigLong(CONFIG_RETRY_MAX_DELAY_MS, DEFAULT_MAX_BACKOFF_MS);
        if (ms > 0L) {
            return ms;
        }
        logger.warn("[Embedding:OPENAI] {}.{} must be positive: {}. Using default {}.", getConfigPrefix(), CONFIG_RETRY_MAX_DELAY_MS, ms,
                DEFAULT_MAX_BACKOFF_MS);
        return DEFAULT_MAX_BACKOFF_MS;
    }

    /**
     * Masks credentials in a URL before it is logged, covering both credential-bearing query
     * parameters and the authority's userinfo component. See
     * {@link CredentialUrlUtil#maskCredentialInUrl(String)} for the exact rules.
     *
     * <p>OpenAI uses header authentication - the canonical {@code https://api.openai.com}
     * URL does not contain credentials - but {@code content_chunker.embedding.openai.api.url}
     * may point at a gateway (Azure, vLLM, custom) that takes its credential as a query
     * parameter, so all log lines that include a URL route through this helper. The userinfo
     * rule is defensive only: HttpClient rejects a userinfo-bearing request URI outright, and this
     * client now refuses such an {@code api.url} before any call site that masks a URL is reached,
     * so no production path feeds userinfo to the masking rules at all.
     *
     * @param url the URL to mask (may be {@code null}).
     * @return the URL with credential values replaced by {@code ***}, or {@code null} when input is null.
     */
    static String maskCredentialInUrl(final String url) {
        return CredentialUrlUtil.maskCredentialInUrl(url);
    }

    /**
     * Matches the {@code 400} OpenAI returns when a request carries more tokens than
     * {@code /v1/embeddings} accepts across all its inputs - observed as
     * {@code Invalid 'input': maximum request size is 300000 tokens per request.}
     *
     * <p>Deliberately narrow. This is the one {@code 400} that sending fewer inputs can fix, so
     * it must not match the sibling per-input limit
     * ({@code Invalid 'input[0]': maximum input length is 8192 tokens.}), which no amount of
     * re-splitting resolves - a single input is already indivisible here. Matching that one
     * would spend log2(n) doomed requests before failing anyway. Any other {@code 400}
     * (unsupported {@code dimensions}, unknown model, bad key) stays a hard failure.
     *
     * @param statusCode the HTTP status of the failed call.
     * @param errorDetails the rendered error envelope from {@link #extractErrorDetails(String)}.
     * @return true when the batch should be split and retried rather than failed.
     */
    static boolean isRequestTokenLimitError(final int statusCode, final String errorDetails) {
        if (statusCode != 400 || errorDetails == null) {
            return false;
        }
        final String lower = errorDetails.toLowerCase(Locale.ROOT);
        return lower.contains("maximum request size") && lower.contains("token");
    }

    /**
     * Signals the one {@code 400} that a smaller sub-batch can resolve, so {@link #doEmbed(List)}
     * can re-split instead of failing. Never escapes this class: it is either converted into a
     * successful split or rethrown as an {@link EmbeddingException}.
     */
    static final class BatchTooLargeException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        BatchTooLargeException(final String message) {
            super(message);
        }
    }

    /**
     * Executes {@code call} with retry on {@link OpenAiRetry.RetryableHttpException}. {@link IOException}
     * and {@link ParseException} propagate immediately without retry - matches
     * {@code OpenAiLlmClient}'s contract: if the request reached the server, retrying may
     * double-bill. {@link EmbeddingException} (RuntimeException) is not caught here either
     * and propagates immediately.
     *
     * <p>Backoff is exponential ({@code base * 2^(attempt-1)}) with +/-20% jitter, but a
     * server-provided {@code Retry-After} (in seconds, capped at {@link OpenAiRetry#RETRY_AFTER_CAP_SECONDS})
     * takes precedence when present. Either way, each individual sleep is capped at
     * {@link #getRetryMaxDelayMs()} so a persistently-throttling endpoint cannot stall the
     * sequential caller for an unbounded time.
     *
     * @param operation log label (e.g. {@code "embed"}).
     * @param call the HTTP call body.
     * @param <T> the call result type.
     * @return the call result on success.
     * @throws IOException if a non-retryable transport failure occurs or the retry budget is exhausted with a trailing IOException.
     * @throws ParseException if a non-retryable response-parsing failure occurs.
     */
    <T> T executeWithRetry(final String operation, final OpenAiRetry.HttpCall<T> call) throws IOException, ParseException {
        final int maxAttempts = Math.max(1, getRetryMaxAttempts());
        final long baseDelay = Math.max(0L, getRetryBaseDelayMs());
        final long maxDelay = getRetryMaxDelayMs();
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return call.call();
            } catch (final OpenAiRetry.RetryableHttpException e) {
                if (attempt == maxAttempts) {
                    logger.warn("[Embedding:OPENAI] {} retry exhausted. attempts={}, lastStatus={}, retryAfter={}s", operation, attempt,
                            e.statusCode, e.retryAfterSeconds);
                    throw new EmbeddingException("OpenAI API retryable error: " + e.statusCode + " " + e.reason, e);
                }
                sleepBackoff(operation, attempt, maxAttempts, baseDelay, maxDelay, e);
            }
        }
        throw new IllegalStateException("executeWithRetry exited without exception or success");
    }

    /**
     * Sleeps the computed backoff interval. When the {@link OpenAiRetry.RetryableHttpException} carries
     * a positive {@code Retry-After} hint (in seconds), it overrides the
     * exponential-backoff + jitter computation. The resulting sleep is capped at
     * {@code maxDelay}. Restores interrupt status if interrupted.
     *
     * @param operation log label.
     * @param attempt 1-based current attempt index.
     * @param maxAttempts total attempts including the first.
     * @param baseDelay base delay in milliseconds (already clamped to {@code >= 0}).
     * @param maxDelay per-sleep cap in milliseconds (always {@code > 0}).
     * @param cause the {@link OpenAiRetry.RetryableHttpException} that triggered the retry.
     * @throws IOException if the sleep is interrupted.
     */
    private void sleepBackoff(final String operation, final int attempt, final int maxAttempts, final long baseDelay, final long maxDelay,
            final OpenAiRetry.RetryableHttpException cause) throws IOException {
        final long sleepMs = computeBackoffMs(attempt, baseDelay, maxDelay, cause.retryAfterSeconds);
        logger.info("[Embedding:OPENAI] {} retrying. attempt={}/{}, status={}, retryAfter={}s, sleepMs={}", operation, attempt, maxAttempts,
                cause.statusCode, cause.retryAfterSeconds, sleepMs);
        try {
            Thread.sleep(sleepMs);
        } catch (final InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IOException("Retry interrupted", ie);
        }
    }

    /**
     * Computes one retry's backoff sleep in milliseconds, capped at {@code maxDelayMs}. A
     * <em>positive</em> {@code retryAfterSeconds} (server {@code Retry-After} hint) overrides the
     * exponential-backoff + jitter computation. Both the honored {@code Retry-After} and the
     * exponential path are bounded by {@code maxDelayMs} so a persistently-throttling endpoint
     * cannot stall a sequential caller (e.g. {@code ChunkVectorJob}) for an unbounded time.
     *
     * <p>A literal {@code Retry-After: 0} is deliberately <em>not</em> honored: unlike an absent,
     * blank, non-numeric or negative header - all of which {@link OpenAiRetry#parseRetryAfterSeconds(String)}
     * maps to {@code -1} - it would otherwise short-circuit into {@code Thread.sleep(0)}, and with
     * the default {@code retry.max} of 10 that means up to nine back-to-back requests fired at a
     * server that has just rate-limited us. A zero hint therefore falls through to exponential
     * backoff, exactly like the absent case.
     *
     * @param attempt 1-based current attempt index (drives the exponential term).
     * @param baseDelay base delay in milliseconds (already clamped to {@code >= 0}).
     * @param maxDelayMs per-sleep cap in milliseconds (always {@code > 0}).
     * @param retryAfterSeconds honored {@code Retry-After} seconds, or {@code -1} when absent.
     * @return the sleep interval in milliseconds, in {@code [0, maxDelayMs]}.
     */
    static long computeBackoffMs(final int attempt, final long baseDelay, final long maxDelayMs, final long retryAfterSeconds) {
        final long delayMs;
        if (retryAfterSeconds > 0L) {
            delayMs = retryAfterSeconds * 1000L;
        } else {
            final long jitter = (long) (baseDelay * 0.2 * ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
            delayMs = (long) (baseDelay * Math.pow(2, attempt - 1)) + jitter;
        }
        return Math.min(maxDelayMs, Math.max(0L, delayMs));
    }
}
