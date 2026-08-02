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
package org.codelibs.fess.openai.util;

import java.io.IOException;

import org.apache.hc.core5.http.ParseException;

/**
 * Shared retry vocabulary for the OpenAI clients in this plugin: which HTTP statuses are worth
 * retrying, how a {@code Retry-After} header is read, and the signal a call body raises to ask for
 * a retry.
 *
 * <p>The LLM client and the embedding client talk to the same API and therefore observe the same
 * throttling and overload behavior, but each owns its own retry <em>policy</em> (attempt budget,
 * backoff shape, per-sleep cap). Only the vocabulary is shared here, so the two cannot disagree
 * about what a {@code 502} or a {@code Retry-After: 3600} means.
 *
 * @author FessProject
 */
public final class OpenAiRetry {

    /** Maximum seconds we'll honor from a server-provided {@code Retry-After}. */
    public static final long RETRY_AFTER_CAP_SECONDS = 600L;

    private OpenAiRetry() {
        // utility class
    }

    /**
     * Returns whether the given HTTP status code should be retried. Retryable: {@code 429}
     * (rate limit), {@code 500} (server error), {@code 502} (bad gateway - OpenAI returns this
     * under upstream overload), {@code 503} (service unavailable), {@code 504} (gateway timeout).
     * All other statuses propagate immediately.
     *
     * @param statusCode the HTTP status code.
     * @return true when the status is retryable.
     */
    public static boolean isRetryableStatus(final int statusCode) {
        return statusCode == 429 || statusCode == 500 || statusCode == 502 || statusCode == 503 || statusCode == 504;
    }

    /**
     * Parses an HTTP {@code Retry-After} header value as integer seconds. HTTP-date format is
     * intentionally unsupported (returns {@code -1}) so the caller falls back to exponential
     * backoff. Negative or non-numeric values also return {@code -1}. Values exceeding
     * {@link #RETRY_AFTER_CAP_SECONDS} are clamped.
     *
     * @param value the raw {@code Retry-After} header value, or {@code null}.
     * @return the parsed seconds, or {@code -1} when absent/unparseable/negative.
     */
    public static long parseRetryAfterSeconds(final String value) {
        if (value == null) {
            return -1L;
        }
        final String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            return -1L;
        }
        try {
            final long seconds = Long.parseLong(trimmed);
            if (seconds < 0) {
                return -1L;
            }
            return Math.min(seconds, RETRY_AFTER_CAP_SECONDS);
        } catch (final NumberFormatException e) {
            return -1L;
        }
    }

    /**
     * Internal signal raised by an HTTP call body to indicate the received status code is retryable
     * per {@link #isRetryableStatus(int)}. Caught by each client's {@code executeWithRetry}; never
     * escapes the client.
     */
    public static final class RetryableHttpException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        /** The HTTP status code that triggered the retry. */
        public final int statusCode;

        /** The HTTP reason phrase accompanying {@link #statusCode}. */
        public final String reason;

        /** Seconds parsed from {@code Retry-After}, or {@code -1} when absent/unparseable. */
        public final long retryAfterSeconds;

        /**
         * Creates a retry signal.
         *
         * @param statusCode the HTTP status code.
         * @param reason the HTTP reason phrase.
         * @param retryAfterSeconds seconds parsed from {@code Retry-After}, or {@code -1}.
         */
        public RetryableHttpException(final int statusCode, final String reason, final long retryAfterSeconds) {
            super("retryable http error: " + statusCode + " " + reason);
            this.statusCode = statusCode;
            this.reason = reason;
            this.retryAfterSeconds = retryAfterSeconds;
        }
    }

    /**
     * The retryable HTTP call body executed by each client's {@code executeWithRetry}.
     *
     * @param <T> the call result type.
     */
    @FunctionalInterface
    public interface HttpCall<T> {
        /**
         * Performs one attempt.
         *
         * @return the call result.
         * @throws IOException on a transport failure.
         * @throws ParseException on a response-parsing failure.
         */
        T call() throws IOException, ParseException;
    }
}
