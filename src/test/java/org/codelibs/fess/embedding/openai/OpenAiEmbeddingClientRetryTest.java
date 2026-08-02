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
import java.util.concurrent.atomic.AtomicInteger;

import org.codelibs.fess.embedding.EmbeddingException;
import org.codelibs.fess.openai.util.OpenAiRetry;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;

/**
 * Drives {@link OpenAiEmbeddingClient#executeWithRetry} directly, mirroring
 * {@code OpenAiLlmClientRetryTest}'s pattern (subclassing inline, overriding
 * {@code getRetryMaxAttempts()}/{@code getRetryBaseDelayMs()} with a zero base delay to
 * skip real sleeps) rather than a weaker reimplementation.
 */
public class OpenAiEmbeddingClientRetryTest extends UnitFessTestCase {

    /**
     * Pins deviation #2: {@link IOException} must NOT be retried - matches
     * {@code OpenAiLlmClient.executeWithRetry}'s contract ("if the request reached the
     * server, retrying may double-bill"), unlike {@code OllamaEmbeddingClient}, which
     * retries {@link IOException}. A naive port of Ollama's IOException-retry behavior
     * would make this test fail by driving more than one attempt.
     */
    @Test
    public void test_executeWithRetry_doesNotRetryOnIOException() throws Exception {
        final OpenAiEmbeddingClient client = clientWithMaxAttempts(5);

        final AtomicInteger attempts = new AtomicInteger(0);
        try {
            client.executeWithRetry("embed", () -> {
                attempts.incrementAndGet();
                throw new IOException("connect refused");
            });
            fail("expected IOException to propagate without retry");
        } catch (final IOException expected) {
            assertEquals("connect refused", expected.getMessage());
        }
        assertEquals("IOException must not trigger a retry", 1, attempts.get());
    }

    /**
     * Verifies that a retryable HTTP status is retried up to {@code getRetryMaxAttempts()}
     * times and that exhaustion surfaces as {@link EmbeddingException}. Mirrors
     * {@code OpenAiLlmClientRetryTest}'s exhaustion coverage for the ported retry engine.
     */
    @Test
    public void test_executeWithRetry_exhaustsAfterMaxAttempts() throws Exception {
        final OpenAiEmbeddingClient client = clientWithMaxAttempts(3);

        final AtomicInteger attempts = new AtomicInteger(0);
        try {
            client.executeWithRetry("embed", () -> {
                attempts.incrementAndGet();
                throw new OpenAiRetry.RetryableHttpException(500, "boom", -1L);
            });
            fail("expected EmbeddingException after retries are exhausted");
        } catch (final EmbeddingException expected) {
            assertTrue("message should mention retryable error: " + expected.getMessage(), expected.getMessage().contains("retryable"));
        }
        assertEquals("HTTP call should run exactly maxAttempts times", 3, attempts.get());
    }

    /**
     * Verifies that a {@code Retry-After} hint on the {@link OpenAiRetry.RetryableHttpException}
     * still allows a subsequent successful attempt to return normally (the sleep itself is
     * exercised end-to-end via MockWebServer in {@code OpenAiEmbeddingClientTest#test_embed_honorsRetryAfterHeader}).
     */
    @Test
    public void test_executeWithRetry_succeedsAfterRetryableFailure() throws Exception {
        final OpenAiEmbeddingClient client = clientWithMaxAttempts(2);

        final AtomicInteger attempts = new AtomicInteger(0);
        final String result = client.executeWithRetry("embed", () -> {
            if (attempts.incrementAndGet() == 1) {
                throw new OpenAiRetry.RetryableHttpException(503, "unavailable", -1L);
            }
            return "ok";
        });

        assertEquals("ok", result);
        assertEquals(2, attempts.get());
    }

    /**
     * A persistently rate-limited endpoint can return {@code Retry-After: 600} on every attempt;
     * with {@code retry.max=10} that is nine 600s sleeps (~90 min) of stall for the sequential
     * {@code ChunkVectorJob} unless each sleep is capped. Verifies the honored {@code Retry-After}
     * is clamped to the per-sleep cap.
     */
    @Test
    public void test_computeBackoffMs_capsHonoredRetryAfter() {
        // Retry-After: 600s = 600000ms, capped to the 60000ms ceiling.
        assertEquals(60000L, OpenAiEmbeddingClient.computeBackoffMs(1, 2000L, 60000L, 600L));
    }

    /**
     * The exponential term {@code base * 2^(attempt-1)} grows without bound (attempt 10 at
     * base=2000ms is ~1,024,000ms); the per-sleep cap must clamp it even after +/-20% jitter.
     */
    @Test
    public void test_computeBackoffMs_capsExponentialBackoff() {
        // 2000 * 2^9 = 1,024,000ms; jitter is +/-20% of base (400ms), so the result is always
        // far above the 60000ms ceiling and clamps to it deterministically.
        assertEquals(60000L, OpenAiEmbeddingClient.computeBackoffMs(10, 2000L, 60000L, -1L));
    }

    /**
     * A backoff already under the cap is returned unchanged. The {@code Retry-After} path carries
     * no jitter, so this is exact: {@code Retry-After: 1} = 1000ms, below the 60000ms cap.
     */
    @Test
    public void test_computeBackoffMs_belowCapUnchanged() {
        assertEquals(1000L, OpenAiEmbeddingClient.computeBackoffMs(1, 2000L, 60000L, 1L));
    }

    /**
     * {@code parseRetryAfterSeconds} maps absent/blank/non-numeric/negative hints to {@code -1} but
     * a literal {@code Retry-After: 0} to {@code 0}. Honoring that zero verbatim short-circuits the
     * exponential path into {@code Thread.sleep(0)}, so with the default {@code retry.max=10} the
     * client fires up to nine back-to-back requests at a server that just rate-limited it. A zero
     * hint must therefore fall through to exponential backoff, exactly like the absent case.
     */
    @Test
    public void test_computeBackoffMs_zeroRetryAfterFallsBackToExponentialBackoff() {
        // base=2000, attempt=1 => 2000ms +/- 20% jitter (400ms), i.e. always within [1600, 2400]
        // and far below the 60000ms cap. Honoring the zero verbatim would return 0.
        final long backoffMs = OpenAiEmbeddingClient.computeBackoffMs(1, 2000L, 60000L, 0L);
        assertTrue("Retry-After: 0 must not produce a zero-length backoff (was " + backoffMs + "ms)", backoffMs > 0L);
        assertTrue("Retry-After: 0 must fall through to the exponential path (was " + backoffMs + "ms)",
                backoffMs >= 1600L && backoffMs <= 2400L);
    }

    /**
     * {@code parseRetryAfterSeconds} distinguishes a literal {@code 0} (returns {@code 0}) from
     * absent/blank/non-numeric/negative (all {@code -1}); pinning that keeps the zero case a real,
     * separately-reachable input to {@link OpenAiEmbeddingClient#computeBackoffMs} rather than an
     * alias of the absent case.
     */
    @Test
    public void test_parseRetryAfterSeconds_zeroIsDistinctFromAbsent() {
        assertEquals(0L, OpenAiRetry.parseRetryAfterSeconds("0"));
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds(null));
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds("  "));
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds("Wed, 21 Oct 2015 07:28:00 GMT"));
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds("-1"));
    }

    /**
     * The production {@code getRetryMaxDelayMs()} defaults to {@code 60000}ms when the config key
     * is unset, mirroring {@code OllamaEmbeddingClient}'s per-sleep cap.
     */
    @Test
    public void test_getRetryMaxDelayMs_default() {
        assertEquals(60000L, new OpenAiEmbeddingClient().getRetryMaxDelayMs());
    }

    /**
     * Builds a client whose retry budget is {@code maxAttempts} and whose backoff sleeps are
     * zero-length, so a retry test exercises the attempt accounting without real delay.
     *
     * @param maxAttempts the attempt budget (initial call plus retries).
     * @return the configured client.
     */
    private static OpenAiEmbeddingClient clientWithMaxAttempts(final int maxAttempts) {
        return new OpenAiEmbeddingClient() {
            @Override
            protected int getRetryMaxAttempts() {
                return maxAttempts;
            }

            @Override
            protected long getRetryBaseDelayMs() {
                return 0L; // skip real backoff sleep
            }
        };
    }

}
