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

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.codelibs.fess.embedding.EmbeddingException;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.codelibs.fess.util.ComponentUtil;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

public class OpenAiEmbeddingClientTest extends UnitFessTestCase {

    /** The real config key read by the production (non-overridden) {@link OpenAiEmbeddingClient#getDimension()}. */
    private static final String DIMENSION_CONFIG_KEY = "content_chunker.embedding.dimension";

    private TestableOpenAiEmbeddingClient client;
    private MockWebServer mockServer;

    /**
     * This test class mutates the {@code systemProperties} component (see the
     * {@code content_chunker.embedding.openai.*}-channel tests below and the pre-existing
     * {@code test_getDimension_*} tests), which UTFlute otherwise caches and shares as a
     * JVM-lifetime singleton across every test class using the same {@code test_app.xml}
     * config file. Overriding this to {@code true} destroys and reinitializes that container
     * around <em>every</em> test method in this class, in both directions: before the test
     * runs ({@code LastaDiTestCase#xdoPrepareTestCaseContainer}, which destroys any existing
     * container first whenever this returns {@code true}), so this class's tests never inherit
     * residue left by an earlier test or class; and after the test finishes
     * ({@code InjectionTestCase#xdestroyTestCaseContainer}, which likewise destroys the
     * container and clears the cached config-file marker whenever this returns {@code true}),
     * so a later class cannot recycle the container this class mutated. The {@code finally}
     * blocks below are still good practice, but this override - not them - is what keeps a
     * mutation here from corrupting another test class order-dependently while the suite stays
     * green.
     */
    @Override
    protected boolean isUseOneTimeContainer() {
        return true;
    }

    @Override
    public void setUp(final TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableOpenAiEmbeddingClient();
        mockServer = new MockWebServer();
        mockServer.start();
    }

    @Override
    public void tearDown(final TestInfo testInfo) throws Exception {
        if (client != null) {
            client.destroy();
        }
        if (mockServer != null) {
            mockServer.shutdown();
        }
        super.tearDown(testInfo);
    }

    private void setupClientForMockServer() {
        final String baseUrl = mockServer.url("").toString();
        final String apiUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        client.setTestApiUrl(apiUrl);
        client.setTestApiKey("sk-test-key");
        client.setTestDimension(3);
        client.setTestTimeout(30000);
        client.init();
    }

    @Test
    public void test_getName() {
        assertEquals("openai", client.getName());
    }

    // ========== embedDocuments() / embedQuery() ==========

    @Test
    public void test_embedDocuments_success() throws Exception {
        final String responseJson = """
                {
                  "object": "list",
                  "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
                  ],
                  "model": "text-embedding-3-small",
                  "usage": {"prompt_tokens": 4, "total_tokens": 4}
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<float[]> result = client.embedDocuments(List.of("chunk one", "chunk two"));

        assertEquals(2, result.size());
        assertEquals(3, result.get(0).length);
        assertEquals(0.1f, result.get(0)[0]);
        assertEquals(0.2f, result.get(0)[1]);
        assertEquals(0.6f, result.get(1)[2]);

        final RecordedRequest recordedRequest = mockServer.takeRequest();
        assertEquals("POST", recordedRequest.getMethod());
        assertTrue(recordedRequest.getPath().endsWith("/embeddings"), "unexpected path: " + recordedRequest.getPath());
        assertEquals("Bearer sk-test-key", recordedRequest.getHeader("Authorization"));
        final String body = recordedRequest.getBody().readUtf8();
        assertTrue(body.contains("\"input\""), "request body should carry 'input': " + body);
        assertTrue(body.contains("chunk one") && body.contains("chunk two"), "request body should carry both inputs: " + body);
        assertTrue(body.contains("\"model\":\"text-embedding-3-small\""), "request body should carry the model name: " + body);
        assertTrue(body.contains("\"dimensions\":3"), "request body should carry the configured dimension: " + body);
    }

    @Test
    public void test_embedDocuments_reordersByIndex() throws Exception {
        // data[] is deliberately listed out of index order: index 1 first, index 0 second.
        // A naive implementation that trusted array position over the "index" field would
        // return chunk two's vector at position 0 and fail this assertion.
        final String responseJson = """
                {
                  "object": "list",
                  "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
                  ],
                  "model": "text-embedding-3-small"
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<float[]> result = client.embedDocuments(List.of("chunk one", "chunk two"));

        assertEquals(2, result.size());
        assertEquals(0.1f, result.get(0)[0]);
        assertEquals(0.2f, result.get(0)[1]);
        assertEquals(0.3f, result.get(0)[2]);
        assertEquals(0.4f, result.get(1)[0]);
        assertEquals(0.5f, result.get(1)[1]);
        assertEquals(0.6f, result.get(1)[2]);
    }

    @Test
    public void test_embedDocuments_dimensionsParamOmittedForAda002() throws Exception {
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        client.setTestModel("text-embedding-ada-002");
        setupClientForMockServer();

        client.embedDocuments(List.of("chunk"));

        final RecordedRequest recordedRequest = mockServer.takeRequest();
        final String body = recordedRequest.getBody().readUtf8();
        assertFalse(body.contains("\"dimensions\""), "dimensions must be omitted for ada-002: " + body);
    }

    @Test
    public void test_embedDocuments_dimensionsParamIncludedForTextEmbedding3Large() throws Exception {
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        client.setTestModel("text-embedding-3-large");
        setupClientForMockServer();

        client.embedDocuments(List.of("chunk"));

        final RecordedRequest recordedRequest = mockServer.takeRequest();
        final String body = recordedRequest.getBody().readUtf8();
        assertTrue(body.contains("\"dimensions\":3"), "dimensions must be included for text-embedding-3-large: " + body);
    }

    @Test
    public void test_embedDocuments_emptyInput_returnsEmptyList() throws Exception {
        setupClientForMockServer();

        assertEquals(0, client.embedDocuments(null).size());
        assertEquals(0, client.embedDocuments(List.of()).size());
        assertEquals("no HTTP call should be made for empty input", 0, mockServer.getRequestCount());
    }

    @Test
    public void test_embedDocuments_dimensionMismatch_throws() throws Exception {
        // Server returns 2-dim vectors but the configured dimension is 3.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException on dimension mismatch");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("dimension mismatch"), "message should mention dimension mismatch: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_nonNumericVectorComponent_throws() throws Exception {
        // The "embedding" array's second element is a JSON null instead of a number.
        // A naive Jackson asDouble() call would silently coerce this to 0.0 and corrupt
        // the stored vector instead of surfacing a clear error.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,null,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException on non-numeric vector component");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("not numeric"), "message should mention non-numeric component: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_nonFiniteVectorComponent_throws() throws Exception {
        // The "embedding" array's second element is the JSON literal 1e999, which exceeds the
        // double range and Jackson parses to Double.POSITIVE_INFINITY. isNumber() is still true,
        // so without a finiteness guard a non-finite value would be stored into the vector and
        // later poison the kNN index.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,1e999,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException on non-finite vector component");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("not finite"), "message should mention non-finite component: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_countMismatch_throws() throws Exception {
        // Only 1 entry returned for 2 input texts.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk one", "chunk two"));
            fail("expected EmbeddingException on count mismatch");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("count mismatch"), "message should mention count mismatch: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_dataNotArray_throws() throws Exception {
        // "data" is a JSON object, not an array. Must fail the isArray() guard before the
        // count check even looks at it.
        final String responseJson = "{\"object\":\"list\",\"data\":{}}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException when 'data' is not an array");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("'data' array"), "message should mention missing 'data' array: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_duplicateIndex_throws() throws Exception {
        // Two entries, both index 0, for two input texts: size matches expectedCount so the
        // count guard passes, but the second entry collides on an already-filled slot.
        final String responseJson =
                "{\"object\":\"list\",\"data\":[" + "{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]},"
                        + "{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.4,0.5,0.6]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk one", "chunk two"));
            fail("expected EmbeddingException on duplicate index");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("duplicate index"), "message should mention duplicate index: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_indexOutOfRange_throws() throws Exception {
        // Single entry for a single input text (count guard passes), but its index (5) is
        // outside [0, expectedCount).
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":5,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException on out-of-range index");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("out of range"), "message should mention out of range: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_entryMissingIndexField_throws() throws Exception {
        // Entry omits the "index" field entirely.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException when an entry has no 'index' field");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("'index' field"), "message should mention missing 'index' field: " + e.getMessage());
        }
    }

    // Note: the "missing index" gap guard (a slot in [0, expectedCount) never filled) is
    // unreachable via any response body: the count guard forces data.size()==expectedCount,
    // and with the out-of-range and duplicate guards, expectedCount entries must fill
    // expectedCount distinct in-range slots (a bijection by pigeonhole), so no gap can
    // survive to that final loop. It remains as defensive code and has no reachable test.

    @Test
    public void test_embedDocuments_nonJsonResponse_throws() throws Exception {
        // A 200 response whose body is not JSON (e.g. an HTML gateway page) must surface as
        // an EmbeddingException from the readTree() IOException catch, not a silent failure.
        mockServer.enqueue(new MockResponse().setBody("this is not valid json").addHeader("Content-Type", "text/plain"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException on non-JSON response body");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("Failed to parse"), "message should mention parse failure: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_errorBodyDetailSurfacedInException() throws Exception {
        // A non-2xx response carrying OpenAI's {"error":{...}} envelope: the parsed type/code/
        // message must reach both the log and the thrown exception (Finding 1). 400 is not
        // retryable, so exactly one HTTP call is made.
        final String errorBody = "{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"model_not_found\","
                + "\"message\":\"The model does not exist\"}}";
        mockServer.enqueue(new MockResponse().setResponseCode(400).setBody(errorBody).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException on 400");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("model_not_found"), "exception should surface the parsed error code: " + e.getMessage());
            assertTrue(e.getMessage().contains("The model does not exist"),
                    "exception should surface the parsed error message: " + e.getMessage());
        }
        assertEquals("400 must not be retried", 1, mockServer.getRequestCount());
    }

    @Test
    public void test_embedDocuments_doesNotRetryOn400() throws Exception {
        mockServer.enqueue(new MockResponse().setResponseCode(400).setBody("bad request"));
        client.setTestRetryMax(5);
        client.setTestRetryBaseDelayMs(1L);
        setupClientForMockServer();

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException");
        } catch (final EmbeddingException e) {
            // expected
        }
        assertEquals("400 must not be retried", 1, mockServer.getRequestCount());
    }

    @Test
    public void test_embedDocuments_retriesOn503() throws Exception {
        mockServer.enqueue(new MockResponse().setResponseCode(503));
        mockServer.enqueue(new MockResponse().setResponseCode(503));
        final String successBody = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(successBody).addHeader("Content-Type", "application/json"));
        client.setTestRetryBaseDelayMs(0L);
        setupClientForMockServer();

        final List<float[]> result = client.embedDocuments(List.of("chunk"));

        assertEquals(1, result.size());
        assertEquals(3, mockServer.getRequestCount());
    }

    @Test
    public void test_embedDocuments_honorsRetryAfterHeader() throws Exception {
        // Pins deviation #1: a server-provided Retry-After header must be honored even
        // though OllamaEmbeddingClient has no concept of it at all.
        //
        // requestCount==2 alone does NOT discriminate "Retry-After honored" from "Retry-After
        // ignored, plain exponential backoff kicked in instead" - both paths retry exactly
        // once and succeed. To actually discriminate, the base backoff delay is set large
        // (10s) so the two paths diverge sharply in wall-clock time: honoring Retry-After: 1
        // sleeps ~1s, while falling through to exponential backoff would sleep ~8-12s (base
        // * 2^0 +/- 20% jitter). The elapsed-time assertion below fails if that fallback ever
        // happens. Mirrors OpenAiLlmClientTest#test_chat_honorsRetryAfterHeader_429's
        // huge-backoff-to-detect-override pattern.
        client.setTestRetryBaseDelayMs(10000L);
        mockServer.enqueue(new MockResponse().setResponseCode(429).setHeader("Retry-After", "1"));
        final String successBody = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(successBody).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final long start = System.currentTimeMillis();
        final List<float[]> result = client.embedDocuments(List.of("chunk"));
        final long elapsedMs = System.currentTimeMillis() - start;

        assertEquals(1, result.size());
        assertEquals("retry must have happened after honoring Retry-After", 2, mockServer.getRequestCount());
        assertTrue("Retry-After: 1 must override the 10s backoff base (elapsedMs=" + elapsedMs + ")", elapsedMs < 5000);
    }

    @Test
    public void test_embedQuery_success() throws Exception {
        // Exercises embedQuery() end-to-end at least once; embedDocuments() above already
        // covers the shared request/response handling in depth.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<float[]> result = client.embedQuery(List.of("search this"));

        assertEquals(1, result.size());
        assertEquals(3, result.get(0).length);
        assertEquals(0.1f, result.get(0)[0]);

        final RecordedRequest recordedRequest = mockServer.takeRequest();
        assertTrue(recordedRequest.getPath().endsWith("/embeddings"), "unexpected path: " + recordedRequest.getPath());
        final String body = recordedRequest.getBody().readUtf8();
        assertTrue(body.contains("search this"), "request body should carry the query text: " + body);
    }

    @Test
    public void test_embedDocuments_and_embedQuery_produceIdenticalRequestBodies() throws Exception {
        // Documents/verifies the deliberate design decision (see class Javadoc): OpenAI's
        // embeddings API has no query/document distinction mechanism, so embedDocuments()
        // and embedQuery() must send byte-for-byte identical request bodies for the same
        // input texts - unlike OllamaEmbeddingClient (prefixes) or GeminiEmbeddingClient
        // (task_type), whose two methods would diverge here.
        final String responseJson =
                "{\"object\":\"list\",\"data\":[" + "{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]},"
                        + "{\"object\":\"embedding\",\"index\":1,\"embedding\":[0.4,0.5,0.6]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> texts = List.of("chunk one", "chunk two");
        client.embedDocuments(texts);
        client.embedQuery(texts);

        final String documentsBody = mockServer.takeRequest().getBody().readUtf8();
        final String queryBody = mockServer.takeRequest().getBody().readUtf8();
        assertEquals("embedDocuments() and embedQuery() must send identical request bodies", documentsBody, queryBody);
    }

    // ========== sub-batching against OpenAI per-request limits ==========

    /** Shared mapper for asserting on recorded request bodies. */
    private static final ObjectMapper TEST_MAPPER = new ObjectMapper();

    /**
     * Builds an {@code /embeddings} response body of {@code count} entries whose {@code index}
     * fields are 0-based within this (sub-batch) response, exactly as OpenAI returns per call.
     * Each vector's first component encodes the caller-supplied {@code globalOffset + i} so the
     * test can verify cross-sub-batch reassembly order. Dimension is 3 (matching test setup).
     */
    private static String buildEmbedResponse(final int count, final int globalOffset) {
        final StringBuilder sb = new StringBuilder("{\"object\":\"list\",\"data\":[");
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append("{\"object\":\"embedding\",\"index\":")
                    .append(i)
                    .append(",\"embedding\":[")
                    .append(globalOffset + i)
                    .append(".0,0.0,0.0]}");
        }
        return sb.append("]}").toString();
    }

    /** Returns the number of entries in the {@code input} array of a recorded request body. */
    private static int countInputs(final RecordedRequest req) throws Exception {
        final JsonNode body = TEST_MAPPER.readTree(req.getBody().readUtf8());
        return body.path("input").size();
    }

    @Test
    public void test_embedDocuments_splitsBatchExceedingMaxItems() throws Exception {
        // 2500 short texts exceed OpenAI's 2048-item array cap, so the client must split into
        // two calls of 2048 + 452 items. The per-sub-batch response sizes (2048, 452) also pin
        // the split boundaries: a differently-sized split would trip parseEmbedResponse's
        // count guard on one of the sub-batch responses.
        final int total = 2500;
        mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(2048, 0)).addHeader("Content-Type", "application/json"));
        mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(452, 2048)).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> texts = new ArrayList<>(total);
        for (int i = 0; i < total; i++) {
            texts.add("t" + i);
        }
        final List<float[]> result = client.embedDocuments(texts);

        assertEquals(total, result.size());
        assertEquals("input over 2048 items must be split into exactly two sub-batch calls", 2, mockServer.getRequestCount());
        final RecordedRequest r1 = mockServer.takeRequest();
        final RecordedRequest r2 = mockServer.takeRequest();
        assertEquals("first sub-batch must be capped at 2048 items", 2048, countInputs(r1));
        assertEquals("second sub-batch carries the remainder", 452, countInputs(r2));
        for (int i = 0; i < total; i++) {
            assertEquals("reassembled vector order must match input order at " + i, (float) i, result.get(i)[0]);
        }
    }

    @Test
    public void test_embedDocuments_splitsBatchExceedingTokenBudget() throws Exception {
        // Only 3 texts (far under the 2048-item cap), but each is 160,000 CJK characters. CJK
        // tokenizes at ~1 token/char, so each text alone is ~160,000 estimated tokens and any
        // two together exceed the ~300,000-token per-request budget -- forcing one item per
        // sub-batch, i.e. three calls. This exercises the CJK-aware estimate: had CJK been
        // mis-counted at 4 chars/token, all three would fit one call and the enqueued 1-entry
        // responses would trip a count mismatch. (This test proves the split fires on the
        // token budget; it does not by itself certify the exact divisor -- that is a
        // deliberate, conservative reasoning choice in estimateTokens.)
        final String cjkText = "あ".repeat(160_000);
        for (int i = 0; i < 3; i++) {
            mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(1, i)).addHeader("Content-Type", "application/json"));
        }
        setupClientForMockServer();

        final List<float[]> result = client.embedDocuments(List.of(cjkText, cjkText, cjkText));

        assertEquals(3, result.size());
        assertEquals("token budget must force one item per sub-batch even under the item cap", 3, mockServer.getRequestCount());
        assertEquals("first sub-batch must carry a single item", 1, countInputs(mockServer.takeRequest()));
        assertEquals("second sub-batch must carry a single item", 1, countInputs(mockServer.takeRequest()));
        assertEquals("third sub-batch must carry a single item", 1, countInputs(mockServer.takeRequest()));
        assertEquals(0.0f, result.get(0)[0]);
        assertEquals(1.0f, result.get(1)[0]);
        assertEquals(2.0f, result.get(2)[0]);
    }

    @Test
    public void test_estimate_countsCjkPunctuationAtCjkDensity() throws Exception {
        // Character.UnicodeScript reports COMMON for the ideographic comma/full stop and the
        // corner brackets that Japanese is actually written with, so a script-only test rates
        // them at 4 chars/token. Measured on Japanese prose they are about a third of all
        // non-Han/kana characters -- enough to push a full sub-batch past the real limit on
        // their own. Two texts of 120,000 such characters must therefore split into two calls;
        // at a quarter token each they would total 60,000 tokens and fit in one.
        final String punctuation = "、。「」".repeat(30_000);
        for (int i = 0; i < 2; i++) {
            mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(1, i)).addHeader("Content-Type", "application/json"));
        }
        setupClientForMockServer();

        final List<float[]> result = client.embedDocuments(List.of(punctuation, punctuation));

        assertEquals(2, result.size());
        assertEquals("CJK punctuation must be counted at CJK density, forcing a split", 2, mockServer.getRequestCount());
    }

    @Test
    public void test_isRequestTokenLimitError_matchesOnlyTheSplittable400() throws Exception {
        // The per-request total is the one 400 a smaller batch can fix.
        assertTrue(OpenAiEmbeddingClient.isRequestTokenLimitError(400, "type=invalid_request_error,code=null,param=null,"
                + "message=Invalid 'input': maximum request size is 300000 tokens per request."));
        // The per-input ceiling is NOT splittable -- a single input is already indivisible, so
        // matching it would spend log2(n) doomed requests before failing anyway.
        assertFalse(OpenAiEmbeddingClient.isRequestTokenLimitError(400,
                "type=invalid_request_error,code=null,param=null," + "message=Invalid 'input[0]': maximum input length is 8192 tokens."));
        // Unrelated 400s stay hard failures.
        assertFalse(OpenAiEmbeddingClient.isRequestTokenLimitError(400,
                "type=invalid_request_error,message=This model does not support specifying dimensions."));
        assertFalse(OpenAiEmbeddingClient.isRequestTokenLimitError(429, "maximum request size is 300000 tokens per request"));
        assertFalse(OpenAiEmbeddingClient.isRequestTokenLimitError(400, null));
    }

    @Test
    public void test_embedDocuments_halvesAndRetriesWhenEstimateWasTooLow() throws Exception {
        // estimateTokens is a character-class heuristic; the real tokenizer may disagree. When
        // it does, the provider answers a non-retryable 400 and -- without the split-and-retry
        // -- the document is marked fail forever, because every later run rebuilds the same
        // sub-batch and sends the identical payload. Here 4 short texts fit the estimate in one
        // call, the server rejects that call as over the token limit, and the client must
        // recover by halving into 2 + 2 rather than failing.
        mockServer.enqueue(new MockResponse().setResponseCode(400)
                .setBody("{\"error\":{\"message\":\"Invalid 'input': maximum request size is 300000 tokens per request.\"}}"));
        mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(2, 0)).addHeader("Content-Type", "application/json"));
        mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(2, 2)).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();
        client.setTestRetryBaseDelayMs(0L);

        final List<float[]> result = client.embedDocuments(List.of("a", "b", "c", "d"));

        assertEquals(4, result.size());
        assertEquals("one rejected call plus the two halves", 3, mockServer.getRequestCount());
        assertEquals("the rejected call carried all four inputs", 4, countInputs(mockServer.takeRequest()));
        assertEquals("first half", 2, countInputs(mockServer.takeRequest()));
        assertEquals("second half", 2, countInputs(mockServer.takeRequest()));
        // Order must survive the split: the halves are concatenated back in input order.
        for (int i = 0; i < 4; i++) {
            assertEquals("split must preserve input order at " + i, (float) i, result.get(i)[0]);
        }
    }

    @Test
    public void test_embedDocuments_singleOverLimitInputFailsWithChunkSizeHint() throws Exception {
        // Recursion bottoms out at one input, which cannot be divided further. That case means
        // a single chunk exceeds the per-request limit, which only a smaller chunk size fixes,
        // so the failure must say so instead of looping or reporting a bare 400.
        mockServer.enqueue(new MockResponse().setResponseCode(400)
                .setBody("{\"error\":{\"message\":\"Invalid 'input': maximum request size is 300000 tokens per request.\"}}"));
        setupClientForMockServer();
        client.setTestRetryBaseDelayMs(0L);

        try {
            client.embedDocuments(List.of("one huge chunk"));
            fail("expected an indivisible over-limit input to fail");
        } catch (final EmbeddingException expected) {
            assertTrue("message must point at the chunk size: " + expected.getMessage(),
                    expected.getMessage().contains("content_chunker.length.chunk_size"));
        }
        assertEquals("an indivisible input must not be retried", 1, mockServer.getRequestCount());
    }

    @Test
    public void test_embedDocuments_normalBatchIssuesSingleCall() throws Exception {
        // A batch comfortably under both limits (100 short texts) must not be split: exactly
        // one HTTP call, guarding against a regression that fragments the common case.
        final int total = 100;
        mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(total, 0)).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> texts = new ArrayList<>(total);
        for (int i = 0; i < total; i++) {
            texts.add("chunk " + i);
        }
        final List<float[]> result = client.embedDocuments(texts);

        assertEquals(total, result.size());
        assertEquals("a batch under both limits must issue exactly one call", 1, mockServer.getRequestCount());
        assertEquals("the single call must carry all inputs", total, countInputs(mockServer.takeRequest()));
    }

    @Test
    public void test_embedDocuments_nonFirstSubBatchFailurePropagates() throws Exception {
        // 2500 texts split into 2048 + 452. The first sub-batch succeeds; the second returns a
        // non-retryable 400. The whole call must fail with EmbeddingException and return no
        // partial result -- fess core's ChunkVectorHelper treats any throw as a whole-batch
        // failure, so leaking the first sub-batch's vectors would corrupt its index-range slicing.
        mockServer.enqueue(new MockResponse().setBody(buildEmbedResponse(2048, 0)).addHeader("Content-Type", "application/json"));
        mockServer.enqueue(new MockResponse().setResponseCode(400).setBody("{\"error\":{\"message\":\"too many inputs\"}}"));
        client.setTestRetryBaseDelayMs(0L);
        setupClientForMockServer();

        final List<String> texts = new ArrayList<>(2500);
        for (int i = 0; i < 2500; i++) {
            texts.add("t" + i);
        }
        try {
            client.embedDocuments(texts);
            fail("expected EmbeddingException when a non-first sub-batch fails");
        } catch (final EmbeddingException e) {
            // expected: no partial result is returned
        }
        assertEquals("both sub-batches attempted, second failed without retry", 2, mockServer.getRequestCount());
    }

    // ========== getDimension() (real, non-overridden) ==========
    //
    // The embedDocuments()/embedQuery()-driven tests above exercise TestableOpenAiEmbeddingClient's own
    // hand-written getDimension() override, never the production method. These tests
    // use a plain `new OpenAiEmbeddingClient()` (no subclass) to drive the real
    // ComponentUtil.getFessConfig().getSystemProperty("content_chunker.embedding.dimension", ...)
    // config-read seam directly.

    @Test
    public void test_getDimension_configured() {
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "1536");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            assertEquals(1536, realClient.getDimension());
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_notConfigured_throws() {
        ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException when dimension is unconfigured");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("not configured"), "message should mention not configured: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_zero_throws() {
        // A zero dimension is a misconfiguration: it would make the response's per-vector
        // dimension guard demand an empty embedding and would send "dimensions":0 to OpenAI
        // (a non-retryable 400). Reject it up front, network-call-free.
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "0");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException when dimension is zero");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("must be positive"), "message should mention must be positive: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_negative_throws() {
        // A negative dimension would blow up as a NegativeArraySizeException deep inside
        // parseEmbedResponse; reject it early with a clear message instead.
        //
        // The configured value is deliberately "-05" rather than "-5": the "not configured" and
        // "Invalid ... value" branches both echo the raw configured string, and this branch must
        // too (matching OllamaEmbeddingClient / GeminiEmbeddingClient / OpenSearchEmbeddingClient),
        // so that an operator greps the message and finds what they actually typed. Reporting the
        // *parsed* int instead would render "-5" here, which "-05" discriminates.
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "-05");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException when dimension is negative");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("must be positive"), "message should mention must be positive: " + e.getMessage());
                assertTrue(e.getMessage().endsWith("must be positive: -05"),
                        "message should echo the raw configured value, not the parsed int: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    // ========== content_chunker.embedding.openai.* channel (getConfigString / getSystemProperty) ==========
    //
    // Like the getDimension() tests above, these drive the real, non-overridden production
    // accessors on a plain `new OpenAiEmbeddingClient()` through the actual
    // ComponentUtil.getSystemProperties() seam wired by test_app.xml as the "systemProperties"
    // component (see test_app.xml). This is deliberately different from both
    // test_getConfigString_returnsDefaultWhenUnset (which only proves the hardcoded default is
    // returned when nothing is configured anywhere - channel-blind by construction) and every
    // embedDocuments()/embedQuery() test that uses the `client` field
    // (TestableOpenAiEmbeddingClient overrides getApiKey()/getApiUrl()/getModel()/
    // getRetryBaseDelayMs() directly, and overrides getConfigString() itself - which
    // supportsDimensionsParam() and getRetryMaxDelayMs() go through, since neither is
    // overridden - to read a test-local Map, never touching ComponentUtil at all). A test
    // that only ever observes an overridden getter, or
    // only ever observes the hardcoded default, cannot tell getSystemProperty
    // (conf/system.properties) apart from getOrDefault (fess_config.properties).
    // These tests set a real value on the systemProperties channel and assert the production
    // getter observes it, so a regression back to getOrDefault fails them (verified below by
    // temporarily reverting each corresponding production read and confirming its test goes red).
    // All six of getApiKey()/getApiUrl()/getModel()/supportsDimensionsParam()/
    // getRetryBaseDelayMs()/getRetryMaxDelayMs() - i.e. every converted read - are covered here.

    private static final String API_KEY_CONFIG_KEY = "content_chunker.embedding.openai.api.key";
    private static final String API_URL_CONFIG_KEY = "content_chunker.embedding.openai.api.url";
    private static final String MODEL_CONFIG_KEY = "content_chunker.embedding.openai.model";
    private static final String DIMENSIONS_ENABLED_CONFIG_KEY = "content_chunker.embedding.openai.dimensions.enabled";
    private static final String RETRY_BASE_DELAY_MS_CONFIG_KEY = "content_chunker.embedding.openai.retry.base.delay.ms";
    private static final String RETRY_MAX_DELAY_MS_CONFIG_KEY = "content_chunker.embedding.openai.retry.max.delay.ms";

    @Test
    public void test_getApiKey_readsFromSystemProperties() {
        ComponentUtil.getSystemProperties().setProperty(API_KEY_CONFIG_KEY, "sk-real-secret");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            assertEquals("sk-real-secret", realClient.getApiKey());
        } finally {
            ComponentUtil.getSystemProperties().remove(API_KEY_CONFIG_KEY);
        }
    }

    @Test
    public void test_getApiUrl_readsFromSystemProperties() {
        ComponentUtil.getSystemProperties().setProperty(API_URL_CONFIG_KEY, "https://gw.example.com/v1");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            assertEquals("https://gw.example.com/v1", realClient.getApiUrl());
        } finally {
            ComponentUtil.getSystemProperties().remove(API_URL_CONFIG_KEY);
        }
    }

    @Test
    public void test_getModel_readsFromSystemProperties() {
        ComponentUtil.getSystemProperties().setProperty(MODEL_CONFIG_KEY, "text-embedding-3-large");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            assertEquals("text-embedding-3-large", realClient.getModel());
        } finally {
            ComponentUtil.getSystemProperties().remove(MODEL_CONFIG_KEY);
        }
    }

    @Test
    public void test_supportsDimensionsParam_readsFromSystemProperties() {
        // Unset (default "auto") would infer true for a text-embedding-3-* model; forcing
        // "false" on the systemProperties channel must flip the result. supportsDimensionsParam()
        // itself - unlike getConfigString(), which it calls internally and which is declared on
        // fess-core's AbstractEmbeddingClient in a different package - is declared directly on
        // OpenAiEmbeddingClient (same package as this test), so it can be invoked here without a
        // probe subclass; the virtual call from inside it still reaches the real, inherited
        // getConfigString().
        ComponentUtil.getSystemProperties().setProperty(DIMENSIONS_ENABLED_CONFIG_KEY, "false");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            assertFalse(realClient.supportsDimensionsParam("text-embedding-3-large"));
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSIONS_ENABLED_CONFIG_KEY);
        }
    }

    @Test
    public void test_getRetryBaseDelayMs_readsFromSystemProperties() {
        ComponentUtil.getSystemProperties().setProperty(RETRY_BASE_DELAY_MS_CONFIG_KEY, "12345");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            assertEquals(12345L, realClient.getRetryBaseDelayMs());
        } finally {
            ComponentUtil.getSystemProperties().remove(RETRY_BASE_DELAY_MS_CONFIG_KEY);
        }
    }

    @Test
    public void test_getRetryMaxDelayMs_readsFromSystemProperties() {
        ComponentUtil.getSystemProperties().setProperty(RETRY_MAX_DELAY_MS_CONFIG_KEY, "98765");
        try {
            final OpenAiEmbeddingClient realClient = new OpenAiEmbeddingClient();
            assertEquals(98765L, realClient.getRetryMaxDelayMs());
        } finally {
            ComponentUtil.getSystemProperties().remove(RETRY_MAX_DELAY_MS_CONFIG_KEY);
        }
    }

    // ========== checkAvailabilityNow() / isAvailable() ==========

    @Test
    public void test_checkAvailabilityNow_success() throws Exception {
        mockServer.enqueue(new MockResponse().setBody("{\"data\":[]}").addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        assertTrue(client.isAvailable());
    }

    @Test
    public void test_checkAvailabilityNow_blankApiKey_returnsFalseWithoutHttpCall() throws Exception {
        // setupClientForMockServer() sets a non-blank test API key, so the blank override
        // must be applied afterward - init() itself doesn't depend on the API key.
        setupClientForMockServer();
        client.setTestApiKey("");

        assertFalse(client.isAvailable());
        assertEquals("blank apiKey must short-circuit before any HTTP call", 0, mockServer.getRequestCount());
    }

    // ========== appendPath() endpoint composition ==========

    @Test
    public void test_appendPath_stripsSingleTrailingSlash() {
        // A base URL configured with a trailing slash must not produce a doubled slash before
        // the fixed sub-path (e.g. https://host/v1//embeddings), which some OpenAI-compatible
        // gateways reject.
        assertEquals("https://api.openai.com/v1/embeddings", OpenAiEmbeddingClient.appendPath("https://api.openai.com/v1/", "/embeddings"));
        assertEquals("https://api.openai.com/v1/embeddings", OpenAiEmbeddingClient.appendPath("https://api.openai.com/v1", "/embeddings"));
        assertEquals("https://api.openai.com/v1/models", OpenAiEmbeddingClient.appendPath("https://api.openai.com/v1/", "/models"));
    }

    @Test
    public void test_appendPath_nullBaseUrl_returnsPath() {
        assertEquals("/embeddings", OpenAiEmbeddingClient.appendPath(null, "/embeddings"));
    }

    @Test
    public void test_appendPath_preservesQueryString() {
        // The classic Azure OpenAI endpoint form carries the API version as a query string on the
        // configured base URL. Naively concatenating the sub-path onto the whole base URL buries
        // "/embeddings" inside the query value ("?api-version=2024-02-01/embeddings"), which the
        // gateway rejects. The query must be split off, the path appended, and the query re-attached.
        assertEquals("https://h.openai.azure.com/openai/deployments/d/embeddings?api-version=2024-02-01",
                OpenAiEmbeddingClient.appendPath("https://h.openai.azure.com/openai/deployments/d?api-version=2024-02-01", "/embeddings"));
        assertEquals("https://h/v1/embeddings?api-version=X",
                OpenAiEmbeddingClient.appendPath("https://h/v1?api-version=X", "/embeddings"));
        // A trailing slash before the query must still be collapsed, not preserved.
        assertEquals("https://h/v1/models?api-version=X", OpenAiEmbeddingClient.appendPath("https://h/v1/?api-version=X", "/models"));
        // A bare '?' with no parameters is preserved verbatim rather than being treated as path.
        assertEquals("https://h/v1/embeddings?", OpenAiEmbeddingClient.appendPath("https://h/v1?", "/embeddings"));
    }

    // ========== maskCredentialInUrl() ==========

    @Test
    public void test_maskCredentialInUrl_masksQueryParam() {
        assertEquals("https://gw.example.com/v1?api-key=***",
                OpenAiEmbeddingClient.maskCredentialInUrl("https://gw.example.com/v1?api-key=secret"));
        assertEquals("https://gw.example.com/v1?a=1&access_token=***&b=2",
                OpenAiEmbeddingClient.maskCredentialInUrl("https://gw.example.com/v1?a=1&access_token=shhh&b=2"));
    }

    @Test
    public void test_maskCredentialInUrl_masksUserInfo() {
        // Defensive rule: HttpClient rejects a userinfo-bearing request URI outright, so such a
        // URL can never issue a request. The rule only keeps a mistyped credential in that
        // position out of the log, since the query-parameter pattern never matches it.
        assertEquals("https://***:***@gw.example.com/v1", OpenAiEmbeddingClient.maskCredentialInUrl("https://user:pass@gw.example.com/v1"));
        assertEquals("http://***:***@gw.example.com/v1/embeddings",
                OpenAiEmbeddingClient.maskCredentialInUrl("http://user:pass@gw.example.com/v1/embeddings"));
        // Both rules must apply to the same URL.
        assertEquals("https://***:***@gw.example.com/v1?api-key=***",
                OpenAiEmbeddingClient.maskCredentialInUrl("https://user:pass@gw.example.com/v1?api-key=secret"));
    }

    @Test
    public void test_maskCredentialInUrl_cleanUrlUnchanged() {
        // No credentials anywhere: the URL must survive byte-for-byte. In particular the userinfo
        // rule must not fire on a port-bearing authority ("host:443") or on a path colon.
        assertEquals("https://api.openai.com/v1/embeddings",
                OpenAiEmbeddingClient.maskCredentialInUrl("https://api.openai.com/v1/embeddings"));
        assertEquals("https://gw.example.com:8443/v1/embeddings",
                OpenAiEmbeddingClient.maskCredentialInUrl("https://gw.example.com:8443/v1/embeddings"));
        assertEquals("https://h/v1/embeddings?api-version=2024-02-01",
                OpenAiEmbeddingClient.maskCredentialInUrl("https://h/v1/embeddings?api-version=2024-02-01"));
        assertNull(OpenAiEmbeddingClient.maskCredentialInUrl(null), "null must pass through");
    }

    // ========== dimensions.enabled override ==========

    @Test
    public void test_getConfigString_returnsDefaultWhenUnset() {
        // Drives the real, inherited AbstractEmbeddingClient#getConfigString seam (no subclass
        // override of getConfigString itself) to prove it falls back to the default when the key
        // is absent from the systemProperties channel, which is what makes "auto" the shipped
        // default for dimensions.enabled. Routed through ConfigStringProbe because getConfigString
        // is declared protected on AbstractEmbeddingClient (a different package): a non-subclass
        // caller cannot invoke an inherited protected member across packages, only a subclass can.
        assertEquals("auto", new ConfigStringProbe().probeConfigString("dimensions.enabled", "auto"));
    }

    @Test
    public void test_embedDocuments_dimensionsParamForcedOnForAzureDeploymentName() throws Exception {
        // On Azure OpenAI the "model" field is the operator-chosen deployment name, so a deployment
        // named "embeddings-prod" backed by text-embedding-3-small fails the name-prefix inference,
        // omits "dimensions", receives a native-length vector and trips the dimension guard on every
        // chunk forever. An explicit dimensions.enabled=true must force the parameter on.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        client.setTestModel("embeddings-prod");
        client.setTestConfig("content_chunker.embedding.openai.dimensions.enabled", "true");
        setupClientForMockServer();

        client.embedDocuments(List.of("chunk"));

        final String body = mockServer.takeRequest().getBody().readUtf8();
        assertTrue(body.contains("\"dimensions\":3"), "dimensions.enabled=true must force the parameter on: " + body);
    }

    @Test
    public void test_embedDocuments_dimensionsParamForcedOffForTextEmbedding3Large() throws Exception {
        // The inverse escape hatch: an OpenAI-compatible gateway may reject the "dimensions"
        // parameter even for a model whose name matches the text-embedding-3 prefix.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        client.setTestModel("text-embedding-3-large");
        client.setTestConfig("content_chunker.embedding.openai.dimensions.enabled", "false");
        setupClientForMockServer();

        client.embedDocuments(List.of("chunk"));

        final String body = mockServer.takeRequest().getBody().readUtf8();
        assertFalse(body.contains("\"dimensions\""), "dimensions.enabled=false must force the parameter off: " + body);
    }

    @Test
    public void test_supportsDimensionsParam_explicitAutoMatchesNameInference() {
        // "auto" (the default) must be byte-identical to the pre-existing name-prefix inference,
        // including the blank-model guard.
        client.setTestConfig("content_chunker.embedding.openai.dimensions.enabled", "auto");
        assertTrue(client.supportsDimensionsParam("text-embedding-3-small"));
        assertTrue(client.supportsDimensionsParam("text-embedding-3-large"));
        assertFalse(client.supportsDimensionsParam("text-embedding-ada-002"));
        assertFalse(client.supportsDimensionsParam("embeddings-prod"));
        assertFalse(client.supportsDimensionsParam(""));
        assertFalse(client.supportsDimensionsParam(null));
    }

    @Test
    public void test_supportsDimensionsParam_unrecognizedValueFallsBackToAuto() {
        // A typo must not silently disable the parameter for a text-embedding-3 model; it degrades
        // to the inference path (and is logged as a misconfiguration).
        client.setTestConfig("content_chunker.embedding.openai.dimensions.enabled", "ture");
        assertTrue(client.supportsDimensionsParam("text-embedding-3-small"));
        assertFalse(client.supportsDimensionsParam("embeddings-prod"));
    }

    @Test
    public void test_supportsDimensionsParam_explicitTrueIgnoresBlankModel() {
        // An explicit opt-in wins even over the blank-model guard: the operator has stated the
        // endpoint accepts the parameter.
        client.setTestConfig("content_chunker.embedding.openai.dimensions.enabled", "TRUE");
        assertTrue(client.supportsDimensionsParam("anything"));
        assertTrue(client.supportsDimensionsParam(""));
        assertTrue(client.supportsDimensionsParam(null));
    }

    @Test
    public void test_embedDocuments_trailingSlashBaseUrlProducesSingleSlashPath() throws Exception {
        // End-to-end: a base URL ending in '/' must reach the server as '/embeddings', not
        // '//embeddings'. appendPath() applies regardless of how getApiUrl() is sourced.
        final String responseJson = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}";
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        final String baseUrl = mockServer.url("").toString(); // MockWebServer base already ends with '/'
        client.setTestApiUrl(baseUrl);
        client.setTestApiKey("sk-test-key");
        client.setTestDimension(3);
        client.setTestTimeout(30000);
        client.init();

        client.embedDocuments(List.of("chunk"));

        final RecordedRequest recordedRequest = mockServer.takeRequest();
        assertEquals("trailing-slash base URL must yield a single-slash embeddings path", "/embeddings", recordedRequest.getPath());
    }

    // ========== malformed api.url must not leak the credential ==========

    /**
     * A configured {@code api.url} that carries a credential as a query parameter - the only
     * credential-in-URL form a working configuration can genuinely use - and whose value contains
     * a character that is illegal in a URI (here a space). {@code URI.create} rejects it and
     * quotes the whole URI in the {@link IllegalArgumentException} it raises.
     */
    private static final String MALFORMED_CREDENTIAL_URL = "https://gw.example.com/v1?api_key=sk secret";

    /** The credential value that must appear nowhere in a log line or in a propagated exception. */
    private static final String RAW_CREDENTIAL = "sk secret";

    /** Renders a throwable the way a log layout would: every message plus every frame of the cause chain. */
    private static String renderThrowable(final Throwable t) {
        final StringWriter sw = new StringWriter();
        t.printStackTrace(new PrintWriter(sw));
        return sw.toString();
    }

    @Test
    public void test_embedDocuments_malformedApiUrlDoesNotLeakCredential() {
        client.setTestApiUrl(MALFORMED_CREDENTIAL_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestDimension(3);
        client.setTestTimeout(30000);
        client.init();
        final ListAppender app = attachLogCapture();
        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException for a malformed api.url");
        } catch (final EmbeddingException e) {
            final String propagated = renderThrowable(e);
            assertFalse("propagated exception must not carry the credential: " + propagated, propagated.contains(RAW_CREDENTIAL));
        } finally {
            final String logged = app.rendered();
            detachLogCapture(app);
            assertFalse("log must not carry the credential: " + logged, logged.contains(RAW_CREDENTIAL));
            assertTrue("log must still identify the endpoint in masked form: " + logged, logged.contains("api_key=***"));
        }
    }

    @Test
    public void test_isAvailable_malformedApiUrlDoesNotLeakCredential() {
        client.setTestApiUrl(MALFORMED_CREDENTIAL_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestTimeout(30000);
        client.init();
        final ListAppender app = attachLogCapture();
        try {
            assertFalse(client.isAvailable());
        } finally {
            final String logged = app.rendered();
            detachLogCapture(app);
            assertFalse("log must not carry the credential: " + logged, logged.contains(RAW_CREDENTIAL));
        }
    }

    /**
     * A credential sitting in the URL's userinfo and containing a space. The userinfo masking
     * rule excludes whitespace, so it cannot match this URL - masking it is a no-op. Any
     * exception that echoes the URL, masked or not, therefore hands the credential straight back,
     * which is why the replacement exception must carry no URL at all.
     */
    private static final String MALFORMED_USERINFO_URL = "https://user:pw spaced@gw.example.com/v1";

    /** The userinfo credential that must appear in no exception the request-building path produces. */
    private static final String RAW_USERINFO_CREDENTIAL = "pw spaced";

    @Test
    public void test_embedDocuments_malformedUserInfoNotEchoedByException() {
        client.setTestApiUrl(MALFORMED_USERINFO_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestDimension(3);
        client.setTestTimeout(30000);
        client.init();
        final ListAppender app = attachLogCapture();
        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException for a malformed api.url");
        } catch (final EmbeddingException e) {
            final String propagated = renderThrowable(e);
            assertFalse("propagated cause chain must not carry the credential: " + propagated,
                    propagated.contains(RAW_USERINFO_CREDENTIAL));
        } finally {
            final String thrown = app.renderedThrowables();
            detachLogCapture(app);
            assertFalse("logged throwable must not carry the credential: " + thrown, thrown.contains(RAW_USERINFO_CREDENTIAL));
        }
    }

    // ========== userinfo-bearing api.url is refused before any request ==========

    /**
     * A {@code content_chunker.embedding.openai.api.url} carrying a userinfo credential. RFC 9110
     * forbids userinfo in an http/https target URI and HttpClient enforces that unconditionally, so
     * this value can never issue a request; OpenAI-compatible gateways authenticate with
     * {@code Authorization: Bearer}, and an endpoint behind an authenticating proxy is configured
     * through {@code http.proxy.*}. It is therefore an operator error with a supported alternative,
     * and the client must say so instead of failing opaquely at embed time.
     */
    private static final String USERINFO_API_URL = "https://user:s3cr3tUserinfo@gw.example.com/v1";

    /** The userinfo credential that must appear in no message, throwable or cause. */
    private static final String USERINFO_CREDENTIAL = "s3cr3tUserinfo";

    /** The supported alternative the refusal must name. */
    private static final String PROXY_USERNAME_KEY = "http.proxy.username";

    /** The supported alternative the refusal must name. */
    private static final String PROXY_PASSWORD_KEY = "http.proxy.password";

    @Test
    public void test_isAvailable_userInfoApiUrlReportsUnavailableWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestTimeout(30000);
        final ListAppender app = attachLogCapture();
        try {
            assertFalse("a userinfo-bearing api.url can never issue a request", client.isAvailable());
            final List<String> errors = app.messagesAt(Level.ERROR);
            assertEquals("the refusal must be reported at ERROR: " + app.messages(), 1, errors.size());
            final String error = errors.get(0);
            assertTrue("the offending configuration key must be named: " + error,
                    error.contains("content_chunker.embedding.openai.api.url"));
            assertTrue("the supported alternative must be named: " + error, error.contains(PROXY_USERNAME_KEY));
            assertTrue("the supported alternative must be named: " + error, error.contains(PROXY_PASSWORD_KEY));
        } finally {
            final String logged = app.rendered();
            final String thrown = app.renderedThrowables();
            detachLogCapture(app);
            assertFalse("log must not carry the credential: " + logged, logged.contains(USERINFO_CREDENTIAL));
            assertFalse("logged throwable must not carry the credential: " + thrown, thrown.contains(USERINFO_CREDENTIAL));
        }
    }

    @Test
    public void test_isAvailable_userInfoApiUrlErrorIsLoggedOnce() {
        // checkAvailabilityNow() runs on a timer in production, so an ERROR per call would flood
        // the log for as long as the misconfiguration lasts.
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestTimeout(30000);
        final ListAppender app = attachLogCapture();
        try {
            assertFalse(client.isAvailable());
            assertFalse(client.isAvailable());
            assertFalse(client.isAvailable());
            assertEquals("the remedy must be stated once, not on every availability check", 1, app.messagesAt(Level.ERROR).size());
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_isAvailable_portBearingApiUrlIsUnaffected() {
        // "host:port" is a port, not userinfo: the refusal must not fire on an ordinary
        // port-bearing gateway URL. The mock server supplies a real port-bearing authority.
        mockServer.enqueue(new MockResponse().setBody("{\"data\":[]}").addHeader("Content-Type", "application/json"));
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            assertTrue("a port-bearing api.url must still be usable", client.isAvailable());
            assertEquals("no refusal may be reported for a port-bearing URL: " + app.messages(), 0, app.messagesAt(Level.ERROR).size());
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_updateAvailability_userInfoApiUrlFailsClosedInsteadOfThrowing() {
        // The DI container runs init() as a postConstruct init method, and init() reaches
        // checkAvailabilityNow() through startAvailabilityCheck() -> updateAvailability(). An
        // exception escaping that frame would propagate through the eager init-method assembler
        // and stop the application from starting, so a bad api.url must disable this one optional
        // client, never abort startup. This pins that decision: turning the refusal into a throw
        // reddens this test.
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestTimeout(30000);
        client.testUpdateAvailability();
        assertFalse("the client must report itself unavailable", client.isAvailable());
    }

    @Test
    public void test_embedDocuments_userInfoApiUrlIsRefusedWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestDimension(3);
        client.setTestTimeout(30000);
        client.init();
        final ListAppender app = attachLogCapture();
        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException for a userinfo-bearing api.url");
        } catch (final EmbeddingException e) {
            final String propagated = renderThrowable(e);
            assertFalse("propagated exception must not carry the credential: " + propagated, propagated.contains(USERINFO_CREDENTIAL));
            assertTrue("the supported alternative must be named: " + e.getMessage(), e.getMessage().contains(PROXY_USERNAME_KEY));
            assertNull(e.getCause(), "cause must be absent: nothing may carry the URL");
        } finally {
            final String logged = app.rendered();
            detachLogCapture(app);
            assertFalse("log must not carry the credential: " + logged, logged.contains(USERINFO_CREDENTIAL));
        }
    }

    @Test
    public void test_embedQuery_userInfoApiUrlIsRefusedWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        client.setTestDimension(3);
        client.setTestTimeout(30000);
        client.init();
        try {
            client.embedQuery(List.of("query"));
            fail("expected EmbeddingException for a userinfo-bearing api.url");
        } catch (final EmbeddingException e) {
            final String propagated = renderThrowable(e);
            assertFalse("propagated exception must not carry the credential: " + propagated, propagated.contains(USERINFO_CREDENTIAL));
            assertTrue("the supported alternative must be named: " + e.getMessage(), e.getMessage().contains(PROXY_USERNAME_KEY));
        }
    }

    /** Captures log events for {@link OpenAiEmbeddingClient}, retaining the throwable as well as the message. */
    private static final class ListAppender extends AbstractAppender {
        final List<LogEvent> events = new ArrayList<>();

        ListAppender() {
            super("ListAppender", null, null, true, null);
        }

        @Override
        public void append(final LogEvent event) {
            events.add(event.toImmutable());
        }

        /** Every captured event's formatted message, in order. */
        List<String> messages() {
            return events.stream().map(e -> e.getMessage().getFormattedMessage()).toList();
        }

        /** The formatted messages of the events logged at the given level, in order. */
        List<String> messagesAt(final Level level) {
            return events.stream().filter(e -> e.getLevel().equals(level)).map(e -> e.getMessage().getFormattedMessage()).toList();
        }

        /**
         * Everything a real appender would write out: the formatted message <em>and</em> the
         * attached throwable. Reading only the formatted message cannot see a throwable, so an
         * assertion built on it goes green while the rendered log still leaks through the stack trace.
         */
        String rendered() {
            final StringBuilder buf = new StringBuilder();
            for (final LogEvent event : events) {
                buf.append(event.getMessage().getFormattedMessage()).append('\n');
                if (event.getThrown() != null) {
                    buf.append(renderThrowable(event.getThrown()));
                }
            }
            return buf.toString();
        }

        /**
         * Only the attached throwables, rendered with their full cause chain. Used where the
         * {@code url=} field of the message is itself outside the assertion's scope because the
         * masking rules - not the exception - govern what it may contain.
         */
        String renderedThrowables() {
            final StringBuilder buf = new StringBuilder();
            for (final LogEvent event : events) {
                if (event.getThrown() != null) {
                    buf.append(renderThrowable(event.getThrown()));
                }
            }
            return buf.toString();
        }
    }

    private ListAppender attachLogCapture() {
        final ListAppender appender = new ListAppender();
        appender.start();
        final LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        final LoggerConfig cfg = ctx.getConfiguration().getLoggerConfig(OpenAiEmbeddingClient.class.getName());
        cfg.addAppender(appender, Level.DEBUG, null);
        cfg.setLevel(Level.DEBUG);
        ctx.updateLoggers();
        return appender;
    }

    private void detachLogCapture(final ListAppender appender) {
        final LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        final LoggerConfig cfg = ctx.getConfiguration().getLoggerConfig(OpenAiEmbeddingClient.class.getName());
        cfg.removeAppender(appender.getName());
        ctx.updateLoggers();
        appender.stop();
    }

    /**
     * Exposes {@code AbstractEmbeddingClient#getConfigString}, otherwise unreachable from this
     * test: the method is {@code protected} and declared on a fess-core class in a different
     * package (not overridden here, unlike {@link TestableOpenAiEmbeddingClient}), so only a
     * subclass - not an unrelated caller such as this test class - may invoke the inherited
     * member. This probe adds no behavior, only package-visible access.
     */
    private static class ConfigStringProbe extends OpenAiEmbeddingClient {
        String probeConfigString(final String keySuffix, final String defaultValue) {
            return getConfigString(keySuffix, defaultValue);
        }
    }

    /**
     * Testable subclass of OpenAiEmbeddingClient that allows setting configuration values
     * directly without depending on FessConfig. Mirrors TestableOpenAiLlmClient's naming
     * convention from OpenAiLlmClientTest.
     */
    private static class TestableOpenAiEmbeddingClient extends OpenAiEmbeddingClient {
        private String testApiKey = "sk-test-key";
        private String testApiUrl = "https://api.openai.com/v1";
        private String testModel = "text-embedding-3-small";
        private int testTimeout = 60000;
        private int testRetryMax = 10;
        private long testRetryBaseDelayMs = 2000L;
        private Integer testDimension = 1536;
        /** Keyed by the *full* config key, so prefix + suffix composition is exercised too. */
        private final Map<String, String> testConfig = new HashMap<>();

        void setTestConfig(final String key, final String value) {
            testConfig.put(key, value);
        }

        @Override
        protected String getConfigString(final String keySuffix, final String defaultValue) {
            return testConfig.getOrDefault(getConfigPrefix() + "." + keySuffix, defaultValue);
        }

        void setTestApiKey(final String apiKey) {
            this.testApiKey = apiKey;
        }

        void setTestApiUrl(final String apiUrl) {
            this.testApiUrl = apiUrl;
        }

        void setTestModel(final String model) {
            this.testModel = model;
        }

        void setTestTimeout(final int timeout) {
            this.testTimeout = timeout;
        }

        void setTestRetryMax(final int max) {
            this.testRetryMax = max;
        }

        void setTestRetryBaseDelayMs(final long ms) {
            this.testRetryBaseDelayMs = ms;
        }

        void setTestDimension(final Integer dimension) {
            this.testDimension = dimension;
        }

        @Override
        protected String getApiKey() {
            return testApiKey;
        }

        @Override
        protected String getApiUrl() {
            return testApiUrl;
        }

        @Override
        protected String getModel() {
            return testModel;
        }

        @Override
        protected int getTimeout() {
            return testTimeout;
        }

        @Override
        protected int getRetryMaxAttempts() {
            return testRetryMax;
        }

        @Override
        protected long getRetryBaseDelayMs() {
            return testRetryBaseDelayMs;
        }

        @Override
        public int getDimension() {
            if (testDimension == null) {
                throw new EmbeddingException("content_chunker.embedding.dimension is not configured");
            }
            return testDimension;
        }

        @Override
        protected String getEmbeddingType() {
            // Matches getName() so AbstractEmbeddingClient#init() actually builds the
            // HTTP client instead of skipping (the gate it uses in production to decide
            // whether this provider is the one currently selected).
            return NAME;
        }

        @Override
        protected int getAvailabilityCheckInterval() {
            // Disable periodic availability-check scheduling in tests.
            return 0;
        }

        /** Exposes the frame the container reaches through {@code init()}, so a test can prove it never throws. */
        void testUpdateAvailability() {
            updateAvailability();
        }
    }
}
