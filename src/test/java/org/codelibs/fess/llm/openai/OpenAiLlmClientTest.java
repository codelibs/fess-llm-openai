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

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.codelibs.fess.llm.LlmChatRequest;
import org.codelibs.fess.llm.LlmChatResponse;
import org.codelibs.fess.llm.LlmException;
import org.codelibs.fess.llm.LlmMessage;
import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.openai.util.OpenAiRetry;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

public class OpenAiLlmClientTest extends UnitFessTestCase {

    private TestableOpenAiLlmClient client;
    private MockWebServer mockServer;

    @Override
    protected void setUp(TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableOpenAiLlmClient();
        mockServer = new MockWebServer();
        mockServer.start();
    }

    @Override
    protected void tearDown(TestInfo testInfo) throws Exception {
        if (client != null) {
            client.destroy();
        }
        if (mockServer != null) {
            mockServer.shutdown();
        }
        super.tearDown(testInfo);
    }

    @Test
    public void test_getName() {
        assertEquals("openai", client.getName());
    }

    @Test
    public void test_isAvailable_noApiKey() {
        client.setTestApiKey("");
        client.setTestApiUrl("https://api.openai.com/v1");
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_nullApiKey() {
        client.setTestApiKey(null);
        client.setTestApiUrl("https://api.openai.com/v1");
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_noApiUrl() {
        client.setTestApiKey("sk-test-key");
        client.setTestApiUrl("");
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_nullApiUrl() {
        client.setTestApiKey("sk-test-key");
        client.setTestApiUrl(null);
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_valid() throws IOException {
        // Mock the /models endpoint for availability check
        mockServer.enqueue(new MockResponse().setBody("{\"data\":[]}").addHeader("Content-Type", "application/json"));
        setupClientForMockServer();
        assertTrue(client.isAvailable());
    }

    @Test
    public void test_isAvailable_logsMaskedUrlOnError() throws IOException {
        // checkAvailabilityNow() must mask credential query params in DEBUG logs so a
        // proxy URL like https://gw.example/v1?api_key=secret never leaks the key.
        client.setTestApiKey("sk-test");
        client.setTestApiUrl(mockServer.url("/").toString().replaceAll("/$", "") + "?api_key=secret");
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(401).setBody("{}"));
            assertFalse(client.isAvailable());
            assertTrue("availability DEBUG must mask credential",
                    app.messagesAt(org.apache.logging.log4j.Level.DEBUG)
                            .stream()
                            .anyMatch(s -> s.contains("api_key=***") && !s.contains("api_key=secret")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_convertMessage_user() {
        final LlmMessage message = LlmMessage.user("Hello, how are you?");
        final Map<String, String> result = client.convertMessage(message);

        assertEquals("user", result.get("role"));
        assertEquals("Hello, how are you?", result.get("content"));
    }

    @Test
    public void test_convertMessage_assistant() {
        final LlmMessage message = LlmMessage.assistant("I'm doing well, thank you!");
        final Map<String, String> result = client.convertMessage(message);

        assertEquals("assistant", result.get("role"));
        assertEquals("I'm doing well, thank you!", result.get("content"));
    }

    @Test
    public void test_convertMessage_system() {
        final LlmMessage message = LlmMessage.system("You are a helpful assistant.");
        final Map<String, String> result = client.convertMessage(message);

        assertEquals("system", result.get("role"));
        assertEquals("You are a helpful assistant.", result.get("content"));
    }

    @Test
    public void test_buildRequestBody_defaultValues() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-5-mini", body.get("model"));
        assertEquals(false, body.get("stream"));
        assertNull(body.get("temperature"));
        assertEquals(4096, body.get("max_completion_tokens"));

        @SuppressWarnings("unchecked")
        final List<Map<String, String>> messages = (List<Map<String, String>>) body.get("messages");
        assertEquals(1, messages.size());
        assertEquals("user", messages.get(0).get("role"));
        assertEquals("Hello", messages.get(0).get("content"));
    }

    @Test
    public void test_buildRequestBody_withRequestModel() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setModel("gpt-3.5-turbo").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-3.5-turbo", body.get("model"));
    }

    @Test
    public void test_buildRequestBody_withRequestTemperature() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setTemperature(0.5).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals(0.5, body.get("temperature"));
    }

    @Test
    public void test_buildRequestBody_reasoningModelOmitsSamplingParams() {
        // Verified against the live API: gpt-5 answers 400 "Unsupported parameter: 'top_p' is
        // not supported with this model." and the same for both penalties -- the identical
        // treatment temperature already gets. Sending them fails the whole chat rather than
        // degrading it, so a configured rag.llm.openai.<promptType>.top.p must be dropped.
        client.setTestModel("gpt-5-nano");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        request.putExtraParam("top_p", "0.9");
        request.putExtraParam("frequency_penalty", "0.5");
        request.putExtraParam("presence_penalty", "0.5");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("top_p"), "top_p must not be sent to a reasoning model");
        assertNull(body.get("frequency_penalty"), "frequency_penalty must not be sent to a reasoning model");
        assertNull(body.get("presence_penalty"), "presence_penalty must not be sent to a reasoning model");
    }

    @Test
    public void test_buildRequestBody_nonReasoningModelKeepsSamplingParams() {
        // The fallback path for unrecognized names (Azure deployments, OpenAI-compatible
        // gateways) must keep accepting these, so the suppression is gated on the model family
        // and not applied unconditionally.
        client.setTestModel("my-azure-deployment");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        request.putExtraParam("top_p", "0.9");
        request.putExtraParam("frequency_penalty", "0.25");
        request.putExtraParam("presence_penalty", "0.125");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals(0.9, body.get("top_p"));
        assertEquals(0.25, body.get("frequency_penalty"));
        assertEquals(0.125, body.get("presence_penalty"));
    }

    @Test
    public void test_modelFamilyPredicatesAgree() {
        // useMaxCompletionTokens and supportsTemperature delegate to isReasoningModel so the
        // three cannot drift apart. This also pins the blank-model defaults, which differ:
        // a blank name is treated as non-reasoning, hence max_tokens and temperature allowed.
        for (final String model : new String[] { "gpt-5", "gpt-5-nano", "o1-preview", "o3-mini", "o4-mini" }) {
            assertTrue(model + " must be a reasoning model", client.isReasoningModel(model));
            assertTrue(model + " must use max_completion_tokens", client.useMaxCompletionTokens(model));
            assertFalse(model + " must not accept temperature", client.supportsTemperature(model));
            assertFalse(model + " must not accept sampling params", client.supportsSamplingParams(model));
        }
        // gpt-4o and friends are outside the tested set but must stay usable when configured,
        // so the classic parameter set is pinned for them alongside unknown names.
        for (final String model : new String[] { "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "my-azure-deployment", "" }) {
            assertFalse("'" + model + "' must not be a reasoning model", client.isReasoningModel(model));
            assertFalse("'" + model + "' must use max_tokens", client.useMaxCompletionTokens(model));
            assertTrue("'" + model + "' must accept temperature", client.supportsTemperature(model));
            assertTrue("'" + model + "' must accept sampling params", client.supportsSamplingParams(model));
        }
    }

    @Test
    public void test_buildRequestBody_withRequestMaxTokens() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(1000).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals(1000, body.get("max_completion_tokens"));
    }

    @Test
    public void test_buildRequestBody_streaming() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, true);

        assertEquals(true, body.get("stream"));
    }

    @Test
    public void test_buildRequestBody_streamingIncludesUsageByDefault() {
        final Map<String, Object> body = client.buildRequestBody(buildSimpleRequest(), true);
        @SuppressWarnings("unchecked")
        final Map<String, Object> opts = (Map<String, Object>) body.get("stream_options");
        assertNotNull(opts, "stream_options should be present for streaming requests");
        assertEquals(Boolean.TRUE, opts.get("include_usage"));
    }

    @Test
    public void test_buildRequestBody_nonStreamingExcludesStreamOptions() {
        final Map<String, Object> body = client.buildRequestBody(buildSimpleRequest(), false);
        assertNull(body.get("stream_options"), "stream_options must not appear on non-streaming requests");
    }

    @Test
    public void test_buildRequestBody_streamingOmitsUsageWhenDisabled() {
        client.setTestConfig("stream.include.usage", "false");
        final Map<String, Object> body = client.buildRequestBody(buildSimpleRequest(), true);
        assertNull(body.get("stream_options"), "opt-out via config");
    }

    @Test
    public void test_buildRequestBody_multipleMessages() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addSystemMessage("You are a helpful assistant.")
                .addUserMessage("What is the weather?")
                .addAssistantMessage("I cannot access weather information.")
                .addUserMessage("OK");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        @SuppressWarnings("unchecked")
        final List<Map<String, String>> messages = (List<Map<String, String>>) body.get("messages");
        assertEquals(4, messages.size());

        assertEquals("system", messages.get(0).get("role"));
        assertEquals("You are a helpful assistant.", messages.get(0).get("content"));

        assertEquals("user", messages.get(1).get("role"));
        assertEquals("What is the weather?", messages.get(1).get("content"));

        assertEquals("assistant", messages.get(2).get("role"));
        assertEquals("I cannot access weather information.", messages.get(2).get("content"));

        assertEquals("user", messages.get(3).get("role"));
        assertEquals("OK", messages.get(3).get("content"));
    }

    @Test
    public void test_buildRequestBody_blankModelUsesDefault() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setModel("").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-5-mini", body.get("model"));
    }

    @Test
    public void test_buildRequestBody_nullModelUsesDefault() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setModel(null).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-5-mini", body.get("model"));
    }

    @Test
    public void test_init() {
        client.setTestTimeout(30000);
        client.init();
        assertNotNull(client.getHttpClient());
    }

    @Test
    public void test_getHttpClient_lazyInitialization() {
        client.setTestTimeout(60000);
        // First call should initialize the client
        assertNotNull(client.getHttpClient());
        // Second call should return the same client
        assertNotNull(client.getHttpClient());
    }

    // ========== chat() method tests ==========

    @Test
    public void test_chat_success() throws IOException {
        final String responseJson = """
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals("Hello! How can I help you today?", response.getContent());
        assertEquals("stop", response.getFinishReason());
        assertEquals("gpt-4", response.getModel());
        assertEquals(10, response.getPromptTokens());
        assertEquals(20, response.getCompletionTokens());
        assertEquals(30, response.getTotalTokens());
    }

    @Test
    public void test_chat_successWithMinimalResponse() throws IOException {
        final String responseJson = """
                {
                    "choices": [{
                        "message": {
                            "content": "Response text"
                        }
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals("Response text", response.getContent());
    }

    @Test
    public void test_chat_errorResponse_withBody() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "message": "Invalid API key provided",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(401).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("401"));
        }
    }

    @Test
    public void test_chat_errorResponse_rateLimitExceeded() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(429).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();
        // 429 is retryable; cap attempts at 1 so this asserts surfacing rather than retry behavior.
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "0");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            // After retry exhaustion the LlmException wraps the RetryableHttpException whose message contains the status code.
            final Throwable cause = error.getCause();
            assertNotNull(cause);
            assertTrue("expected status code 429 in cause message: " + cause.getMessage(),
                    cause.getMessage() != null && cause.getMessage().contains("429"));
            assertEquals("429 must surface as ERROR_RATE_LIMIT, not ERROR_CONNECTION", LlmException.ERROR_RATE_LIMIT, error.getErrorCode());
        }
    }

    @Test
    public void test_chat_errorResponse_serverError() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "message": "Internal server error",
                        "type": "server_error"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(500).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();
        // 500 is retryable; cap attempts at 1 to surface the failure immediately.
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "0");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            final Throwable cause = error.getCause();
            assertNotNull(cause);
            assertTrue("expected status code 500 in cause message: " + cause.getMessage(),
                    cause.getMessage() != null && cause.getMessage().contains("500"));
            // 500 maps to ERROR_UNKNOWN per resolveErrorCode (only 429/401/403/404/408/502/503 have explicit codes).
            assertEquals(LlmException.ERROR_UNKNOWN, error.getErrorCode());
        }
    }

    @Test
    public void test_chat_errorResponse_emptyBody() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("").addHeader("Content-Type", "application/json"));

        setupClientForMockServer();
        // 503 is retryable; cap attempts at 1 to surface the failure immediately.
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "0");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            final Throwable cause = error.getCause();
            assertNotNull(cause);
            assertTrue("expected status code 503 in cause message: " + cause.getMessage(),
                    cause.getMessage() != null && cause.getMessage().contains("503"));
            assertEquals("503 must surface as ERROR_SERVICE_UNAVAILABLE, not ERROR_CONNECTION", LlmException.ERROR_SERVICE_UNAVAILABLE,
                    error.getErrorCode());
        }
    }

    @Test
    public void test_chat_emptyChoices() throws IOException {
        final String responseJson = """
                {
                    "choices": []
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertNull(response.getContent());
    }

    @Test
    public void test_chat_nullFinishReason() throws IOException {
        final String responseJson = """
                {
                    "choices": [{
                        "message": {
                            "content": "Test"
                        },
                        "finish_reason": null
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals("Test", response.getContent());
        assertNull(response.getFinishReason());
    }

    @Test
    public void test_chat_partialUsage() throws IOException {
        final String responseJson = """
                {
                    "choices": [{
                        "message": {
                            "content": "Test"
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 5
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals(5, response.getPromptTokens());
        assertNull(response.getCompletionTokens());
        assertNull(response.getTotalTokens());
    }

    // ========== chat() retry/diagnostics tests (Task 10) ==========

    @Test
    public void test_chat_retriesOn429ThenSucceeds() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "3");
        client.setTestConfig("retry.base.delay.ms", "10");
        mockServer.enqueue(new MockResponse().setResponseCode(429)
                .addHeader("Content-Type", "application/json")
                .setBody("{\"error\":{\"message\":\"rate limit\",\"type\":\"rate_limit_exceeded\"}}"));
        mockServer.enqueue(new MockResponse().setResponseCode(200).setBody(simpleSuccessBody()));
        final LlmChatResponse resp = client.chat(buildSimpleRequest());
        assertEquals("ok", resp.getContent());
        assertEquals(2, mockServer.getRequestCount());
    }

    @Test
    public void test_chat_honorsRetryAfterHeader_429() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "100000"); // huge backoff to detect override
        mockServer.enqueue(new MockResponse().setResponseCode(429)
                .addHeader("Retry-After", "0")
                .setBody("{\"error\":{\"type\":\"rate_limit_exceeded\"}}"));
        mockServer.enqueue(new MockResponse().setResponseCode(200).setBody(simpleSuccessBody()));
        final long start = System.currentTimeMillis();
        client.chat(buildSimpleRequest());
        // Threshold loose enough to absorb CI noise; the configured backoff is 100s, so any sub-30s
        // run proves Retry-After=0 took precedence.
        assertTrue("Retry-After=0 must override large backoff", System.currentTimeMillis() - start < 30000);
    }

    @Test
    public void test_chat_honorsRetryAfterHeader_503() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "100000");
        mockServer.enqueue(new MockResponse().setResponseCode(503).addHeader("Retry-After", "0").setBody("{}"));
        mockServer.enqueue(new MockResponse().setResponseCode(200).setBody(simpleSuccessBody()));
        final long start = System.currentTimeMillis();
        client.chat(buildSimpleRequest());
        assertTrue(System.currentTimeMillis() - start < 30000);
    }

    @Test
    public void test_chat_retryAfterClamped_largeValueReducesTo600s() throws Exception {
        // parseRetryAfterSeconds clamps any value > 600s down to 600s. With retry.max=1 the single
        // attempt fails immediately (no real sleep) but the WARN line must record the clamped value
        // proving the call site at OpenAiLlmClient.java:393-395 applies the clamp.
        setupClientForMockServer();
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "10");
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(503).addHeader("Retry-After", "99999").setBody("{}"));
            try {
                client.chat(buildSimpleRequest());
                fail("expected LlmException after retry exhaustion");
            } catch (final LlmException expected) {
                assertEquals(LlmException.ERROR_SERVICE_UNAVAILABLE, expected.getErrorCode());
            }
            assertTrue("WARN must record retry exhaustion with clamped retryAfter=600s",
                    app.messagesAt(org.apache.logging.log4j.Level.WARN)
                            .stream()
                            .anyMatch(s -> s.contains("retry exhausted") && s.contains("retryAfter=600s")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_retryAfterHttpDate_fallsBackToBackoff() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "10"); // tiny so backoff is fast
        mockServer.enqueue(new MockResponse().setResponseCode(429).addHeader("Retry-After", "Wed, 21 Oct 2026 07:28:00 GMT").setBody("{}"));
        mockServer.enqueue(new MockResponse().setResponseCode(200).setBody(simpleSuccessBody()));
        client.chat(buildSimpleRequest());
        assertEquals(2, mockServer.getRequestCount());
    }

    @Test
    public void test_chat_doesNotRetry_400() throws Exception {
        setupClientForMockServer();
        mockServer.enqueue(new MockResponse().setResponseCode(400)
                .setBody("{\"error\":{\"message\":\"bad request\",\"type\":\"invalid_request_error\"}}"));
        try {
            client.chat(buildSimpleRequest());
            fail("expected LlmException");
        } catch (final LlmException expected) {
            assertEquals(1, mockServer.getRequestCount());
        }
    }

    @Test
    public void test_chat_retriesOn500_502_503_504() throws Exception {
        for (final int status : new int[] { 500, 502, 503, 504 }) {
            // Re-init fresh state for each iteration.
            if (mockServer != null)
                mockServer.shutdown();
            mockServer = new MockWebServer();
            mockServer.start();
            client = new TestableOpenAiLlmClient();
            setupClientForMockServer();
            client.setTestConfig("retry.max", "2");
            client.setTestConfig("retry.base.delay.ms", "10");
            mockServer.enqueue(new MockResponse().setResponseCode(status).setBody("{}"));
            mockServer.enqueue(new MockResponse().setResponseCode(200).setBody(simpleSuccessBody()));
            client.chat(buildSimpleRequest());
            assertEquals("status " + status + " must retry", 2, mockServer.getRequestCount());
        }
    }

    @Test
    public void test_chat_exhaustsRetries() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "3");
        client.setTestConfig("retry.base.delay.ms", "10");
        for (int i = 0; i < 3; i++) {
            mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("{}"));
        }
        try {
            client.chat(buildSimpleRequest());
            fail("expected LlmException after retry exhaustion");
        } catch (final LlmException expected) {
            assertEquals(3, mockServer.getRequestCount());
            // Retry exhaustion must preserve the status-driven error code, not collapse to ERROR_CONNECTION.
            assertEquals(LlmException.ERROR_SERVICE_UNAVAILABLE, expected.getErrorCode());
        }
    }

    @Test
    public void test_chat_exhaustsRetries_429PreservesRateLimitCode() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "10");
        for (int i = 0; i < 2; i++) {
            mockServer.enqueue(new MockResponse().setResponseCode(429).setBody("{}"));
        }
        try {
            client.chat(buildSimpleRequest());
            fail("expected LlmException after retry exhaustion");
        } catch (final LlmException expected) {
            assertEquals(LlmException.ERROR_RATE_LIMIT, expected.getErrorCode());
        }
    }

    @Test
    public void test_chat_exhaustsRetries_502PreservesServiceUnavailableCode() throws Exception {
        // 502 must map to ERROR_SERVICE_UNAVAILABLE on retry exhaustion (not ERROR_CONNECTION).
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "10");
        for (int i = 0; i < 2; i++) {
            mockServer.enqueue(new MockResponse().setResponseCode(502).setBody("{}"));
        }
        try {
            client.chat(buildSimpleRequest());
            fail("expected LlmException after retry exhaustion");
        } catch (final LlmException expected) {
            assertEquals(LlmException.ERROR_SERVICE_UNAVAILABLE, expected.getErrorCode());
        }
    }

    @Test
    public void test_chat_exhaustsRetries_504MapsToUnknown() throws Exception {
        // 504 is retryable but resolveErrorCode treats it as ERROR_UNKNOWN (only 502/503 map to
        // SERVICE_UNAVAILABLE per the central mapping in AbstractLlmClient.resolveErrorCode).
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "10");
        for (int i = 0; i < 2; i++) {
            mockServer.enqueue(new MockResponse().setResponseCode(504).setBody("{}"));
        }
        try {
            client.chat(buildSimpleRequest());
            fail("expected LlmException after retry exhaustion");
        } catch (final LlmException expected) {
            assertEquals(LlmException.ERROR_UNKNOWN, expected.getErrorCode());
        }
    }

    @Test
    public void test_chat_warnsOnMessageRefusal() throws Exception {
        // Non-streaming refusal must surface symmetrically with streamChat's delta.refusal handling
        // — otherwise refusal pairs with finish_reason=stop and null content and is silently logged
        // as a normal empty success.
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(200)
                    .setBody("{\"id\":\"chatcmpl-r\",\"model\":\"gpt-5-mini\","
                            + "\"choices\":[{\"message\":{\"content\":null,\"refusal\":\"I cannot help with that.\"},"
                            + "\"finish_reason\":\"stop\"}]}"));
            client.chat(buildSimpleRequest());
            assertTrue("WARN must include refusal text", app.messagesAt(org.apache.logging.log4j.Level.WARN)
                    .stream()
                    .anyMatch(s -> s.contains("Chat refusal") && s.contains("I cannot help with that") && s.contains("id=chatcmpl-r")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_messageContentAndRefusalCoexist() throws Exception {
        // When the assistant returns BOTH partial content and a refusal field with finish_reason=stop,
        // the response content must still be exposed and the WARN must surface the refusal text.
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(200)
                    .setBody("{\"id\":\"chatcmpl-mix\",\"model\":\"gpt-5-mini\","
                            + "\"choices\":[{\"message\":{\"content\":\"partial answer\",\"refusal\":\"I cannot help with the rest\"},"
                            + "\"finish_reason\":\"stop\"}]}"));
            final LlmChatResponse resp = client.chat(buildSimpleRequest());
            assertEquals("partial answer", resp.getContent());
            assertTrue("WARN must surface refusal text", app.messagesAt(org.apache.logging.log4j.Level.WARN)
                    .stream()
                    .anyMatch(
                            s -> s.contains("Chat refusal") && s.contains("I cannot help with the rest") && s.contains("id=chatcmpl-mix")));
            // INFO line should still report the actual content length (= "partial answer".length() = 14).
            assertTrue("INFO line must report contentLength matching the visible content",
                    app.messagesAt(org.apache.logging.log4j.Level.INFO)
                            .stream()
                            .anyMatch(s -> s.contains("Chat response received") && s.contains("contentLength=14")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_messageContentFilterWithRefusal() throws Exception {
        // content_filter finish reason combined with a refusal must fire BOTH the abnormal-finish WARN
        // and the refusal WARN — they are independent diagnostics.
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(200)
                    .setBody("{\"id\":\"chatcmpl-cf\",\"model\":\"gpt-5-mini\","
                            + "\"choices\":[{\"message\":{\"content\":null,\"refusal\":\"Cannot comply with policy.\"},"
                            + "\"finish_reason\":\"content_filter\"}]}"));
            client.chat(buildSimpleRequest());
            assertTrue("expected refusal WARN",
                    app.messagesAt(org.apache.logging.log4j.Level.WARN)
                            .stream()
                            .anyMatch(s -> s.contains("Chat refusal") && s.contains("Cannot comply with policy")));
            assertTrue("expected abnormal-finish WARN with finishReason=content_filter",
                    app.messagesAt(org.apache.logging.log4j.Level.WARN)
                            .stream()
                            .anyMatch(s -> s.contains("Chat finished abnormally") && s.contains("finishReason=content_filter")
                                    && s.contains("id=chatcmpl-cf")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_logsAbnormalFinishReason_length() throws Exception {
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(200)
                    .setBody("{\"id\":\"chatcmpl-1\",\"model\":\"gpt-5-mini\","
                            + "\"choices\":[{\"message\":{\"content\":\"truncated...\"},\"finish_reason\":\"length\"}],"
                            + "\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":50,\"total_tokens\":51}}"));
            client.chat(buildSimpleRequest());
            assertTrue("abnormal-finish WARN must include id and finishReason",
                    app.messagesAt(org.apache.logging.log4j.Level.WARN)
                            .stream()
                            .anyMatch(s -> s.contains("Chat finished abnormally") && s.contains("finishReason=length")
                                    && s.contains("id=chatcmpl-1")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_logsResponseIdAndUsageDetails() throws Exception {
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(200)
                    .setBody("{\"id\":\"chatcmpl-9\",\"system_fingerprint\":\"fp_abc\",\"model\":\"gpt-5-mini\","
                            + "\"choices\":[{\"message\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
                            + "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,"
                            + "\"completion_tokens_details\":{\"reasoning_tokens\":3},"
                            + "\"prompt_tokens_details\":{\"cached_tokens\":7}}}"));
            client.chat(buildSimpleRequest());
            final String line = app.messagesAt(org.apache.logging.log4j.Level.INFO)
                    .stream()
                    .filter(s -> s.contains("Chat response received"))
                    .findFirst()
                    .orElse("");
            assertTrue("missing id: " + line, line.contains("id=chatcmpl-9"));
            assertTrue("missing systemFingerprint: " + line, line.contains("systemFingerprint=fp_abc"));
            assertTrue("missing reasoningTokens: " + line, line.contains("reasoningTokens=3"));
            assertTrue("missing cachedTokens: " + line, line.contains("cachedTokens=7"));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_oldModelWithoutSystemFingerprint() throws Exception {
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(200)
                    .setBody("{\"id\":\"chatcmpl-x\",\"model\":\"gpt-3.5-turbo\","
                            + "\"choices\":[{\"message\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
                            + "\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"));
            client.chat(buildSimpleRequest());
            final String line = app.messagesAt(org.apache.logging.log4j.Level.INFO)
                    .stream()
                    .filter(s -> s.contains("Chat response received"))
                    .findFirst()
                    .orElse("");
            assertTrue("INFO line must still fire when system_fingerprint absent", line.contains("systemFingerprint=null"));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_errorBodyParsedIntoLog() throws Exception {
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(401)
                    .setBody("{\"error\":{\"message\":\"Invalid API key\",\"type\":\"invalid_request_error\","
                            + "\"code\":\"invalid_api_key\",\"param\":null}}"));
            try {
                client.chat(buildSimpleRequest());
                fail();
            } catch (final LlmException e) { /* expected */ }
            assertTrue(app.messagesAt(org.apache.logging.log4j.Level.WARN)
                    .stream()
                    .anyMatch(s -> s.contains("API error") && s.contains("type=invalid_request_error")
                            && s.contains("code=invalid_api_key")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_chat_logsMaskedUrlOnError() throws Exception {
        // Configure a credential-bearing URL; verify it's masked in logs.
        client.setTestApiKey("sk-test");
        client.setTestApiUrl(mockServer.url("/").toString().replaceAll("/$", "") + "?api_key=secret");
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(new MockResponse().setResponseCode(400).setBody("{\"error\":{}}"));
            try {
                client.chat(buildSimpleRequest());
                fail();
            } catch (final LlmException e) { /* expected */ }
            assertTrue("URL must be masked",
                    app.messagesAt(org.apache.logging.log4j.Level.WARN)
                            .stream()
                            .anyMatch(s -> s.contains("api_key=***") && !s.contains("api_key=secret")));
        } finally {
            detachLogCapture(app);
        }
    }

    // ========== streamChat() method tests ==========

    @Test
    public void test_streamChat_retriesInitialConnectOn503() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "10");
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("{}"));
        mockServer.enqueue(
                new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(simpleStreamSseBody()));
        final StringBuilder collected = new StringBuilder();
        client.streamChat(buildSimpleRequest(), (content, done) -> collected.append(content));
        assertEquals("ok", collected.toString());
        assertEquals(2, mockServer.getRequestCount());
    }

    @Test
    public void test_streamChat_exhaustsRetries_429PreservesRateLimitCode() throws Exception {
        // Mirror chat()'s retry-exhaustion code preservation: streamChat() must surface the status-driven
        // ERROR_RATE_LIMIT (not collapse to ERROR_CONNECTION) both via the thrown LlmException and via onError.
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "10");
        for (int i = 0; i < 2; i++) {
            mockServer.enqueue(new MockResponse().setResponseCode(429).setBody("{}"));
        }
        final java.util.concurrent.atomic.AtomicReference<Throwable> errorRef = new java.util.concurrent.atomic.AtomicReference<>();
        try {
            client.streamChat(buildSimpleRequest(), new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on retry exhaustion");
                }

                @Override
                public void onError(final Throwable t) {
                    errorRef.set(t);
                }
            });
            fail("expected LlmException after retry exhaustion");
        } catch (final LlmException expected) {
            assertEquals(LlmException.ERROR_RATE_LIMIT, expected.getErrorCode());
            assertTrue("onError must receive the same LlmException, not a collapsed ERROR_CONNECTION wrapper", expected == errorRef.get());
        }
    }

    @Test
    public void test_streamChat_exhaustsRetries_503PreservesServiceUnavailableCode() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "2");
        client.setTestConfig("retry.base.delay.ms", "10");
        for (int i = 0; i < 2; i++) {
            mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("{}"));
        }
        final java.util.concurrent.atomic.AtomicReference<Throwable> errorRef = new java.util.concurrent.atomic.AtomicReference<>();
        try {
            client.streamChat(buildSimpleRequest(), new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on retry exhaustion");
                }

                @Override
                public void onError(final Throwable t) {
                    errorRef.set(t);
                }
            });
            fail("expected LlmException after retry exhaustion");
        } catch (final LlmException expected) {
            assertEquals(LlmException.ERROR_SERVICE_UNAVAILABLE, expected.getErrorCode());
            assertTrue("onError must receive the same LlmException, not a collapsed ERROR_CONNECTION wrapper", expected == errorRef.get());
            assertTrue("onError throwable must also carry SERVICE_UNAVAILABLE", errorRef.get() instanceof LlmException
                    && LlmException.ERROR_SERVICE_UNAVAILABLE.equals(((LlmException) errorRef.get()).getErrorCode()));
        }
    }

    @Test
    public void test_streamChat_partialStreamErrorPropagates() throws Exception {
        setupClientForMockServer();
        client.setTestConfig("retry.max", "5");
        client.setTestConfig("retry.base.delay.ms", "10");
        // Truncated body — connection appears to drop mid-stream.
        mockServer.enqueue(new MockResponse().setResponseCode(200)
                .addHeader("Content-Type", "text/event-stream")
                .setBody("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hel"));
        try {
            client.streamChat(buildSimpleRequest(), (c, d) -> { /* swallow */ });
        } catch (final LlmException e) { /* allowed */ }
        assertEquals("partial-stream errors must NOT trigger retry", 1, mockServer.getRequestCount());
    }

    @Test
    public void test_streamChat_callbackRuntimeException_invokesOnError() throws Exception {
        // A RuntimeException thrown by the consumer's onChunk must surface via onError so
        // consumers using the SPI's onError contract for cleanup don't silently leak.
        setupClientForMockServer();
        mockServer.enqueue(
                new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(simpleStreamSseBody()));
        final java.util.concurrent.atomic.AtomicReference<Throwable> errorRef = new java.util.concurrent.atomic.AtomicReference<>();
        try {
            client.streamChat(buildSimpleRequest(), new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    throw new RuntimeException("consumer boom");
                }

                @Override
                public void onError(final Throwable t) {
                    errorRef.set(t);
                }
            });
            fail("expected propagation");
        } catch (final LlmException expected) {
            // ok
        }
        assertNotNull(errorRef.get(), "onError must fire when callback throws");
    }

    @Test
    public void test_streamChat_capturesFinalUsageChunk_afterDoneChunk() throws Exception {
        // Verifies the critical fix: usage chunk arrives after the done chunk and must be captured.
        setupClientForMockServer();
        final java.util.concurrent.atomic.AtomicReference<OpenAiLlmClient.StreamSummary> ref =
                new java.util.concurrent.atomic.AtomicReference<>();
        client.setStreamSummaryConsumer(ref::set);
        final String body = "data: {\"id\":\"chatcmpl-1\",\"system_fingerprint\":\"fp_abc\","
                + "\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,"
                + "\"completion_tokens\":1,\"total_tokens\":4,\"completion_tokens_details\":{\"reasoning_tokens\":2},"
                + "\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n" + "data: [DONE]\n\n";
        mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
        client.streamChat(buildSimpleRequest(), (c, d) -> {});
        final OpenAiLlmClient.StreamSummary s = ref.get();
        assertNotNull(s);
        assertEquals("chatcmpl-1", s.responseId);
        assertEquals("fp_abc", s.systemFingerprint);
        assertEquals("stop", s.finishReason);
        assertEquals(Integer.valueOf(3), s.promptTokens);
        assertEquals(Integer.valueOf(1), s.cachedTokens);
        assertEquals(Integer.valueOf(1), s.completionTokens);
        assertEquals(Integer.valueOf(2), s.reasoningTokens);
        assertEquals(Integer.valueOf(4), s.totalTokens);
    }

    @Test
    public void test_streamChat_doesNotDoubleFireDoneCallback() throws Exception {
        // The done callback fires once on the finish_reason chunk; [DONE] must NOT re-fire it.
        setupClientForMockServer();
        final java.util.List<Boolean> doneSignals = new ArrayList<>();
        final String body = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" + "data: [DONE]\n\n";
        mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
        client.streamChat(buildSimpleRequest(), (c, d) -> {
            if (d)
                doneSignals.add(Boolean.TRUE);
        });
        assertEquals("done callback must fire exactly once", 1, doneSignals.size());
    }

    @Test
    public void test_streamChat_finishReasonLength_warns() throws Exception {
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            final String body = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
                    + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" + "data: [DONE]\n\n";
            mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
            client.streamChat(buildSimpleRequest(), (c, d) -> {});
            assertTrue(app.messagesAt(org.apache.logging.log4j.Level.WARN)
                    .stream()
                    .anyMatch(
                            s -> s.contains("Stream finished abnormally") && s.contains("finishReason=length") && s.contains("chatcmpl-1")),
                    "expected abnormal-finish WARN with finishReason=length and id");
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_streamChat_finishReasonContentFilter_warns() throws Exception {
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            final String body = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},"
                    + "\"finish_reason\":\"content_filter\"}]}\n\ndata: [DONE]\n\n";
            mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
            client.streamChat(buildSimpleRequest(), (c, d) -> {});
            assertTrue(
                    app.messagesAt(org.apache.logging.log4j.Level.WARN).stream().anyMatch(s -> s.contains("finishReason=content_filter")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_streamChat_toleratesSseComments() throws Exception {
        setupClientForMockServer();
        final String body = ": ping\n\n: another\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" + "data: [DONE]\n\n";
        mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
        final StringBuilder collected = new StringBuilder();
        client.streamChat(buildSimpleRequest(), (c, d) -> collected.append(c));
        assertEquals("x", collected.toString());
    }

    @Test
    public void test_streamChat_initialRoleOnlyChunk_noEarlyChunk() throws Exception {
        // First chunk has only delta.role; must not invoke onChunk.
        setupClientForMockServer();
        final java.util.concurrent.atomic.AtomicInteger contentCalls = new java.util.concurrent.atomic.AtomicInteger();
        final String body = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" + "data: [DONE]\n\n";
        mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
        client.streamChat(buildSimpleRequest(), (c, d) -> {
            if (!c.isEmpty())
                contentCalls.incrementAndGet();
        });
        assertEquals(1, contentCalls.get());
    }

    @Test
    public void test_streamChat_logsHttpStatusAndContentTypeAtDebug() throws Exception {
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            mockServer.enqueue(
                    new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(simpleStreamSseBody()));
            client.streamChat(buildSimpleRequest(), (c, d) -> {});
            assertTrue(
                    app.messagesAt(org.apache.logging.log4j.Level.DEBUG)
                            .stream()
                            .anyMatch(s -> s.contains("statusCode=200") && s.contains("text/event-stream")),
                    "expected DEBUG line with statusCode=200 and contentType");
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_streamChat_warnsOnDeltaRefusal() throws Exception {
        // Structured-output refusal field: capture and WARN.
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            final String body = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"refusal\":\"I cannot help with that.\"},"
                    + "\"finish_reason\":null}]}\n\n"
                    + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" + "data: [DONE]\n\n";
            mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
            client.streamChat(buildSimpleRequest(), (c, d) -> {});
            assertTrue(
                    app.messagesAt(org.apache.logging.log4j.Level.WARN)
                            .stream()
                            .anyMatch(s -> s.contains("refusal") && s.contains("I cannot help with that")),
                    "WARN must include refusal text");
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_streamChat_refusalConcatenatedAcrossChunks() throws Exception {
        // refusal text split across multiple delta chunks must be concatenated in the WARN line
        // (matches the implementation: lastRefusal = lastRefusal + r).
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            final String body = "data: {\"id\":\"chatcmpl-r1\",\"choices\":[{\"delta\":{\"refusal\":\"I \"},\"finish_reason\":null}]}\n\n"
                    + "data: {\"id\":\"chatcmpl-r1\",\"choices\":[{\"delta\":{\"refusal\":\"cannot \"},\"finish_reason\":null}]}\n\n"
                    + "data: {\"id\":\"chatcmpl-r1\",\"choices\":[{\"delta\":{\"refusal\":\"help.\"},\"finish_reason\":null}]}\n\n"
                    + "data: {\"id\":\"chatcmpl-r1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" + "data: [DONE]\n\n";
            mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
            client.streamChat(buildSimpleRequest(), (c, d) -> {});
            assertTrue("WARN must contain joined refusal text", app.messagesAt(org.apache.logging.log4j.Level.WARN)
                    .stream()
                    .anyMatch(s -> s.contains("Stream refusal") && s.contains("refusal=I cannot help.") && s.contains("id=chatcmpl-r1")));
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_streamChat_usageOnlyChunkDoesNotInvokeOnChunk() throws Exception {
        // Usage-only chunk (choices=[]) carries token totals after finish_reason=stop. It must be
        // parsed (StreamSummary.totalTokens populated) but must NOT trigger onChunk again.
        setupClientForMockServer();
        final java.util.concurrent.atomic.AtomicReference<OpenAiLlmClient.StreamSummary> summaryRef =
                new java.util.concurrent.atomic.AtomicReference<>();
        client.setStreamSummaryConsumer(summaryRef::set);
        final AtomicInteger nonEmptyCalls = new AtomicInteger();
        final AtomicInteger doneCalls = new AtomicInteger();
        final String body = "data: {\"id\":\"chatcmpl-u\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-u\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                + "data: {\"id\":\"chatcmpl-u\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\n\n"
                + "data: [DONE]\n\n";
        mockServer.enqueue(new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(body));
        client.streamChat(buildSimpleRequest(), (content, done) -> {
            if (!content.isEmpty()) {
                nonEmptyCalls.incrementAndGet();
            }
            if (done) {
                doneCalls.incrementAndGet();
            }
        });
        assertEquals("only the content chunk should produce a non-empty onChunk call", 1, nonEmptyCalls.get());
        assertEquals("done callback must fire exactly once", 1, doneCalls.get());
        final OpenAiLlmClient.StreamSummary s = summaryRef.get();
        assertNotNull(s, "StreamSummary must be captured");
        assertEquals("totalTokens must come from the usage-only chunk", Integer.valueOf(7), s.totalTokens);
    }

    @Test
    public void test_streamChat_noUsageChunk_summaryHasNullTokens() throws Exception {
        // Compat backend that rejects stream_options: usage absent. StreamSummary tokens are null.
        setupClientForMockServer();
        client.setTestConfig("stream.include.usage", "false");
        final java.util.concurrent.atomic.AtomicReference<OpenAiLlmClient.StreamSummary> ref =
                new java.util.concurrent.atomic.AtomicReference<>();
        client.setStreamSummaryConsumer(ref::set);
        mockServer.enqueue(
                new MockResponse().setResponseCode(200).addHeader("Content-Type", "text/event-stream").setBody(simpleStreamSseBody()));
        client.streamChat(buildSimpleRequest(), (c, d) -> {});
        final OpenAiLlmClient.StreamSummary s = ref.get();
        assertNotNull(s);
        assertEquals("stop", s.finishReason);
        assertNull(s.promptTokens);
        assertNull(s.completionTokens);
        assertNull(s.totalTokens);
    }

    @Test
    public void test_streamChat_success() throws IOException {
        final String sseResponse = """
                data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}

                data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" World"}}]}

                data: {"id":"chatcmpl-123","choices":[{"delta":{},"finish_reason":"stop"}]}

                data: [DONE]

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(3, chunks.size());
        assertEquals("Hello", chunks.get(0));
        assertEquals(" World", chunks.get(1));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_multipleChunks() throws IOException {
        final String sseResponse = """
                data: {"choices":[{"delta":{"content":"A"}}]}

                data: {"choices":[{"delta":{"content":"B"}}]}

                data: {"choices":[{"delta":{"content":"C"}}]}

                data: {"choices":[{"delta":{"content":"D"}}]}

                data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

                data: [DONE]

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final StringBuilder content = new StringBuilder();
        final AtomicInteger chunkCount = new AtomicInteger(0);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String chunk, final boolean done) {
                content.append(chunk);
                chunkCount.incrementAndGet();
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals("ABCD", content.toString());
        assertTrue(chunkCount.get() >= 4);
    }

    @Test
    public void test_streamChat_errorResponse_withBody() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "message": "Insufficient quota",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(429).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();
        // 429 is retryable; cap attempts at 1 so this asserts surfacing rather than retry behavior.
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "0");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean errorReceived = new AtomicBoolean(false);

        try {
            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on error");
                }

                @Override
                public void onError(final Throwable error) {
                    errorReceived.set(true);
                }
            });
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            // After retry exhaustion the LlmException wraps an IOException whose message contains the status code.
            final Throwable cause = error.getCause();
            assertNotNull(cause);
            assertTrue(cause.getMessage() != null && cause.getMessage().contains("429"),
                    "expected status code 429 in cause message: " + cause.getMessage());
            assertTrue(errorReceived.get());
        }
    }

    @Test
    public void test_streamChat_errorResponse_serverError() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "message": "The server had an error while processing your request"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(500).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();
        // 500 is retryable; cap attempts at 1 to surface the failure immediately.
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "0");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean errorReceived = new AtomicBoolean(false);

        try {
            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on error");
                }

                @Override
                public void onError(final Throwable error) {
                    errorReceived.set(true);
                }
            });
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            final Throwable cause = error.getCause();
            assertNotNull(cause);
            assertTrue(cause.getMessage() != null && cause.getMessage().contains("500"),
                    "expected status code 500 in cause message: " + cause.getMessage());
            assertTrue(errorReceived.get());
        }
    }

    @Test
    public void test_streamChat_emptyBody() throws IOException {
        // Empty body with MockWebServer doesn't result in null body
        // Just verify that no chunks are received
        mockServer.enqueue(new MockResponse().setResponseCode(200).setBody("").addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        // No chunks should be received for empty body
        assertEquals(0, chunks.size());
    }

    @Test
    public void test_streamChat_doneMarkerOnly() throws IOException {
        final String sseResponse = """
                data: [DONE]

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean doneReceived = new AtomicBoolean(false);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_finishReasonWithoutDone() throws IOException {
        final String sseResponse = """
                data: {"choices":[{"delta":{"content":"Test"}}]}

                data: {"choices":[{"delta":{},"finish_reason":"length"}]}

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean doneReceived = new AtomicBoolean(false);
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(2, chunks.size());
        assertEquals("Test", chunks.get(0));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_malformedJson() throws IOException {
        final String sseResponse = """
                data: {"choices":[{"delta":{"content":"Hello"}}]}

                data: {invalid json}

                data: {"choices":[{"delta":{"content":" World"}}]}

                data: [DONE]

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                if (!content.isEmpty()) {
                    chunks.add(content);
                }
            }

            @Override
            public void onError(final Throwable error) {
                // Malformed JSON is logged but doesn't stop streaming
            }
        });

        // Should still receive valid chunks
        assertTrue(chunks.size() >= 2);
        assertTrue(chunks.contains("Hello"));
        assertTrue(chunks.contains(" World"));
    }

    @Test
    public void test_streamChat_ignoresBlankLines() throws IOException {
        final String sseResponse = """

                data: {"choices":[{"delta":{"content":"Test"}}]}



                data: [DONE]

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(2, chunks.size());
        assertEquals("Test", chunks.get(0));
    }

    @Test
    public void test_streamChat_ignoresNonDataLines() throws IOException {
        final String sseResponse = """
                event: message
                data: {"choices":[{"delta":{"content":"Test"}}]}

                : this is a comment
                data: [DONE]

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(2, chunks.size());
        assertEquals("Test", chunks.get(0));
    }

    // ========== destroy() tests ==========

    @Test
    public void test_destroy_closesHttpClient() {
        client.setTestTimeout(30000);
        client.init();
        assertNotNull(client.getHttpClient());
        client.destroy();
        // After destroy, calling getHttpClient() triggers re-init
        // Verify no exception is thrown during destroy
    }

    @Test
    public void test_destroy_beforeInit() {
        // destroy before init should not throw
        client.destroy();
    }

    // ========== Request format & Authorization header verification tests ==========

    @Test
    public void test_chat_verifyRequestFormat() throws Exception {
        final String responseJson = """
                {
                    "choices": [{
                        "message": {
                            "content": "Response"
                        }
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        client.chat(request);

        final RecordedRequest recorded = mockServer.takeRequest();
        assertEquals("POST", recorded.getMethod());
        assertEquals("/chat/completions", recorded.getPath());
        assertEquals("application/json; charset=UTF-8", recorded.getHeader("Content-Type"));

        // Verify Authorization header
        assertEquals("Bearer sk-test-key", recorded.getHeader("Authorization"));

        // Verify body contains expected structure
        final String body = recorded.getBody().readUtf8();
        assertTrue(body.contains("\"model\""));
        assertTrue(body.contains("\"messages\""));
        assertTrue(body.contains("\"stream\":false"));
        assertTrue(body.contains("Hello"));
    }

    @Test
    public void test_chat_verifyAuthorizationHeader() throws Exception {
        final String responseJson = """
                {
                    "choices": [{
                        "message": {
                            "content": "Test"
                        }
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        client.chat(request);

        final RecordedRequest recorded = mockServer.takeRequest();
        final String authHeader = recorded.getHeader("Authorization");
        assertNotNull(authHeader);
        assertTrue(authHeader.startsWith("Bearer "));
        assertEquals("Bearer sk-test-key", authHeader);
    }

    @Test
    public void test_streamChat_verifyRequestFormat() throws Exception {
        final String sseResponse = """
                data: {"choices":[{"delta":{"content":"Test"}}]}

                data: [DONE]

                """;

        mockServer.enqueue(new MockResponse().setBody(sseResponse).addHeader("Content-Type", "text/event-stream"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
            }

            @Override
            public void onError(final Throwable error) {
            }
        });

        final RecordedRequest recorded = mockServer.takeRequest();
        assertEquals("POST", recorded.getMethod());
        assertEquals("/chat/completions", recorded.getPath());

        // Verify Authorization header is present for streaming requests
        assertEquals("Bearer sk-test-key", recorded.getHeader("Authorization"));

        // Verify body has stream=true
        final String body = recorded.getBody().readUtf8();
        assertTrue(body.contains("\"stream\":true"));
    }

    // ========== checkAvailabilityNow() tests ==========

    @Test
    public void test_checkAvailabilityNow_success() throws Exception {
        mockServer.enqueue(new MockResponse().setBody("{\"data\":[]}").addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        assertTrue(client.checkAvailabilityNow());

        final RecordedRequest recorded = mockServer.takeRequest();
        assertEquals("GET", recorded.getMethod());
        assertEquals("/models", recorded.getPath());

        // Verify Authorization header is present for availability check
        assertEquals("Bearer sk-test-key", recorded.getHeader("Authorization"));
    }

    @Test
    public void test_checkAvailabilityNow_serverError() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(500).setBody("Internal Server Error"));

        setupClientForMockServer();

        assertFalse(client.checkAvailabilityNow());
    }

    @Test
    public void test_isAvailable_serverError() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(401).setBody("Unauthorized"));

        setupClientForMockServer();

        assertFalse(client.isAvailable());
    }

    @Test
    public void test_streamChat_serviceUnavailable() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("Service Unavailable"));

        setupClientForMockServer();
        // 503 is retryable; cap attempts at 1 to surface the failure immediately.
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "0");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean errorReceived = new AtomicBoolean(false);

        try {
            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on error");
                }

                @Override
                public void onError(final Throwable error) {
                    errorReceived.set(true);
                }
            });
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            final Throwable cause = error.getCause();
            assertNotNull(cause);
            assertTrue(cause.getMessage() != null && cause.getMessage().contains("503"),
                    "expected status code 503 in cause message: " + cause.getMessage());
            assertTrue(errorReceived.get());
        }
    }

    @Test
    public void test_chat_serviceUnavailable() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("Service Unavailable"));

        setupClientForMockServer();
        // 503 is retryable; cap attempts at 1 so this asserts surfacing rather than retry behavior.
        client.setTestConfig("retry.max", "1");
        client.setTestConfig("retry.base.delay.ms", "0");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            final Throwable cause = error.getCause();
            assertNotNull(cause);
            assertTrue("expected status code 503 in cause message: " + cause.getMessage(),
                    cause.getMessage() != null && cause.getMessage().contains("503"));
        }
    }

    // ========== useMaxCompletionTokens tests ==========

    @Test
    public void test_useMaxCompletionTokens_legacyModels() {
        assertFalse(client.useMaxCompletionTokens("gpt-3.5-turbo"));
        assertFalse(client.useMaxCompletionTokens("gpt-4"));
        assertFalse(client.useMaxCompletionTokens("gpt-4o"));
        assertFalse(client.useMaxCompletionTokens("gpt-4o-mini"));
        assertFalse(client.useMaxCompletionTokens("gpt-4-turbo"));
    }

    @Test
    public void test_useMaxCompletionTokens_newerModels() {
        assertTrue(client.useMaxCompletionTokens("o1"));
        assertTrue(client.useMaxCompletionTokens("o1-mini"));
        assertTrue(client.useMaxCompletionTokens("o1-preview"));
        assertTrue(client.useMaxCompletionTokens("o3"));
        assertTrue(client.useMaxCompletionTokens("o3-mini"));
        assertTrue(client.useMaxCompletionTokens("o4-mini"));
        assertTrue(client.useMaxCompletionTokens("gpt-5"));
        assertTrue(client.useMaxCompletionTokens("gpt-5-mini"));
        assertTrue(client.useMaxCompletionTokens("gpt-5.1"));
        assertTrue(client.useMaxCompletionTokens("gpt-5.2"));
    }

    @Test
    public void test_useMaxCompletionTokens_blankOrNull() {
        assertFalse(client.useMaxCompletionTokens(null));
        assertFalse(client.useMaxCompletionTokens(""));
        assertFalse(client.useMaxCompletionTokens("  "));
    }

    @Test
    public void test_buildRequestBody_legacyModel_usesMaxTokens() {
        client.setTestModel("gpt-4");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-4", body.get("model"));
        assertEquals(4096, body.get("max_tokens"));
        assertNull(body.get("max_completion_tokens"));
    }

    @Test
    public void test_buildRequestBody_requestModelOverride() {
        client.setTestModel("gpt-4");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setModel("o3-mini").setMaxTokens(2048).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("o3-mini", body.get("model"));
        assertEquals(2048, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    // ========== useMaxCompletionTokens: boundary/edge cases ==========

    @Test
    public void test_useMaxCompletionTokens_o1Variants() {
        assertTrue(client.useMaxCompletionTokens("o1"));
        assertTrue(client.useMaxCompletionTokens("o1-mini"));
        assertTrue(client.useMaxCompletionTokens("o1-preview"));
        assertTrue(client.useMaxCompletionTokens("o1-2024-12-17"));
        assertTrue(client.useMaxCompletionTokens("o1-pro"));
    }

    @Test
    public void test_useMaxCompletionTokens_o3Variants() {
        assertTrue(client.useMaxCompletionTokens("o3"));
        assertTrue(client.useMaxCompletionTokens("o3-mini"));
        assertTrue(client.useMaxCompletionTokens("o3-mini-2025-01-31"));
        assertTrue(client.useMaxCompletionTokens("o3-pro"));
    }

    @Test
    public void test_useMaxCompletionTokens_o4Variants() {
        assertTrue(client.useMaxCompletionTokens("o4-mini"));
        assertTrue(client.useMaxCompletionTokens("o4-mini-2025-04-16"));
    }

    @Test
    public void test_useMaxCompletionTokens_gpt5Variants() {
        assertTrue(client.useMaxCompletionTokens("gpt-5"));
        assertTrue(client.useMaxCompletionTokens("gpt-5-mini"));
        assertTrue(client.useMaxCompletionTokens("gpt-5-turbo"));
        assertTrue(client.useMaxCompletionTokens("gpt-5.1"));
        assertTrue(client.useMaxCompletionTokens("gpt-5.2"));
        assertTrue(client.useMaxCompletionTokens("gpt-5-2025-06-01"));
    }

    @Test
    public void test_useMaxCompletionTokens_gpt4FamilyReturnsFalse() {
        assertFalse(client.useMaxCompletionTokens("gpt-4"));
        assertFalse(client.useMaxCompletionTokens("gpt-4o"));
        assertFalse(client.useMaxCompletionTokens("gpt-4o-mini"));
        assertFalse(client.useMaxCompletionTokens("gpt-4-turbo"));
        assertFalse(client.useMaxCompletionTokens("gpt-4-turbo-2024-04-09"));
        assertFalse(client.useMaxCompletionTokens("gpt-4-0613"));
        assertFalse(client.useMaxCompletionTokens("gpt-4-1106-preview"));
    }

    @Test
    public void test_useMaxCompletionTokens_gpt35ReturnsFalse() {
        assertFalse(client.useMaxCompletionTokens("gpt-3.5-turbo"));
        assertFalse(client.useMaxCompletionTokens("gpt-3.5-turbo-0125"));
        assertFalse(client.useMaxCompletionTokens("gpt-3.5-turbo-16k"));
    }

    @Test
    public void test_useMaxCompletionTokens_chatgptModelReturnsFalse() {
        assertFalse(client.useMaxCompletionTokens("chatgpt-4o-latest"));
    }

    @Test
    public void test_useMaxCompletionTokens_unknownModelReturnsFalse() {
        assertFalse(client.useMaxCompletionTokens("some-custom-model"));
        assertFalse(client.useMaxCompletionTokens("my-fine-tuned-model"));
        assertFalse(client.useMaxCompletionTokens("llama-3"));
    }

    // ========== supportsTemperature tests ==========

    @Test
    public void test_supportsTemperature_legacyModels() {
        assertTrue(client.supportsTemperature("gpt-3.5-turbo"));
        assertTrue(client.supportsTemperature("gpt-4"));
        assertTrue(client.supportsTemperature("gpt-4o"));
        assertTrue(client.supportsTemperature("gpt-4o-mini"));
        assertTrue(client.supportsTemperature("gpt-4-turbo"));
    }

    @Test
    public void test_supportsTemperature_newerModels() {
        assertFalse(client.supportsTemperature("o1"));
        assertFalse(client.supportsTemperature("o1-mini"));
        assertFalse(client.supportsTemperature("o1-preview"));
        assertFalse(client.supportsTemperature("o3"));
        assertFalse(client.supportsTemperature("o3-mini"));
        assertFalse(client.supportsTemperature("o4-mini"));
        assertFalse(client.supportsTemperature("gpt-5"));
        assertFalse(client.supportsTemperature("gpt-5-mini"));
        assertFalse(client.supportsTemperature("gpt-5.1"));
        assertFalse(client.supportsTemperature("gpt-5.2"));
    }

    @Test
    public void test_supportsTemperature_blankOrNull() {
        assertTrue(client.supportsTemperature(null));
        assertTrue(client.supportsTemperature(""));
        assertTrue(client.supportsTemperature("  "));
    }

    @Test
    public void test_supportsTemperature_unknownModelReturnsTrue() {
        assertTrue(client.supportsTemperature("some-custom-model"));
        assertTrue(client.supportsTemperature("my-fine-tuned-model"));
        assertTrue(client.supportsTemperature("llama-3"));
    }

    // ========== isReasoningModel tests ==========

    @Test
    public void test_isReasoningModel_legacyModels() {
        assertFalse(client.isReasoningModel("gpt-3.5-turbo"));
        assertFalse(client.isReasoningModel("gpt-4"));
        assertFalse(client.isReasoningModel("gpt-4o"));
        assertFalse(client.isReasoningModel("gpt-4o-mini"));
        assertFalse(client.isReasoningModel("gpt-4-turbo"));
    }

    @Test
    public void test_isReasoningModel_reasoningModels() {
        assertTrue(client.isReasoningModel("o1"));
        assertTrue(client.isReasoningModel("o1-mini"));
        assertTrue(client.isReasoningModel("o1-preview"));
        assertTrue(client.isReasoningModel("o3"));
        assertTrue(client.isReasoningModel("o3-mini"));
        assertTrue(client.isReasoningModel("o4-mini"));
        assertTrue(client.isReasoningModel("gpt-5"));
        assertTrue(client.isReasoningModel("gpt-5-mini"));
        assertTrue(client.isReasoningModel("gpt-5.1"));
    }

    @Test
    public void test_isReasoningModel_blankOrNull() {
        assertFalse(client.isReasoningModel(null));
        assertFalse(client.isReasoningModel(""));
        assertFalse(client.isReasoningModel("  "));
    }

    // ========== buildRequestBody: reasoning_effort integration ==========

    @Test
    public void test_buildRequestBody_reasoningModel_withReasoningEffortLow() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(500);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("reasoning_effort", "low").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("low", body.get("reasoning_effort"));
    }

    @Test
    public void test_buildRequestBody_reasoningModel_withoutReasoningEffort_noReasoningEffort() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("reasoning_effort"));
    }

    @Test
    public void test_buildRequestBody_legacyModel_withReasoningEffortParam_noReasoningEffort() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("reasoning_effort", "low").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("reasoning_effort"));
    }

    @Test
    public void test_buildRequestBody_o1Model_withReasoningEffortLow() {
        client.setTestModel("o1");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(500);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("reasoning_effort", "low").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("low", body.get("reasoning_effort"));
    }

    // ========== buildRequestBody: temperature integration ==========

    @Test
    public void test_buildRequestBody_gpt5MiniModel_excludesTemperature() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("temperature"));
    }

    @Test
    public void test_buildRequestBody_gpt5MiniModel_excludesTemperatureEvenWithRequestTemperature() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setTemperature(0.5).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("temperature"));
    }

    @Test
    public void test_buildRequestBody_gpt4oModel_includesTemperature() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setTemperature(0.7).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals(0.7, body.get("temperature"));
    }

    @Test
    public void test_buildRequestBody_o1Model_excludesTemperature() {
        client.setTestModel("o1");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(8192);

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("temperature"));
    }

    // ========== buildRequestBody: max_tokens key selection integration ==========

    @Test
    public void test_buildRequestBody_o1Model_usesMaxCompletionTokens() {
        client.setTestModel("o1");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(8192);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(8192).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("o1", body.get("model"));
        assertEquals(8192, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_o3MiniModel_usesMaxCompletionTokens() {
        client.setTestModel("o3-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("o3-mini", body.get("model"));
        assertEquals(4096, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_o4MiniModel_usesMaxCompletionTokens() {
        client.setTestModel("o4-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("o4-mini", body.get("model"));
        assertEquals(4096, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_gpt4oModel_usesMaxTokens() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-4o", body.get("model"));
        assertEquals(4096, body.get("max_tokens"));
        assertNull(body.get("max_completion_tokens"));
    }

    @Test
    public void test_buildRequestBody_gpt35Model_usesMaxTokens() {
        client.setTestModel("gpt-3.5-turbo");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(2048);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(2048).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-3.5-turbo", body.get("model"));
        assertEquals(2048, body.get("max_tokens"));
        assertNull(body.get("max_completion_tokens"));
    }

    @Test
    public void test_buildRequestBody_legacyDefaultModel_requestOverridesToNewer() {
        client.setTestModel("gpt-4");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setModel("gpt-5-mini").setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-5-mini", body.get("model"));
        assertEquals(4096, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_newerDefaultModel_requestOverridesToLegacy() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setModel("gpt-4o").setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-4o", body.get("model"));
        assertEquals(4096, body.get("max_tokens"));
        assertNull(body.get("max_completion_tokens"));
    }

    @Test
    public void test_buildRequestBody_newerModel_withRequestMaxTokens() {
        client.setTestModel("o3-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(1024).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("o3-mini", body.get("model"));
        assertEquals(1024, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_legacyModel_withRequestMaxTokens() {
        client.setTestModel("gpt-4");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(512).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("gpt-4", body.get("model"));
        assertEquals(512, body.get("max_tokens"));
        assertNull(body.get("max_completion_tokens"));
    }

    @Test
    public void test_buildRequestBody_streaming_newerModel_usesMaxCompletionTokens() {
        client.setTestModel("gpt-5");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, true);

        assertEquals(true, body.get("stream"));
        assertEquals("gpt-5", body.get("model"));
        assertEquals(4096, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_streaming_legacyModel_usesMaxTokens() {
        client.setTestModel("gpt-4");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, true);

        assertEquals(true, body.get("stream"));
        assertEquals("gpt-4", body.get("model"));
        assertEquals(4096, body.get("max_tokens"));
        assertNull(body.get("max_completion_tokens"));
    }

    @Test
    public void test_buildRequestBody_blankRequestModel_fallsBackToDefaultKeySelection() {
        client.setTestModel("o1");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setModel("").setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("o1", body.get("model"));
        assertEquals(4096, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_nullRequestModel_fallsBackToDefaultKeySelection() {
        client.setTestModel("o3-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(2048);

        final LlmChatRequest request = new LlmChatRequest().setModel(null).setMaxTokens(2048).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("o3-mini", body.get("model"));
        assertEquals(2048, body.get("max_completion_tokens"));
        assertNull(body.get("max_tokens"));
    }

    @Test
    public void test_buildRequestBody_bodyContainsExactlyOneMaxTokensKey() {
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertTrue(body.containsKey("max_completion_tokens"));
        assertFalse(body.containsKey("max_tokens"));

        // Legacy model should have the opposite
        client.setTestModel("gpt-4");
        final Map<String, Object> body2 = client.buildRequestBody(request, false);

        assertTrue(body2.containsKey("max_tokens"));
        assertFalse(body2.containsKey("max_completion_tokens"));
    }

    // ========== applyDefaultParams tests ==========

    @Test
    public void test_applyDefaultParams_intent() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "intent");
        assertEquals(0.1, request.getTemperature());
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_evaluation() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "evaluation");
        assertEquals(0.1, request.getTemperature());
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_unclear() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "unclear");
        assertEquals(0.7, request.getTemperature());
        assertEquals(Integer.valueOf(512), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_noresults() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "noresults");
        assertEquals(0.7, request.getTemperature());
        assertEquals(Integer.valueOf(512), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_docnotfound() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "docnotfound");
        assertEquals(0.7, request.getTemperature());
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_direct() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "direct");
        assertEquals(0.7, request.getTemperature());
        assertEquals(Integer.valueOf(1024), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_faq() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "faq");
        assertEquals(0.7, request.getTemperature());
        assertEquals(Integer.valueOf(1024), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_answer() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "answer");
        assertEquals(0.5, request.getTemperature());
        assertEquals(Integer.valueOf(2048), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_summary() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "summary");
        assertEquals(0.3, request.getTemperature());
        assertEquals(Integer.valueOf(2048), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_unknownType() {
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "unknown");
        assertNull(request.getTemperature());
        assertNull(request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_doesNotOverrideExisting() {
        final LlmChatRequest request = new LlmChatRequest();
        request.setTemperature(0.9);
        request.setMaxTokens(100);
        client.applyDefaultParams(request, "intent");
        assertEquals(0.9, request.getTemperature());
        assertEquals(Integer.valueOf(100), request.getMaxTokens());
    }

    // ========== applyDefaultParams: reasoning model tests ==========

    @Test
    public void test_applyDefaultParams_reasoningModel_multipliesTokens_intent() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "intent");
        // Default 256 * 4 = 1024
        assertEquals(Integer.valueOf(1024), request.getMaxTokens());
        assertEquals("low", request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_multipliesTokens_evaluation() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "evaluation");
        assertEquals(Integer.valueOf(1024), request.getMaxTokens());
        assertEquals("low", request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_multipliesTokens_unclear() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "unclear");
        // Default 512 * 4 = 2048
        assertEquals(Integer.valueOf(2048), request.getMaxTokens());
        assertEquals("low", request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_multipliesTokens_answer() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "answer");
        // Default 2048 * 4 = 8192
        assertEquals(Integer.valueOf(8192), request.getMaxTokens());
        // answer should NOT set reasoning_effort
        assertNull(request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_multipliesTokens_direct() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "direct");
        // Default 1024 * 4 = 4096
        assertEquals(Integer.valueOf(4096), request.getMaxTokens());
        assertNull(request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_doesNotOverrideUserMaxTokens() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        request.setMaxTokens(500);
        client.applyDefaultParams(request, "intent");
        // User set 500, should NOT be multiplied
        assertEquals(Integer.valueOf(500), request.getMaxTokens());
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_doesNotOverrideUserReasoningEffort() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        request.putExtraParam("reasoning_effort", "high");
        client.applyDefaultParams(request, "intent");
        // User set "high", should NOT be overridden
        assertEquals("high", request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_nonReasoningModel_noMultiplier() {
        client.setTestModel("gpt-4o");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "intent");
        // Non-reasoning model: default 256, no multiplier
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
        assertNull(request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_o1Model_multipliesTokens() {
        client.setTestModel("o1");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "intent");
        assertEquals(Integer.valueOf(1024), request.getMaxTokens());
        assertEquals("low", request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_docnotfound() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "docnotfound");
        assertEquals(Integer.valueOf(1024), request.getMaxTokens());
        assertEquals("low", request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_noresults() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "noresults");
        assertEquals(Integer.valueOf(2048), request.getMaxTokens());
        assertEquals("low", request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_summary() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "summary");
        assertEquals(Integer.valueOf(8192), request.getMaxTokens());
        assertNull(request.getExtraParam("reasoning_effort"));
    }

    @Test
    public void test_applyDefaultParams_reasoningModel_faq() {
        client.setTestModel("gpt-5-mini");
        final LlmChatRequest request = new LlmChatRequest();
        client.applyDefaultParams(request, "faq");
        assertEquals(Integer.valueOf(4096), request.getMaxTokens());
        assertNull(request.getExtraParam("reasoning_effort"));
    }

    // ========== buildRequestBody: top_p, frequency_penalty, presence_penalty tests ==========

    @Test
    public void test_buildRequestBody_withTopP() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("top_p", "0.9").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals(0.9, body.get("top_p"));
    }

    @Test
    public void test_buildRequestBody_withFrequencyPenalty() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("frequency_penalty", "0.5").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals(0.5, body.get("frequency_penalty"));
    }

    @Test
    public void test_buildRequestBody_withPresencePenalty() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("presence_penalty", "0.3").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals(0.3, body.get("presence_penalty"));
    }

    @Test
    public void test_buildRequestBody_withAllExtraParams() {
        // gpt-5-mini is a reasoning model, so only reasoning_effort survives here. This test
        // previously asserted that all four extra params were sent, which is what the live API
        // rejects with 400 "Unsupported parameter" - it was green against a body the server
        // would refuse. The sampling params are covered on a non-reasoning model below.
        client.setTestModel("gpt-5-mini");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("reasoning_effort", "low")
                .putExtraParam("top_p", "0.95")
                .putExtraParam("frequency_penalty", "0.2")
                .putExtraParam("presence_penalty", "0.1")
                .addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertEquals("low", body.get("reasoning_effort"));
        assertNull(body.get("top_p"), "top_p is unsupported by reasoning models");
        assertNull(body.get("frequency_penalty"), "frequency_penalty is unsupported by reasoning models");
        assertNull(body.get("presence_penalty"), "presence_penalty is unsupported by reasoning models");
    }

    @Test
    public void test_buildRequestBody_withAllExtraParamsOnNonReasoningModel() {
        // The counterpart of the above: on a name outside the reasoning families the sampling
        // params are still forwarded, and reasoning_effort is the one that drops out.
        client.setTestModel("my-gateway-deployment");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().putExtraParam("reasoning_effort", "low")
                .putExtraParam("top_p", "0.95")
                .putExtraParam("frequency_penalty", "0.2")
                .putExtraParam("presence_penalty", "0.1")
                .addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("reasoning_effort"), "reasoning_effort is only sent to reasoning models");
        assertEquals(0.95, body.get("top_p"));
        assertEquals(0.2, body.get("frequency_penalty"));
        assertEquals(0.1, body.get("presence_penalty"));
    }

    @Test
    public void test_buildRequestBody_withoutExtraParams() {
        client.setTestModel("gpt-4o");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        assertNull(body.get("top_p"));
        assertNull(body.get("frequency_penalty"));
        assertNull(body.get("presence_penalty"));
    }

    // ========== Budget method tests ==========

    @Test
    public void test_getHistoryMaxChars_default() {
        assertEquals(8000, client.testGetHistoryMaxChars());
    }

    @Test
    public void test_getIntentHistoryMaxMessages_default() {
        assertEquals(8, client.testGetIntentHistoryMaxMessages());
    }

    @Test
    public void test_getHistoryAssistantMaxChars_default() {
        assertEquals(800, client.testGetHistoryAssistantMaxChars());
    }

    // ========== isAbnormalFinishReason tests ==========

    @Test
    public void test_isAbnormalFinishReason_null() {
        assertFalse(OpenAiLlmClient.isAbnormalFinishReason(null));
    }

    @Test
    public void test_isAbnormalFinishReason_stop() {
        assertFalse(OpenAiLlmClient.isAbnormalFinishReason("stop"));
    }

    @Test
    public void test_isAbnormalFinishReason_length() {
        assertTrue(OpenAiLlmClient.isAbnormalFinishReason("length"));
    }

    @Test
    public void test_isAbnormalFinishReason_contentFilter() {
        assertTrue(OpenAiLlmClient.isAbnormalFinishReason("content_filter"));
    }

    @Test
    public void test_isAbnormalFinishReason_toolCalls() {
        assertTrue(OpenAiLlmClient.isAbnormalFinishReason("tool_calls"));
    }

    @Test
    public void test_isAbnormalFinishReason_functionCall() {
        assertTrue(OpenAiLlmClient.isAbnormalFinishReason("function_call"));
    }

    @Test
    public void test_isAbnormalFinishReason_unknown() {
        assertTrue(OpenAiLlmClient.isAbnormalFinishReason("error"));
        assertTrue(OpenAiLlmClient.isAbnormalFinishReason("future_reason"));
    }

    @Test
    public void test_isAbnormalFinishReason_blank() {
        assertFalse(OpenAiLlmClient.isAbnormalFinishReason(""));
        assertFalse(OpenAiLlmClient.isAbnormalFinishReason("  "));
    }

    // ========== isRetryableStatus tests ==========

    @Test
    public void test_isRetryableStatus_429() {
        assertTrue(OpenAiRetry.isRetryableStatus(429));
    }

    @Test
    public void test_isRetryableStatus_500() {
        assertTrue(OpenAiRetry.isRetryableStatus(500));
    }

    @Test
    public void test_isRetryableStatus_502() {
        assertTrue(OpenAiRetry.isRetryableStatus(502));
    }

    @Test
    public void test_isRetryableStatus_503() {
        assertTrue(OpenAiRetry.isRetryableStatus(503));
    }

    @Test
    public void test_isRetryableStatus_504() {
        assertTrue(OpenAiRetry.isRetryableStatus(504));
    }

    @Test
    public void test_isRetryableStatus_400() {
        assertFalse(OpenAiRetry.isRetryableStatus(400));
    }

    @Test
    public void test_isRetryableStatus_401() {
        assertFalse(OpenAiRetry.isRetryableStatus(401));
    }

    @Test
    public void test_isRetryableStatus_404() {
        assertFalse(OpenAiRetry.isRetryableStatus(404));
    }

    @Test
    public void test_isRetryableStatus_408_notRetried() {
        // OpenAI rarely returns 408; if it ever appears in production logs we'll add it.
        assertFalse(OpenAiRetry.isRetryableStatus(408));
    }

    @Test
    public void test_isRetryableStatus_200() {
        assertFalse(OpenAiRetry.isRetryableStatus(200));
    }

    // ========== parseRetryAfterSeconds tests ==========

    @Test
    public void test_parseRetryAfterSeconds_integer() {
        assertEquals(5L, OpenAiRetry.parseRetryAfterSeconds("5"));
    }

    @Test
    public void test_parseRetryAfterSeconds_zero() {
        assertEquals(0L, OpenAiRetry.parseRetryAfterSeconds("0"));
    }

    @Test
    public void test_parseRetryAfterSeconds_largeClamped() {
        assertEquals(600L, OpenAiRetry.parseRetryAfterSeconds("3600"));
    }

    @Test
    public void test_parseRetryAfterSeconds_negative() {
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds("-1"));
    }

    @Test
    public void test_parseRetryAfterSeconds_null() {
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds(null));
    }

    @Test
    public void test_parseRetryAfterSeconds_blank() {
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds(""));
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds("  "));
    }

    @Test
    public void test_parseRetryAfterSeconds_httpDate() {
        // HTTP-date format intentionally unsupported; caller falls back to backoff.
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds("Wed, 21 Oct 2026 07:28:00 GMT"));
    }

    @Test
    public void test_parseRetryAfterSeconds_decimal() {
        assertEquals(-1L, OpenAiRetry.parseRetryAfterSeconds("1.5"));
    }

    @Test
    public void test_parseRetryAfterSeconds_whitespacePadded() {
        assertEquals(7L, OpenAiRetry.parseRetryAfterSeconds("  7  "));
    }

    // ========== maskCredentialInUrl tests ==========

    @Test
    public void test_maskCredentialInUrl_apiKey() {
        assertEquals("https://proxy/v1/chat?api_key=***", OpenAiLlmClient.maskCredentialInUrl("https://proxy/v1/chat?api_key=sk-secret"));
    }

    @Test
    public void test_maskCredentialInUrl_key() {
        assertEquals("https://proxy/v1/chat?key=***&model=x",
                OpenAiLlmClient.maskCredentialInUrl("https://proxy/v1/chat?key=secret&model=x"));
    }

    @Test
    public void test_maskCredentialInUrl_token() {
        assertEquals("https://proxy/v1/chat?model=x&token=***",
                OpenAiLlmClient.maskCredentialInUrl("https://proxy/v1/chat?model=x&token=abc"));
    }

    @Test
    public void test_maskCredentialInUrl_apiKeyMixedCase() {
        assertEquals("https://proxy/v1/chat?Api-Key=***", OpenAiLlmClient.maskCredentialInUrl("https://proxy/v1/chat?Api-Key=secret"));
    }

    @Test
    public void test_maskCredentialInUrl_noCredential() {
        assertEquals("https://api.openai.com/v1/chat/completions",
                OpenAiLlmClient.maskCredentialInUrl("https://api.openai.com/v1/chat/completions"));
    }

    @Test
    public void test_maskCredentialInUrl_null() {
        assertNull(OpenAiLlmClient.maskCredentialInUrl(null));
    }

    @Test
    public void test_maskCredentialInUrl_multipleCredentialParams() {
        // Proxies sometimes carry several credential-bearing query params (api_key, access_token, token).
        // Every credential value must be replaced with *** while non-credential params are preserved.
        final String masked =
                OpenAiLlmClient.maskCredentialInUrl("https://proxy/v1/chat?api_key=secret1&model=gpt&access_token=secret2&token=secret3");
        assertTrue("api_key must be masked: " + masked, masked.contains("api_key=***"));
        assertTrue("access_token must be masked: " + masked, masked.contains("access_token=***"));
        assertTrue("token must be masked: " + masked, masked.contains("token=***"));
        assertFalse("api_key value must not leak: " + masked, masked.contains("secret1"));
        assertFalse("access_token value must not leak: " + masked, masked.contains("secret2"));
        assertFalse("token value must not leak: " + masked, masked.contains("secret3"));
        assertTrue("non-credential param must be preserved: " + masked, masked.contains("model=gpt"));
    }

    @Test
    public void test_maskCredentialInUrl_masksUserInfo() {
        // Defensive rule: HttpClient rejects a userinfo-bearing request URI outright, so such a
        // URL can never issue a request. The rule only keeps a mistyped credential in that
        // position out of the log, since the query-parameter pattern never matches it.
        assertEquals("https://***:***@gw.example.com/v1", OpenAiLlmClient.maskCredentialInUrl("https://user:pass@gw.example.com/v1"));
        assertEquals("http://***:***@gw.example.com/v1/chat/completions",
                OpenAiLlmClient.maskCredentialInUrl("http://user:pass@gw.example.com/v1/chat/completions"));
        // Both rules must apply to the same URL.
        assertEquals("https://***:***@gw.example.com/v1?api-key=***",
                OpenAiLlmClient.maskCredentialInUrl("https://user:pass@gw.example.com/v1?api-key=secret"));
    }

    @Test
    public void test_maskCredentialInUrl_cleanUrlUnchanged() {
        // No credentials anywhere: the URL must survive byte-for-byte. In particular the userinfo
        // rule must not fire on a port-bearing authority ("host:8443") or on a path colon.
        assertEquals("https://gw.example.com:8443/v1", OpenAiLlmClient.maskCredentialInUrl("https://gw.example.com:8443/v1"));
        assertEquals("https://gw.example.com:8443/v1/chat/completions",
                OpenAiLlmClient.maskCredentialInUrl("https://gw.example.com:8443/v1/chat/completions"));
        assertEquals("https://h/v1/chat/completions?api-version=2024-02-01",
                OpenAiLlmClient.maskCredentialInUrl("https://h/v1/chat/completions?api-version=2024-02-01"));
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
    public void test_chat_malformedApiUrlDoesNotLeakCredential() {
        client.setTestApiUrl(MALFORMED_CREDENTIAL_URL);
        client.setTestApiKey("sk-test-key");
        client.init();
        final ListAppender app = attachLogCapture();
        try {
            client.chat(new LlmChatRequest().addUserMessage("Hello"));
            fail("expected LlmException for a malformed api.url");
        } catch (final LlmException e) {
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
    public void test_streamChat_malformedApiUrlDoesNotLeakCredential() {
        client.setTestApiUrl(MALFORMED_CREDENTIAL_URL);
        client.setTestApiKey("sk-test-key");
        client.init();
        final List<Throwable> callbackErrors = new ArrayList<>();
        final ListAppender app = attachLogCapture();
        try {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new LlmStreamCallback() {
                @Override
                public void onChunk(final String chunk, final boolean done) {
                    // not reached
                }

                @Override
                public void onError(final Throwable t) {
                    callbackErrors.add(t);
                }
            });
            fail("expected LlmException for a malformed api.url");
        } catch (final LlmException e) {
            final String propagated = renderThrowable(e);
            assertFalse("propagated exception must not carry the credential: " + propagated, propagated.contains(RAW_CREDENTIAL));
        } finally {
            final String logged = app.rendered();
            detachLogCapture(app);
            assertFalse("log must not carry the credential: " + logged, logged.contains(RAW_CREDENTIAL));
        }
        assertEquals(1, callbackErrors.size());
        final String viaCallback = renderThrowable(callbackErrors.get(0));
        assertFalse("callback error must not carry the credential: " + viaCallback, viaCallback.contains(RAW_CREDENTIAL));
    }

    @Test
    public void test_isAvailable_malformedApiUrlDoesNotLeakCredential() {
        client.setTestApiUrl(MALFORMED_CREDENTIAL_URL);
        client.setTestApiKey("sk-test-key");
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
    public void test_streamChat_malformedUserInfoNotEchoedByException() {
        client.setTestApiUrl(MALFORMED_USERINFO_URL);
        client.setTestApiKey("sk-test-key");
        client.init();
        final List<Throwable> callbackErrors = new ArrayList<>();
        final ListAppender app = attachLogCapture();
        try {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new LlmStreamCallback() {
                @Override
                public void onChunk(final String chunk, final boolean done) {
                    // not reached
                }

                @Override
                public void onError(final Throwable t) {
                    callbackErrors.add(t);
                }
            });
            fail("expected LlmException for a malformed api.url");
        } catch (final LlmException e) {
            final String propagated = renderThrowable(e);
            assertFalse("propagated cause chain must not carry the credential: " + propagated,
                    propagated.contains(RAW_USERINFO_CREDENTIAL));
        } finally {
            final String thrown = app.renderedThrowables();
            detachLogCapture(app);
            assertFalse("logged throwable must not carry the credential: " + thrown, thrown.contains(RAW_USERINFO_CREDENTIAL));
        }
        assertEquals(1, callbackErrors.size());
        final String viaCallback = renderThrowable(callbackErrors.get(0));
        assertFalse("callback error must not carry the credential: " + viaCallback, viaCallback.contains(RAW_USERINFO_CREDENTIAL));
    }

    // ========== userinfo-bearing api.url is refused before any request ==========

    /**
     * A {@code rag.llm.openai.api.url} carrying a userinfo credential. RFC 9110 forbids userinfo
     * in an http/https target URI and HttpClient enforces that unconditionally, so this value can
     * never issue a request; OpenAI-compatible gateways authenticate with {@code Authorization:
     * Bearer}, and an endpoint behind an authenticating proxy is configured through
     * {@code http.proxy.*}. It is therefore an operator error with a supported alternative, and the
     * client must say so instead of failing opaquely at request time.
     */
    private static final String USERINFO_API_URL = "https://user:s3cr3tUserinfo@gw.example.com/v1";

    /** The userinfo credential that must appear in no message, throwable, cause or callback error. */
    private static final String USERINFO_CREDENTIAL = "s3cr3tUserinfo";

    /** The supported alternative the refusal must name. */
    private static final String PROXY_USERNAME_KEY = "http.proxy.username";

    /** The supported alternative the refusal must name. */
    private static final String PROXY_PASSWORD_KEY = "http.proxy.password";

    @Test
    public void test_isAvailable_userInfoApiUrlReportsUnavailableWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        final ListAppender app = attachLogCapture();
        try {
            assertFalse("a userinfo-bearing api.url can never issue a request", client.isAvailable());
            final List<String> errors = app.messagesAt(org.apache.logging.log4j.Level.ERROR);
            assertEquals("the refusal must be reported at ERROR: " + app.messages(), 1, errors.size());
            final String error = errors.get(0);
            assertTrue("the offending configuration key must be named: " + error, error.contains("rag.llm.openai.api.url"));
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
        final ListAppender app = attachLogCapture();
        try {
            assertFalse(client.isAvailable());
            assertFalse(client.isAvailable());
            assertFalse(client.isAvailable());
            assertEquals("the remedy must be stated once, not on every availability check", 1,
                    app.messagesAt(org.apache.logging.log4j.Level.ERROR).size());
        } finally {
            detachLogCapture(app);
        }
    }

    @Test
    public void test_isAvailable_portBearingApiUrlIsUnaffected() throws IOException {
        // "host:port" is a port, not userinfo: the refusal must not fire on an ordinary
        // port-bearing gateway URL. The mock server supplies a real port-bearing authority.
        mockServer.enqueue(new MockResponse().setBody("{\"data\":[]}").addHeader("Content-Type", "application/json"));
        setupClientForMockServer();
        final ListAppender app = attachLogCapture();
        try {
            assertTrue("a port-bearing api.url must still be usable", client.isAvailable());
            assertEquals("no refusal may be reported for a port-bearing URL: " + app.messages(), 0,
                    app.messagesAt(org.apache.logging.log4j.Level.ERROR).size());
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
        client.testUpdateAvailability();
        assertFalse("the client must report itself unavailable", client.isAvailable());
    }

    @Test
    public void test_chat_userInfoApiUrlIsRefusedWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        client.init();
        final ListAppender app = attachLogCapture();
        try {
            client.chat(new LlmChatRequest().addUserMessage("Hello"));
            fail("expected LlmException for a userinfo-bearing api.url");
        } catch (final LlmException e) {
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
    public void test_streamChat_userInfoApiUrlIsRefusedWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestApiKey("sk-test-key");
        client.init();
        final List<Throwable> callbackErrors = new ArrayList<>();
        final ListAppender app = attachLogCapture();
        try {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new LlmStreamCallback() {
                @Override
                public void onChunk(final String chunk, final boolean done) {
                    // not reached
                }

                @Override
                public void onError(final Throwable t) {
                    callbackErrors.add(t);
                }
            });
            fail("expected LlmException for a userinfo-bearing api.url");
        } catch (final LlmException e) {
            final String propagated = renderThrowable(e);
            assertFalse("propagated exception must not carry the credential: " + propagated, propagated.contains(USERINFO_CREDENTIAL));
            assertTrue("the supported alternative must be named: " + e.getMessage(), e.getMessage().contains(PROXY_USERNAME_KEY));
        } finally {
            final String logged = app.rendered();
            detachLogCapture(app);
            assertFalse("log must not carry the credential: " + logged, logged.contains(USERINFO_CREDENTIAL));
        }
        assertEquals(1, callbackErrors.size());
        final String viaCallback = renderThrowable(callbackErrors.get(0));
        assertFalse("callback error must not carry the credential: " + viaCallback, viaCallback.contains(USERINFO_CREDENTIAL));
        assertTrue("callback error must name the supported alternative: " + viaCallback, viaCallback.contains(PROXY_USERNAME_KEY));
    }

    // ========== retry config getter tests ==========

    @Test
    public void test_getRetryMaxAttempts_default() {
        final TestableOpenAiLlmClient bare = new TestableOpenAiLlmClient();
        assertEquals(10, bare.getRetryMaxAttempts());
    }

    @Test
    public void test_getRetryBaseDelayMs_default() {
        final TestableOpenAiLlmClient bare = new TestableOpenAiLlmClient();
        assertEquals(2000L, bare.getRetryBaseDelayMs());
    }

    @Test
    public void test_getRetryMaxAttempts_overridden() {
        client.setTestConfig("retry.max", "5");
        assertEquals(5, client.getRetryMaxAttempts());
    }

    @Test
    public void test_getRetryBaseDelayMs_overridden() {
        client.setTestConfig("retry.base.delay.ms", "500");
        assertEquals(500L, client.getRetryBaseDelayMs());
    }

    // ========== extractErrorDetails tests ==========

    @Test
    public void test_extractErrorDetails_full() {
        final String body = "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"invalid_request_error\","
                + "\"code\":\"invalid_api_key\",\"param\":null}}";
        final String result = client.extractErrorDetails(body);
        assertTrue(result.contains("type=invalid_request_error"));
        assertTrue(result.contains("code=invalid_api_key"));
        assertTrue(result.contains("message=Invalid API key"));
    }

    @Test
    public void test_extractErrorDetails_partial() {
        final String body = "{\"error\":{\"message\":\"oops\"}}";
        final String result = client.extractErrorDetails(body);
        assertTrue(result.contains("message=oops"));
        assertTrue(result.contains("type=null"));
    }

    @Test
    public void test_extractErrorDetails_nonJson_html() {
        final String body = "<html>502 Bad Gateway</html>";
        final String result = client.extractErrorDetails(body);
        // Either the parser throws (we clip and return verbatim) or treats it as a single
        // string token; either way the body content must remain readable in logs.
        assertTrue(result.contains("502 Bad Gateway"));
    }

    @Test
    public void test_extractErrorDetails_emptyJsonObject() {
        // {} parses, but has no "error" key - fall through to clip path.
        assertEquals("{}", client.extractErrorDetails("{}"));
    }

    @Test
    public void test_extractErrorDetails_long_truncated() {
        final String body = "x".repeat(2000);
        final String result = client.extractErrorDetails(body);
        assertTrue(result.endsWith("...(truncated)"));
        assertEquals(1024 + "...(truncated)".length(), result.length());
    }

    @Test
    public void test_extractErrorDetails_null_blank() {
        assertEquals("", client.extractErrorDetails(null));
        assertEquals("", client.extractErrorDetails(""));
    }

    // ========== Helper methods ==========

    private LlmChatRequest buildSimpleRequest() {
        final LlmChatRequest req = new LlmChatRequest();
        final List<LlmMessage> msgs = new ArrayList<>();
        final LlmMessage m = new LlmMessage();
        m.setRole("user");
        m.setContent("hello");
        msgs.add(m);
        req.setMessages(msgs);
        return req;
    }

    private String simpleSuccessBody() {
        return "{\"id\":\"chatcmpl-1\",\"model\":\"gpt-5-mini\","
                + "\"choices\":[{\"message\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
                + "\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}";
    }

    private String simpleStreamSseBody() {
        return "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"
                + "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" + "data: [DONE]\n\n";
    }

    private static final class ListAppender extends AbstractAppender {
        final List<LogEvent> events = new ArrayList<>();

        ListAppender() {
            super("ListAppender", null, null, true, null);
        }

        @Override
        public void append(final LogEvent event) {
            events.add(event.toImmutable());
        }

        List<String> messages() {
            return events.stream().map(e -> e.getMessage().getFormattedMessage()).toList();
        }

        List<String> messagesAt(final org.apache.logging.log4j.Level level) {
            return events.stream().filter(e -> e.getLevel().equals(level)).map(e -> e.getMessage().getFormattedMessage()).toList();
        }

        /**
         * Everything a real appender would write out: the formatted message <em>and</em> the
         * attached throwable. {@link #messages()} alone cannot see a throwable, so an assertion
         * built on it goes green while the rendered log still leaks through the stack trace.
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
        final LoggerConfig cfg = ctx.getConfiguration().getLoggerConfig("org.codelibs.fess.llm.openai.OpenAiLlmClient");
        cfg.addAppender(appender, org.apache.logging.log4j.Level.DEBUG, null);
        cfg.setLevel(org.apache.logging.log4j.Level.DEBUG);
        ctx.updateLoggers();
        return appender;
    }

    private void detachLogCapture(final ListAppender appender) {
        final LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        final LoggerConfig cfg = ctx.getConfiguration().getLoggerConfig("org.codelibs.fess.llm.openai.OpenAiLlmClient");
        cfg.removeAppender(appender.getName());
        ctx.updateLoggers();
        appender.stop();
    }

    private void setupClientForMockServer() {
        final String baseUrl = mockServer.url("").toString();
        // Remove trailing slash
        final String apiUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        client.setTestApiUrl(apiUrl);
        client.setTestApiKey("sk-test-key");
        client.setTestModel("gpt-4");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);
        client.setTestTimeout(30000);
        client.init();
    }

    // ========== Proxy tests ==========

    @Test
    public void test_chat_throughProxy_withoutAuth() throws Exception {
        // With a configured HTTP proxy, HttpClient sends the request to the proxy
        // with an absolute-form request URI. We use a separate MockWebServer as the proxy
        // and target a non-localhost address to verify routing.
        final MockWebServer proxyServer = new MockWebServer();
        try {
            proxyServer.start();
            final String responseJson = """
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "model": "gpt-4",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    }
                    """;
            proxyServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

            client.setTestApiUrl("http://openai.invalid/v1");
            client.setTestApiKey("sk-test-key");
            client.setTestModel("gpt-4");
            client.setTestTimeout(30000);
            client.setTestProxyHost(proxyServer.getHostName());
            client.setTestProxyPort(proxyServer.getPort());
            client.init();

            final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
            final LlmChatResponse response = client.chat(request);
            assertEquals("ok", response.getContent());

            final RecordedRequest recorded = proxyServer.takeRequest();
            assertTrue("Expected absolute-form URI starting with http://openai.invalid/, got: " + recorded.getRequestLine(),
                    recorded.getRequestLine().contains("http://openai.invalid/"));
            assertNull(recorded.getHeader("Proxy-Authorization"), "No proxy auth expected");
        } finally {
            proxyServer.shutdown();
        }
    }

    @Test
    public void test_chat_throughProxy_withBasicAuth() throws Exception {
        final MockWebServer proxyServer = new MockWebServer();
        try {
            proxyServer.start();
            // First response: 407 challenges the client to authenticate.
            proxyServer
                    .enqueue(new MockResponse().setResponseCode(407).addHeader("Proxy-Authenticate", "Basic realm=\"proxy\"").setBody(""));
            // Second response: success after the client retries with credentials.
            final String responseJson = """
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "model": "gpt-4",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    }
                    """;
            proxyServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

            client.setTestApiUrl("http://openai.invalid/v1");
            client.setTestApiKey("sk-test-key");
            client.setTestModel("gpt-4");
            client.setTestTimeout(30000);
            client.setTestProxyHost(proxyServer.getHostName());
            client.setTestProxyPort(proxyServer.getPort());
            client.setTestProxyUsername("proxyuser");
            client.setTestProxyPassword("proxypass");
            client.init();

            final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
            final LlmChatResponse response = client.chat(request);
            assertEquals("ok", response.getContent());

            final RecordedRequest first = proxyServer.takeRequest();
            assertNull(first.getHeader("Proxy-Authorization"));
            final RecordedRequest second = proxyServer.takeRequest();
            final String auth = second.getHeader("Proxy-Authorization");
            assertNotNull(auth, "Proxy-Authorization header expected on retry");
            final String expected = "Basic "
                    + java.util.Base64.getEncoder().encodeToString("proxyuser:proxypass".getBytes(java.nio.charset.StandardCharsets.UTF_8));
            assertEquals(expected, auth);
        } finally {
            proxyServer.shutdown();
        }
    }

    @Test
    public void test_chat_noProxy_directConnection() throws Exception {
        final String responseJson = """
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": "gpt-4",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "direct"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);
        assertEquals("direct", response.getContent());

        final RecordedRequest recorded = mockServer.takeRequest();
        assertTrue("Expected origin-form request line, got: " + recorded.getRequestLine(),
                recorded.getRequestLine().startsWith("POST /chat/completions"));
    }

    /**
     * Testable subclass of OpenAiLlmClient that allows setting configuration values
     * directly without depending on FessConfig.
     */
    private static class TestableOpenAiLlmClient extends OpenAiLlmClient {
        private String testApiKey = "";
        private String testApiUrl = "https://api.openai.com/v1";
        private String testModel = "gpt-5-mini";
        private int testTimeout = 60000;
        private double testTemperature = 0.7;
        private int testMaxTokens = 4096;
        private String testProxyHost = "";
        private Integer testProxyPort = null;
        private String testProxyUsername = "";
        private String testProxyPassword = "";
        private final Map<String, String> testConfigOverrides = new HashMap<>();

        void setTestConfig(final String suffixKey, final String value) {
            testConfigOverrides.put(suffixKey, value);
        }

        @Override
        protected int getRetryMaxAttempts() {
            final String v = testConfigOverrides.get("retry.max");
            return v != null ? Integer.parseInt(v) : super.getRetryMaxAttempts();
        }

        @Override
        protected long getRetryBaseDelayMs() {
            final String v = testConfigOverrides.get("retry.base.delay.ms");
            return v != null ? Long.parseLong(v) : super.getRetryBaseDelayMs();
        }

        @Override
        protected boolean isStreamUsageEnabled() {
            final String v = testConfigOverrides.get("stream.include.usage");
            return v != null ? Boolean.parseBoolean(v) : super.isStreamUsageEnabled();
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

        void setTestTemperature(final double temperature) {
            this.testTemperature = temperature;
        }

        void setTestMaxTokens(final int maxTokens) {
            this.testMaxTokens = maxTokens;
        }

        void setTestProxyHost(final String proxyHost) {
            this.testProxyHost = proxyHost;
        }

        void setTestProxyPort(final Integer proxyPort) {
            this.testProxyPort = proxyPort;
        }

        void setTestProxyUsername(final String proxyUsername) {
            this.testProxyUsername = proxyUsername;
        }

        void setTestProxyPassword(final String proxyPassword) {
            this.testProxyPassword = proxyPassword;
        }

        @Override
        protected String getProxyHost() {
            return testProxyHost;
        }

        @Override
        protected Integer getProxyPort() {
            return testProxyPort;
        }

        @Override
        protected String getProxyUsername() {
            return testProxyUsername;
        }

        @Override
        protected String getProxyPassword() {
            return testProxyPassword;
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

        protected double getTemperature() {
            return testTemperature;
        }

        protected int getMaxTokens() {
            return testMaxTokens;
        }

        @Override
        protected int getReasoningTokenMultiplier() {
            return 4;
        }

        @Override
        protected String getLlmType() {
            return NAME;
        }

        @Override
        protected boolean isRagChatEnabled() {
            return false;
        }

        @Override
        protected int getAvailabilityCheckInterval() {
            return 0;
        }

        @Override
        protected int getHistoryMaxChars() {
            return 8000;
        }

        @Override
        protected int getIntentHistoryMaxMessages() {
            return 8;
        }

        @Override
        protected int getIntentHistoryMaxChars() {
            return 4000;
        }

        @Override
        public int getHistoryAssistantMaxChars() {
            return 800;
        }

        @Override
        public int getHistoryAssistantSummaryMaxChars() {
            return 800;
        }

        /** Exposes the frame the container reaches through {@code init()}, so a test can prove it never throws. */
        void testUpdateAvailability() {
            updateAvailability();
        }

        int testGetHistoryMaxChars() {
            return getHistoryMaxChars();
        }

        int testGetIntentHistoryMaxMessages() {
            return getIntentHistoryMaxMessages();
        }

        int testGetHistoryAssistantMaxChars() {
            return getHistoryAssistantMaxChars();
        }
    }
}
