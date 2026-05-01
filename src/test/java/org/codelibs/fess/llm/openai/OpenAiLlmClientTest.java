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

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("429"));
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

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("500"));
        }
    }

    @Test
    public void test_chat_errorResponse_emptyBody() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("").addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("503"));
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

    // ========== streamChat() method tests ==========

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
            assertTrue(error.getMessage().contains("429"));
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
            assertTrue(error.getMessage().contains("500"));
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
            assertTrue(error.getMessage().contains("503"));
            assertTrue(errorReceived.get());
        }
    }

    @Test
    public void test_chat_serviceUnavailable() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("Service Unavailable"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("503"));
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
        assertTrue(OpenAiLlmClient.isRetryableStatus(429));
    }

    @Test
    public void test_isRetryableStatus_500() {
        assertTrue(OpenAiLlmClient.isRetryableStatus(500));
    }

    @Test
    public void test_isRetryableStatus_502() {
        assertTrue(OpenAiLlmClient.isRetryableStatus(502));
    }

    @Test
    public void test_isRetryableStatus_503() {
        assertTrue(OpenAiLlmClient.isRetryableStatus(503));
    }

    @Test
    public void test_isRetryableStatus_504() {
        assertTrue(OpenAiLlmClient.isRetryableStatus(504));
    }

    @Test
    public void test_isRetryableStatus_400() {
        assertFalse(OpenAiLlmClient.isRetryableStatus(400));
    }

    @Test
    public void test_isRetryableStatus_401() {
        assertFalse(OpenAiLlmClient.isRetryableStatus(401));
    }

    @Test
    public void test_isRetryableStatus_404() {
        assertFalse(OpenAiLlmClient.isRetryableStatus(404));
    }

    @Test
    public void test_isRetryableStatus_408_notRetried() {
        // OpenAI rarely returns 408; if it ever appears in production logs we'll add it.
        assertFalse(OpenAiLlmClient.isRetryableStatus(408));
    }

    @Test
    public void test_isRetryableStatus_200() {
        assertFalse(OpenAiLlmClient.isRetryableStatus(200));
    }

    // ========== parseRetryAfterSeconds tests ==========

    @Test
    public void test_parseRetryAfterSeconds_integer() {
        assertEquals(5L, OpenAiLlmClient.parseRetryAfterSeconds("5"));
    }

    @Test
    public void test_parseRetryAfterSeconds_zero() {
        assertEquals(0L, OpenAiLlmClient.parseRetryAfterSeconds("0"));
    }

    @Test
    public void test_parseRetryAfterSeconds_largeClamped() {
        assertEquals(600L, OpenAiLlmClient.parseRetryAfterSeconds("3600"));
    }

    @Test
    public void test_parseRetryAfterSeconds_negative() {
        assertEquals(-1L, OpenAiLlmClient.parseRetryAfterSeconds("-1"));
    }

    @Test
    public void test_parseRetryAfterSeconds_null() {
        assertEquals(-1L, OpenAiLlmClient.parseRetryAfterSeconds(null));
    }

    @Test
    public void test_parseRetryAfterSeconds_blank() {
        assertEquals(-1L, OpenAiLlmClient.parseRetryAfterSeconds(""));
        assertEquals(-1L, OpenAiLlmClient.parseRetryAfterSeconds("  "));
    }

    @Test
    public void test_parseRetryAfterSeconds_httpDate() {
        // HTTP-date format intentionally unsupported; caller falls back to backoff.
        assertEquals(-1L, OpenAiLlmClient.parseRetryAfterSeconds("Wed, 21 Oct 2026 07:28:00 GMT"));
    }

    @Test
    public void test_parseRetryAfterSeconds_decimal() {
        assertEquals(-1L, OpenAiLlmClient.parseRetryAfterSeconds("1.5"));
    }

    @Test
    public void test_parseRetryAfterSeconds_whitespacePadded() {
        assertEquals(7L, OpenAiLlmClient.parseRetryAfterSeconds("  7  "));
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
