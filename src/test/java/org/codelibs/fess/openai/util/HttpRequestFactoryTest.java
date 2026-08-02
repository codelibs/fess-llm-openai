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

import org.codelibs.fess.unit.UnitFessTestCase;
import org.codelibs.fess.util.CredentialUrlUtil;
import org.junit.jupiter.api.Test;

public class HttpRequestFactoryTest extends UnitFessTestCase {

    /** Credential value that must never survive into the exception raised for a malformed URL. */
    private static final String RAW_CREDENTIAL = "sk secret";

    private static final String CONFIG_KEY = "content_chunker.embedding.openai.api.url";

    @Test
    public void test_createGet_validUrl() throws Exception {
        assertEquals("https://api.openai.com/v1/models",
                HttpRequestFactory.createGet("https://api.openai.com/v1/models", CONFIG_KEY).getUri().toString());
    }

    @Test
    public void test_createPost_validUrl() throws Exception {
        assertEquals("https://gw.example.com/v1/embeddings?api-key=k",
                HttpRequestFactory.createPost("https://gw.example.com/v1/embeddings?api-key=k", CONFIG_KEY).getUri().toString());
    }

    @Test
    public void test_createPost_malformedUrlIsNotEchoed() {
        // A space is illegal in a URI, so URI.create rejects the URL and quotes it in full.
        try {
            HttpRequestFactory.createPost("https://gw.example.com/v1/embeddings?api_key=" + RAW_CREDENTIAL, CONFIG_KEY);
            fail("expected IllegalArgumentException for a malformed URL");
        } catch (final IllegalArgumentException e) {
            final String message = e.getMessage();
            assertFalse("credential must not survive: " + message, message.contains(RAW_CREDENTIAL));
            assertFalse("no part of the URL may be echoed: " + message, message.contains("gw.example.com"));
            assertTrue("the configuration key to inspect must be named: " + message, message.contains(CONFIG_KEY));
            assertTrue("syntax reason must be retained: " + message, message.contains("Illegal character in query"));
            assertTrue("failure index must be retained: " + message, message.contains("at index "));
            assertNull(e.getCause(), "cause must be dropped: it quotes the raw URI");
        }
    }

    @Test
    public void test_createGet_malformedUrlIsNotEchoed() {
        try {
            HttpRequestFactory.createGet("https://gw.example.com/v1/models?token=" + RAW_CREDENTIAL, CONFIG_KEY);
            fail("expected IllegalArgumentException for a malformed URL");
        } catch (final IllegalArgumentException e) {
            final String message = e.getMessage();
            assertFalse("credential must not survive: " + message, message.contains(RAW_CREDENTIAL));
            assertFalse("no part of the URL may be echoed: " + message, message.contains("gw.example.com"));
            assertNull(e.getCause(), "cause must be dropped: it quotes the raw URI");
        }
    }

    @Test
    public void test_createPost_malformedUserInfoIsNotEchoed() {
        // The reason the message carries no URL at all. A URL is unparseable precisely because it
        // holds a character the masking patterns exclude: the userinfo pattern excludes
        // whitespace, so this URL masks to itself and echoing the "masked" form would hand the
        // credential straight back.
        // Three all-String arguments would bind ambiguously across the JUnit 4/5 assertEquals
        // overloads, so this states the premise with the unambiguous two-argument form.
        assertEquals("https://user:pw spaced@gw.example.com/v1",
                CredentialUrlUtil.maskCredentialInUrl("https://user:pw spaced@gw.example.com/v1"));
        try {
            HttpRequestFactory.createPost("https://user:pw spaced@gw.example.com/v1/chat/completions", CONFIG_KEY);
            fail("expected IllegalArgumentException for a malformed URL");
        } catch (final IllegalArgumentException e) {
            final String message = e.getMessage();
            assertFalse("userinfo credential must not survive: " + message, message.contains("pw spaced"));
            assertFalse("no part of the URL may be echoed: " + message, message.contains("gw.example.com"));
            assertTrue("the configuration key to inspect must be named: " + message, message.contains(CONFIG_KEY));
            assertNull(e.getCause(), "cause must be dropped: it quotes the raw URI");
        }
    }

    // ========== userinfo-bearing api.url is refused up front ==========

    /**
     * A userinfo credential that is a perfectly legal URI character sequence, so nothing in the
     * URI parser objects to it. Only a deliberate userinfo check can catch this one.
     */
    private static final String USERINFO_URL = "https://user:s3cr3tUserinfo@gw.example.com/v1/chat/completions";

    /** The userinfo credential that must appear in no message this class produces. */
    private static final String USERINFO_CREDENTIAL = "s3cr3tUserinfo";

    /** The supported alternative the refusal must name. */
    private static final String PROXY_USERNAME_KEY = "http.proxy.username";

    /** The supported alternative the refusal must name. */
    private static final String PROXY_PASSWORD_KEY = "http.proxy.password";

    @Test
    public void test_createPost_userInfoIsRefusedWithRemedy() {
        // RFC 9110 forbids userinfo in an http/https target URI and HttpClient enforces that
        // unconditionally, so this URL can never issue a request. Refusing it here turns an
        // opaque runtime ProtocolException into an actionable configuration error.
        try {
            HttpRequestFactory.createPost(USERINFO_URL, CONFIG_KEY);
            fail("expected IllegalArgumentException for a userinfo-bearing URL");
        } catch (final IllegalArgumentException e) {
            final String message = e.getMessage();
            assertFalse("credential must not survive: " + message, message.contains(USERINFO_CREDENTIAL));
            assertFalse("no part of the URL may be echoed: " + message, message.contains("gw.example.com"));
            assertTrue("the configuration key to inspect must be named: " + message, message.contains(CONFIG_KEY));
            assertTrue("the supported alternative must be named: " + message, message.contains(PROXY_USERNAME_KEY));
            assertTrue("the supported alternative must be named: " + message, message.contains(PROXY_PASSWORD_KEY));
            assertNull(e.getCause(), "cause must be absent: nothing may carry the URL");
        }
    }

    @Test
    public void test_createGet_userInfoIsRefusedWithRemedy() {
        try {
            HttpRequestFactory.createGet("https://user:s3cr3tUserinfo@gw.example.com/v1/models", CONFIG_KEY);
            fail("expected IllegalArgumentException for a userinfo-bearing URL");
        } catch (final IllegalArgumentException e) {
            final String message = e.getMessage();
            assertFalse("credential must not survive: " + message, message.contains(USERINFO_CREDENTIAL));
            assertTrue("the supported alternative must be named: " + message, message.contains(PROXY_USERNAME_KEY));
            assertNull(e.getCause(), "cause must be absent: nothing may carry the URL");
        }
    }

    @Test
    public void test_createPost_userInfoWithWhitespaceIsRefusedAsUserInfo() {
        // The masking regex excludes whitespace, so a detector reusing it would miss exactly this
        // input. Structural authority parsing must catch it, and must catch it *as userinfo* -
        // reporting a generic URI-syntax error here would name no remedy.
        try {
            HttpRequestFactory.createPost("https://user:pw spaced@gw.example.com/v1/chat/completions", CONFIG_KEY);
            fail("expected IllegalArgumentException for a userinfo-bearing URL");
        } catch (final IllegalArgumentException e) {
            final String message = e.getMessage();
            assertFalse("credential must not survive: " + message, message.contains("pw spaced"));
            assertTrue("must be refused as userinfo, naming the remedy: " + message, message.contains(PROXY_USERNAME_KEY));
            assertNull(e.getCause(), "cause must be absent: nothing may carry the URL");
        }
    }

    @Test
    public void test_createPost_userInfoWithoutPasswordIsRefused() {
        // RFC 3986 userinfo needs no colon; a bare "user@" is still userinfo.
        try {
            HttpRequestFactory.createPost("https://someuser@gw.example.com/v1/chat/completions", CONFIG_KEY);
            fail("expected IllegalArgumentException for a userinfo-bearing URL");
        } catch (final IllegalArgumentException e) {
            assertTrue("the supported alternative must be named: " + e.getMessage(), e.getMessage().contains(PROXY_USERNAME_KEY));
        }
    }

    @Test
    public void test_createGet_portBearingAuthorityIsNotRefused() throws Exception {
        // "host:8443" is a port, not userinfo. The refusal must not fire on it.
        assertEquals("https://gw.example.com:8443/v1/models",
                HttpRequestFactory.createGet("https://gw.example.com:8443/v1/models", CONFIG_KEY).getUri().toString());
    }

    @Test
    public void test_createPost_portBearingAuthorityIsNotRefused() throws Exception {
        assertEquals("https://gw.example.com:8443/v1/embeddings",
                HttpRequestFactory.createPost("https://gw.example.com:8443/v1/embeddings", CONFIG_KEY).getUri().toString());
    }

    @Test
    public void test_createPost_atSignOutsideTheAuthorityIsNotRefused() throws Exception {
        // An '@' in the path or in the query is not userinfo: the authority ends at the first
        // '/', '?' or '#'. A pattern scanning the whole URL would false-positive on both.
        assertEquals("https://gw.example.com/v1/user@example.com/embeddings",
                HttpRequestFactory.createPost("https://gw.example.com/v1/user@example.com/embeddings", CONFIG_KEY).getUri().toString());
        assertEquals("https://gw.example.com/v1/embeddings?owner=a@b.example",
                HttpRequestFactory.createPost("https://gw.example.com/v1/embeddings?owner=a@b.example", CONFIG_KEY).getUri().toString());
    }

}
