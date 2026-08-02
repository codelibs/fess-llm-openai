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

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.codelibs.fess.util.CredentialUrlUtil;

/**
 * Builds HttpClient request objects from a configured URL without letting that URL reach a log.
 *
 * <p>{@code new HttpGet(String)} and {@code new HttpPost(String)} delegate to {@code URI.create},
 * which rejects a syntactically invalid URI with an {@link IllegalArgumentException} whose message
 * - and whose {@link java.net.URISyntaxException} cause - quote the offending URI in full. Both
 * clients in this plugin build their request URI from the configured {@code api.url}, and that URL
 * can legitimately carry a credential as a query parameter, so an {@code api.url} whose credential
 * contains a character that is illegal in a URI (a space, for instance) would write the credential
 * verbatim into the WARN log through the formatted message and through the logged stack trace, and
 * would hand it upstream as the exception cause. Every request built here therefore routes that
 * failure through {@link CredentialUrlUtil#invalidUrlException(String, IllegalArgumentException)}.
 *
 * <p>This class also refuses a URL whose authority carries a userinfo component. RFC 9110 4.2.4
 * forbids a sender from generating userinfo in an {@code http}/{@code https} target URI, and
 * HttpClient enforces that unconditionally - there is no setting that turns it off - so such a URL
 * can never issue a request. Refusing it up front, with a message naming the supported
 * alternative, replaces an opaque runtime protocol failure with an actionable configuration error.
 *
 * <p>What a credential in a URL <em>is</em>, and how one is detected or masked, is provider-agnostic
 * and lives in {@link CredentialUrlUtil}. What stays here is the OpenAI-specific remedy: which
 * configuration key to fix and which authentication mechanism to use instead.
 */
public final class HttpRequestFactory {

    private HttpRequestFactory() {
        // nothing
    }

    /**
     * Builds the message reported when a configured URL carries userinfo. It names the offending
     * configuration key and the supported alternative, and - like every other message this class
     * produces - carries no part of the URL, since the URL is what holds the credential.
     *
     * @param configKey the configuration key the URL was read from.
     * @return the message.
     */
    public static String userInfoRejectedMessage(final String configKey) {
        return "Refusing the URL configured in " + configKey
                + ": its authority carries a userinfo credential, which RFC 9110 forbids in an http/https target URI. "
                + "HttpClient rejects such a request URI unconditionally, so this URL can never issue a request. "
                + "Remove the credential from the URL: OpenAI and OpenAI-compatible gateways authenticate with an "
                + "Authorization header built from the configured api.key. If the endpoint sits behind an "
                + "authenticating proxy, configure http.proxy.host, http.proxy.port, http.proxy.username and "
                + "http.proxy.password instead. The URL itself is omitted here because it holds the credential.";
    }

    /**
     * Throws when the given URL's authority carries userinfo.
     *
     * @param url the URL to check.
     * @param configKey the configuration key the URL was read from.
     * @throws IllegalArgumentException if userinfo is present; the exception carries no part of
     *         the URL and no cause.
     */
    private static void rejectUserInfo(final String url, final String configKey) {
        if (CredentialUrlUtil.hasUserInfo(url)) {
            throw new IllegalArgumentException(userInfoRejectedMessage(configKey));
        }
    }

    /**
     * Creates a {@code GET} request for the given URL.
     *
     * @param url the request URL.
     * @param configKey the configuration key the URL was read from, named in the failure message.
     * @return the request.
     * @throws IllegalArgumentException if the URL carries userinfo or is not a valid URI; the
     *         exception carries no part of the URL.
     */
    public static HttpGet createGet(final String url, final String configKey) {
        rejectUserInfo(url, configKey);
        try {
            return new HttpGet(url);
        } catch (final IllegalArgumentException e) {
            throw CredentialUrlUtil.invalidUrlException(configKey, e);
        }
    }

    /**
     * Creates a {@code POST} request for the given URL.
     *
     * @param url the request URL.
     * @param configKey the configuration key the URL was read from, named in the failure message.
     * @return the request.
     * @throws IllegalArgumentException if the URL carries userinfo or is not a valid URI; the
     *         exception carries no part of the URL.
     */
    public static HttpPost createPost(final String url, final String configKey) {
        rejectUserInfo(url, configKey);
        try {
            return new HttpPost(url);
        } catch (final IllegalArgumentException e) {
            throw CredentialUrlUtil.invalidUrlException(configKey, e);
        }
    }

}
