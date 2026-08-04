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

import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;

/**
 * Pins the configuration channel of the capability overrides: {@code getOrDefault}, i.e.
 * {@code fess_config.properties} plus {@code -Dfess.config.*} - the same channel as
 * {@code rag.llm.openai.model} and {@code api.url}. A refactor to
 * {@code AbstractEmbeddingClient#getConfigString} would move them to
 * {@code conf/system.properties} and the property would appear not to work; nothing in
 * {@code OpenAiLlmClientTest} would notice, because that suite overrides the read.
 *
 * <p>Both halves are pinned, and deliberately so: a rewrite to
 * {@code System.getProperty("fess.config." + key, defaultValue)} - the shape
 * {@code fess-webapp-semantic-search} uses, so a plausible one to copy in - keeps the
 * {@code -Dfess.config.*} half working and silently drops the {@code fess_config.properties} half.
 * {@link #test_getConfigString_alsoResolvesFromFessConfigProperties()} is the assertion that
 * catches that; the {@code -D} assertions alone survive it.
 *
 * <p>{@link #isUseOneTimeContainer()} is true because a value read through the real FessConfig is
 * memoized for the lifetime of the container, and this class is the only one that plants such a
 * value. Without it the planted value would leak into whatever class runs next, order-dependently.
 */
public class OpenAiLlmClientCapabilityConfigTest extends UnitFessTestCase {

    private static final String KEY = "fess.config.rag.llm.openai.reasoning.model.enabled";

    /**
     * Probe suffixes for the {@code getConfigString} key-composition test. Deliberately not
     * capability keys: a value read through the real FessConfig is memoized for the lifetime of
     * the container, so planting one under {@code reasoning.model.enabled} would decide the other
     * tests in this class.
     */
    private static final String ABSENT_SUFFIX = "capability.probe.absent";
    private static final String PLANTED_SUFFIX = "capability.probe.planted";
    private static final String PLANTED_KEY = "fess.config.rag.llm.openai." + PLANTED_SUFFIX;

    @Override
    protected boolean isUseOneTimeContainer() {
        return true;
    }

    @Test
    public void test_getConfigString_composesTheKeyAndReadsTheFessConfigChannel() {
        // The real getConfigString, not the test stub: OpenAiLlmClientTest overrides it against a
        // HashMap, so nothing there can tell whether production composes the key at all.
        final OpenAiLlmClient client = new OpenAiLlmClient();
        // Two-arg form deliberately: a three-argument all-String assertEquals binds to
        // (message, expected, actual) and would silently compare the wrong pair.
        // An absent key must yield the supplied default.
        assertEquals("fallback", client.getConfigString(ABSENT_SUFFIX, "fallback"));
        System.setProperty(PLANTED_KEY, "planted");
        try {
            // Planted under getConfigPrefix() + "." + keySuffix; reading it back proves the
            // composition, not just that some lookup happened.
            assertEquals("planted", client.getConfigString(PLANTED_SUFFIX, "fallback"));
        } finally {
            System.clearProperty(PLANTED_KEY);
        }
    }

    /**
     * A sentinel that no properties file defines, so seeing it back proves the lookup missed.
     */
    private static final String SENTINEL = "__supplied_default_was_used__";

    @Test
    public void test_getConfigString_alsoResolvesFromFessConfigProperties() {
        // The sibling test plants its probe as a -Dfess.config.* system property, which is only
        // one of the two sources getOrDefault reads. Nothing under rag.llm.openai.* is declared in
        // the fess_config.properties that Fess core ships, and adding a fess_config.properties to
        // src/test/resources would shadow that whole file rather than extend it - so the file half
        // is reached by pointing the very same production getConfigString body
        // (getOrDefault(getConfigPrefix() + "." + keySuffix, defaultValue)) at a prefix the shipped
        // file does declare. Only the prefix differs from a real capability read.
        final OpenAiLlmClient client = new OpenAiLlmClient() {
            @Override
            protected String getConfigPrefix() {
                return "rag.chat";
            }
        };
        final String value = client.getConfigString("enabled", SENTINEL);
        // Asserting "not the default" rather than the shipped value itself: what has to hold is
        // that fess_config.properties was consulted at all, and Fess core owns whether rag.chat is
        // on by default.
        assertFalse(
                "rag.chat.enabled must resolve out of fess_config.properties rather than falling back to the "
                        + "supplied default - a lookup that reads only System.getProperty(\"fess.config.\" + key) would. value=" + value,
                SENTINEL.equals(value));
    }

    @Test
    public void test_reasoningEnabled_readFromTheFessConfigChannel() {
        System.setProperty(KEY, "true");
        try {
            final OpenAiLlmClient client = new OpenAiLlmClient();
            assertTrue("a non-OpenAI model name must be classified as reasoning when forced through fess_config",
                    client.isReasoningModel("qwen3-32b"));
            assertEquals(Boolean.TRUE, client.getCapabilityOverride("reasoning.model.enabled"));
        } finally {
            System.clearProperty(KEY);
        }
    }

    @Test
    public void test_reasoningEnabled_unsetLeavesNameInference() {
        final OpenAiLlmClient client = new OpenAiLlmClient();
        assertNull(client.getCapabilityOverride("reasoning.model.enabled"));
        assertFalse(client.isReasoningModel("qwen3-32b"));
        assertTrue(client.isReasoningModel("gpt-5-nano"));
    }
}
