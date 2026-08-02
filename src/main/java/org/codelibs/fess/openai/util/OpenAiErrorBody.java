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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Renders an OpenAI error response body as a single-line log diagnostic.
 *
 * <p>The {@code {"error":{...}}} envelope is a property of the API, not of any one caller, so the
 * LLM client and the embedding client share this definition rather than each carrying its own copy
 * of the field names and the clipping rule.
 *
 * @author FessProject
 */
public final class OpenAiErrorBody {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    private static final String ERROR_ENVELOPE_FIELD = "error";
    private static final String ERROR_FIELD_TYPE = "type";
    private static final String ERROR_FIELD_CODE = "code";
    private static final String ERROR_FIELD_PARAM = "param";
    private static final String ERROR_FIELD_MESSAGE = "message";

    /** Maximum characters of a non-JSON body kept in the rendered diagnostic. */
    private static final int MAX_RAW_LENGTH = 1024;

    private OpenAiErrorBody() {
        // utility class
    }

    /**
     * Renders {@code errorBody} as a single-line diagnostic. Returns
     * {@code "type=...,code=...,param=...,message=..."} when the body parses as the documented
     * {@code {"error":{...}}} envelope; otherwise returns the body trimmed (clipped at
     * {@value #MAX_RAW_LENGTH} chars + {@code "...(truncated)"} suffix) so non-JSON gateway pages
     * remain readable in logs.
     *
     * @param errorBody the raw HTTP response body from a failed OpenAI API call.
     * @return a single-line diagnostic suitable for logging.
     */
    public static String render(final String errorBody) {
        if (errorBody == null || errorBody.isEmpty()) {
            return "";
        }
        try {
            final JsonNode root = objectMapper.readTree(errorBody);
            if (root.isObject() && root.has(ERROR_ENVELOPE_FIELD) && root.get(ERROR_ENVELOPE_FIELD).isObject()) {
                final JsonNode err = root.get(ERROR_ENVELOPE_FIELD);
                return ERROR_FIELD_TYPE + "=" + err.path(ERROR_FIELD_TYPE).asText("null") //
                        + "," + ERROR_FIELD_CODE + "=" + err.path(ERROR_FIELD_CODE).asText("null") //
                        + "," + ERROR_FIELD_PARAM + "=" + err.path(ERROR_FIELD_PARAM).asText("null") //
                        + "," + ERROR_FIELD_MESSAGE + "=" + err.path(ERROR_FIELD_MESSAGE).asText("null");
            }
        } catch (final JsonProcessingException e) {
            // fall through to raw clip
        }
        final String trimmed = errorBody.trim();
        return trimmed.length() > MAX_RAW_LENGTH ? trimmed.substring(0, MAX_RAW_LENGTH) + "...(truncated)" : trimmed;
    }
}
