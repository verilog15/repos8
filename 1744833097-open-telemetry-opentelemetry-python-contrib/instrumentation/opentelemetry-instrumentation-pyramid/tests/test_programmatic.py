# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock, patch

from pyramid.config import Configurator

from opentelemetry import trace
from opentelemetry.instrumentation.propagators import (
    TraceResponsePropagator,
    get_global_response_propagator,
    set_global_response_propagator,
)
from opentelemetry.instrumentation.pyramid import PyramidInstrumentor
from opentelemetry.semconv.attributes import exception_attributes
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.test.wsgitestutil import WsgiTestBase
from opentelemetry.util.http import get_excluded_urls

# pylint: disable=import-error
from .pyramid_base_test import InstrumentationTest


def expected_attributes(override_attributes):
    default_attributes = {
        SpanAttributes.HTTP_METHOD: "GET",
        SpanAttributes.HTTP_SERVER_NAME: "localhost",
        SpanAttributes.HTTP_SCHEME: "http",
        SpanAttributes.NET_HOST_PORT: 80,
        SpanAttributes.NET_HOST_NAME: "localhost",
        SpanAttributes.HTTP_HOST: "localhost",
        SpanAttributes.HTTP_TARGET: "/",
        SpanAttributes.HTTP_FLAVOR: "1.1",
        SpanAttributes.HTTP_STATUS_CODE: 200,
    }
    for key, val in override_attributes.items():
        default_attributes[key] = val
    return default_attributes


class TestProgrammatic(InstrumentationTest, WsgiTestBase):
    def setUp(self):
        super().setUp()
        config = Configurator()
        PyramidInstrumentor().instrument_config(config)

        self.config = config

        self._common_initialization(self.config)

        self.env_patch = patch.dict(
            "os.environ",
            {
                "OTEL_PYTHON_PYRAMID_EXCLUDED_URLS": "http://localhost/excluded_arg/123,excluded_noarg"
            },
        )
        self.env_patch.start()
        self.exclude_patch = patch(
            "opentelemetry.instrumentation.pyramid.callbacks._excluded_urls",
            get_excluded_urls("PYRAMID"),
        )
        self.exclude_patch.start()

    def tearDown(self):
        super().tearDown()
        with self.disable_logging():
            PyramidInstrumentor().uninstrument_config(self.config)

    def test_uninstrument(self):
        resp = self.client.get("/hello/123")
        self.assertEqual(200, resp.status_code)
        self.assertEqual([b"Hello: 123"], list(resp.response))
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)

        PyramidInstrumentor().uninstrument_config(self.config)
        # Need to remake the WSGI app export
        self._common_initialization(self.config)

        resp = self.client.get("/hello/123")
        self.assertEqual(200, resp.status_code)
        self.assertEqual([b"Hello: 123"], list(resp.response))
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)

    def test_simple(self):
        expected_attrs = expected_attributes(
            {
                SpanAttributes.HTTP_TARGET: "/hello/123",
                SpanAttributes.HTTP_ROUTE: "/hello/{helloid}",
            }
        )
        self.client.get("/hello/123")
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)
        self.assertEqual(span_list[0].name, "/hello/{helloid}")
        self.assertEqual(span_list[0].kind, trace.SpanKind.SERVER)
        self.assertEqual(span_list[0].attributes, expected_attrs)

    def test_response_headers(self):
        orig = get_global_response_propagator()
        set_global_response_propagator(TraceResponsePropagator())

        response = self.client.get("/hello/500")
        self.assertTraceResponseHeaderMatchesSpan(
            response.headers, self.memory_exporter.get_finished_spans()[0]
        )

        set_global_response_propagator(orig)

    def test_not_recording(self):
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = False
        mock_tracer.start_span.return_value = mock_span
        with patch("opentelemetry.trace.get_tracer"):
            self.client.get("/hello/123")
            span_list = self.memory_exporter.get_finished_spans()
            self.assertEqual(len(span_list), 0)
            self.assertFalse(mock_span.is_recording())
            self.assertTrue(mock_span.is_recording.called)
            self.assertFalse(mock_span.set_attribute.called)
            self.assertFalse(mock_span.set_status.called)

    def test_404(self):
        expected_attrs = expected_attributes(
            {
                SpanAttributes.HTTP_METHOD: "POST",
                SpanAttributes.HTTP_TARGET: "/bye",
                SpanAttributes.HTTP_STATUS_CODE: 404,
            }
        )

        resp = self.client.post("/bye")
        self.assertEqual(404, resp.status_code)
        resp.close()
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)
        self.assertEqual(span_list[0].name, "POST /bye")
        self.assertEqual(span_list[0].kind, trace.SpanKind.SERVER)
        self.assertEqual(span_list[0].attributes, expected_attrs)
        self.assertEqual(len(span_list[0].events), 0)

    def test_internal_error(self):
        expected_attrs = expected_attributes(
            {
                SpanAttributes.HTTP_TARGET: "/hello/500",
                SpanAttributes.HTTP_ROUTE: "/hello/{helloid}",
                SpanAttributes.HTTP_STATUS_CODE: 500,
            }
        )
        resp = self.client.get("/hello/500")
        self.assertEqual(500, resp.status_code)
        resp.close()
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)
        self.assertEqual(span_list[0].name, "/hello/{helloid}")
        self.assertEqual(span_list[0].kind, trace.SpanKind.SERVER)
        self.assertEqual(span_list[0].attributes, expected_attrs)
        self.assertEqual(
            span_list[0].status.status_code, trace.StatusCode.ERROR
        )
        self.assertIn(
            "HTTPInternalServerError", span_list[0].status.description
        )
        self.assertEqual(
            span_list[0]
            .events[0]
            .attributes[exception_attributes.EXCEPTION_TYPE],
            "pyramid.httpexceptions.HTTPInternalServerError",
        )

    def test_internal_exception(self):
        expected_attrs = expected_attributes(
            {
                SpanAttributes.HTTP_TARGET: "/hello/900",
                SpanAttributes.HTTP_ROUTE: "/hello/{helloid}",
                SpanAttributes.HTTP_STATUS_CODE: 500,
            }
        )
        with self.assertRaises(NotImplementedError):
            resp = self.client.get("/hello/900")
            resp.close()

        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)
        self.assertEqual(span_list[0].name, "/hello/{helloid}")
        self.assertEqual(span_list[0].kind, trace.SpanKind.SERVER)
        self.assertEqual(span_list[0].attributes, expected_attrs)
        self.assertEqual(
            span_list[0].status.status_code, trace.StatusCode.ERROR
        )
        self.assertEqual(span_list[0].status.description, "error message")

        expected_error_event_attrs = {
            exception_attributes.EXCEPTION_TYPE: "NotImplementedError",
            exception_attributes.EXCEPTION_MESSAGE: "error message",
        }
        self.assertEqual(span_list[0].events[0].name, "exception")
        # Ensure exception event has specific attributes, but allow additional ones
        self.assertLess(
            expected_error_event_attrs.items(),
            dict(span_list[0].events[0].attributes).items(),
        )

    def test_tween_list(self):
        tween_list = "opentelemetry.instrumentation.pyramid.trace_tween_factory\npyramid.tweens.excview_tween_factory"
        config = Configurator(settings={"pyramid.tweens": tween_list})
        PyramidInstrumentor().instrument_config(config)
        self._common_initialization(config)

        resp = self.client.get("/hello/123")
        self.assertEqual(200, resp.status_code)
        self.assertEqual([b"Hello: 123"], list(resp.response))
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)

        PyramidInstrumentor().uninstrument_config(config)
        # Need to remake the WSGI app export
        self._common_initialization(config)

        resp = self.client.get("/hello/123")
        self.assertEqual(200, resp.status_code)
        self.assertEqual([b"Hello: 123"], list(resp.response))
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)

    @patch("opentelemetry.instrumentation.pyramid.callbacks._logger")
    def test_warnings(self, mock_logger):
        tween_list = "pyramid.tweens.excview_tween_factory"
        config = Configurator(settings={"pyramid.tweens": tween_list})
        PyramidInstrumentor().instrument_config(config)
        self._common_initialization(config)

        self.client.get("/hello/123")
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 0)
        self.assertEqual(mock_logger.warning.called, True)

        mock_logger.warning.called = False

        tween_list = (
            "opentelemetry.instrumentation.pyramid.trace_tween_factory"
        )
        config = Configurator(settings={"pyramid.tweens": tween_list})
        self._common_initialization(config)

        self.client.get("/hello/123")
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 0)
        self.assertEqual(mock_logger.warning.called, True)

    def test_exclude_lists(self):
        self.client.get("/excluded_arg/123")
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 0)

        self.client.get("/excluded_arg/125")
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)

        self.client.get("/excluded_noarg")
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)

        self.client.get("/excluded_noarg2")
        span_list = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(span_list), 1)
