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

import asyncio
from time import time_ns

import redis
import redis.asyncio
from redis.commands.search.field import (
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from opentelemetry import trace
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.test.test_base import TestBase


class TestRedisInstrument(TestBase):
    def setUp(self):
        super().setUp()
        self.redis_client = redis.Redis(port=6379)
        self.redis_client.flushall()
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

    def tearDown(self):
        RedisInstrumentor().uninstrument()
        super().tearDown()

    def _check_span(self, span, name):
        self.assertEqual(span.name, name)
        self.assertIs(span.status.status_code, trace.StatusCode.UNSET)
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_REDIS_DATABASE_INDEX), 0
        )
        self.assertEqual(
            span.attributes[SpanAttributes.NET_PEER_NAME], "localhost"
        )
        self.assertEqual(span.attributes[SpanAttributes.NET_PEER_PORT], 6379)

    def test_long_command_sanitized(self):
        RedisInstrumentor().uninstrument()
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

        self.redis_client.mget(*range(2000))

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "MGET")
        self.assertTrue(
            span.attributes.get(SpanAttributes.DB_STATEMENT).startswith(
                "MGET ? ? ? ?"
            )
        )
        self.assertTrue(
            span.attributes.get(SpanAttributes.DB_STATEMENT).endswith("...")
        )

    def test_long_command(self):
        self.redis_client.mget(*range(1000))

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "MGET")
        self.assertTrue(
            span.attributes.get(SpanAttributes.DB_STATEMENT).startswith(
                "MGET ? ? ? ?"
            )
        )
        self.assertTrue(
            span.attributes.get(SpanAttributes.DB_STATEMENT).endswith("...")
        )

    def test_basics_sanitized(self):
        RedisInstrumentor().uninstrument()
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

        self.assertIsNone(self.redis_client.get("cheese"))
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "GET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "GET ?"
        )
        self.assertEqual(span.attributes.get("db.redis.args_length"), 2)

    def test_basics(self):
        self.assertIsNone(self.redis_client.get("cheese"))
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "GET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "GET ?"
        )
        self.assertEqual(span.attributes.get("db.redis.args_length"), 2)

    def test_pipeline_traced_sanitized(self):
        RedisInstrumentor().uninstrument()
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

        with self.redis_client.pipeline(transaction=False) as pipeline:
            pipeline.set("blah", 32)
            pipeline.rpush("foo", "éé")
            pipeline.hgetall("xxx")
            pipeline.execute()

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "SET RPUSH HGETALL")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT),
            "SET ? ?\nRPUSH ? ?\nHGETALL ?",
        )
        self.assertEqual(span.attributes.get("db.redis.pipeline_length"), 3)

    def test_pipeline_traced(self):
        with self.redis_client.pipeline(transaction=False) as pipeline:
            pipeline.set("blah", 32)
            pipeline.rpush("foo", "éé")
            pipeline.hgetall("xxx")
            pipeline.execute()

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "SET RPUSH HGETALL")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT),
            "SET ? ?\nRPUSH ? ?\nHGETALL ?",
        )
        self.assertEqual(span.attributes.get("db.redis.pipeline_length"), 3)

    def test_pipeline_immediate_sanitized(self):
        RedisInstrumentor().uninstrument()
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

        with self.redis_client.pipeline() as pipeline:
            pipeline.set("a", 1)
            pipeline.immediate_execute_command("SET", "b", 2)
            pipeline.execute()

        spans = self.memory_exporter.get_finished_spans()
        # expecting two separate spans here, rather than a
        # single span for the whole pipeline
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self._check_span(span, "SET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "SET ? ?"
        )

    def test_pipeline_immediate(self):
        with self.redis_client.pipeline() as pipeline:
            pipeline.set("a", 1)
            pipeline.immediate_execute_command("SET", "b", 2)
            pipeline.execute()

        spans = self.memory_exporter.get_finished_spans()
        # expecting two separate spans here, rather than a
        # single span for the whole pipeline
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self._check_span(span, "SET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "SET ? ?"
        )

    def test_parent(self):
        """Ensure OpenTelemetry works with redis."""
        ot_tracer = trace.get_tracer("redis_svc")

        with ot_tracer.start_as_current_span("redis_get"):
            self.assertIsNone(self.redis_client.get("cheese"))

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        child_span, parent_span = spans[0], spans[1]

        # confirm the parenting
        self.assertIsNone(parent_span.parent)
        self.assertIs(child_span.parent, parent_span.get_span_context())

        self.assertEqual(parent_span.name, "redis_get")
        self.assertEqual(parent_span.instrumentation_info.name, "redis_svc")

        self.assertEqual(child_span.name, "GET")


class TestRedisClusterInstrument(TestBase):
    def setUp(self):
        super().setUp()
        self.redis_client = redis.cluster.RedisCluster(
            host="localhost", port=7000
        )
        self.redis_client.flushall()
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

    def tearDown(self):
        super().tearDown()
        RedisInstrumentor().uninstrument()

    def _check_span(self, span, name):
        self.assertEqual(span.name, name)
        self.assertIs(span.status.status_code, trace.StatusCode.UNSET)

    def test_basics(self):
        self.assertIsNone(self.redis_client.get("cheese"))
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "GET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "GET ?"
        )
        self.assertEqual(span.attributes.get("db.redis.args_length"), 2)

    def test_pipeline_traced(self):
        with self.redis_client.pipeline(transaction=False) as pipeline:
            pipeline.set("blah", 32)
            pipeline.rpush("foo", "éé")
            pipeline.hgetall("xxx")
            pipeline.execute()

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "SET RPUSH HGETALL")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT),
            "SET ? ?\nRPUSH ? ?\nHGETALL ?",
        )
        self.assertEqual(span.attributes.get("db.redis.pipeline_length"), 3)

    def test_parent(self):
        """Ensure OpenTelemetry works with redis."""
        ot_tracer = trace.get_tracer("redis_svc")

        with ot_tracer.start_as_current_span("redis_get"):
            self.assertIsNone(self.redis_client.get("cheese"))

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        child_span, parent_span = spans[0], spans[1]

        # confirm the parenting
        self.assertIsNone(parent_span.parent)
        self.assertIs(child_span.parent, parent_span.get_span_context())

        self.assertEqual(parent_span.name, "redis_get")
        self.assertEqual(parent_span.instrumentation_info.name, "redis_svc")

        self.assertEqual(child_span.name, "GET")


def async_call(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class TestAsyncRedisInstrument(TestBase):
    def setUp(self):
        super().setUp()
        self.redis_client = redis.asyncio.Redis(port=6379)
        async_call(self.redis_client.flushall())
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

    def tearDown(self):
        RedisInstrumentor().uninstrument()
        super().tearDown()

    def _check_span(self, span, name):
        self.assertEqual(span.name, name)
        self.assertIs(span.status.status_code, trace.StatusCode.UNSET)
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_REDIS_DATABASE_INDEX), 0
        )
        self.assertEqual(
            span.attributes[SpanAttributes.NET_PEER_NAME], "localhost"
        )
        self.assertEqual(span.attributes[SpanAttributes.NET_PEER_PORT], 6379)

    def test_long_command(self):
        async_call(self.redis_client.mget(*range(1000)))

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "MGET")
        self.assertTrue(
            span.attributes.get(SpanAttributes.DB_STATEMENT).startswith(
                "MGET ? ? ? ?"
            )
        )
        self.assertTrue(
            span.attributes.get(SpanAttributes.DB_STATEMENT).endswith("...")
        )

    def test_basics(self):
        self.assertIsNone(async_call(self.redis_client.get("cheese")))
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "GET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "GET ?"
        )
        self.assertEqual(span.attributes.get("db.redis.args_length"), 2)

    def test_execute_command_traced_full_time(self):
        """Command should be traced for coroutine execution time, not creation time."""
        coro_created_time = None
        finish_time = None

        async def pipeline_simple():
            nonlocal coro_created_time
            nonlocal finish_time

            # delay coroutine creation from coroutine execution
            coro = self.redis_client.get("foo")
            coro_created_time = time_ns()
            await coro
            finish_time = time_ns()

        async_call(pipeline_simple())

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertTrue(span.start_time > coro_created_time)
        self.assertTrue(span.end_time < finish_time)

    def test_pipeline_traced(self):
        async def pipeline_simple():
            async with self.redis_client.pipeline(
                transaction=False
            ) as pipeline:
                pipeline.set("blah", 32)
                pipeline.rpush("foo", "éé")
                pipeline.hgetall("xxx")
                await pipeline.execute()

        async_call(pipeline_simple())

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "SET RPUSH HGETALL")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT),
            "SET ? ?\nRPUSH ? ?\nHGETALL ?",
        )
        self.assertEqual(span.attributes.get("db.redis.pipeline_length"), 3)

    def test_pipeline_traced_full_time(self):
        """Command should be traced for coroutine execution time, not creation time."""
        coro_created_time = None
        finish_time = None

        async def pipeline_simple():
            async with self.redis_client.pipeline(
                transaction=False
            ) as pipeline:
                nonlocal coro_created_time
                nonlocal finish_time
                pipeline.set("blah", 32)
                pipeline.rpush("foo", "éé")
                pipeline.hgetall("xxx")

                # delay coroutine creation from coroutine execution
                coro = pipeline.execute()
                coro_created_time = time_ns()
                await coro
                finish_time = time_ns()

        async_call(pipeline_simple())

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertTrue(span.start_time > coro_created_time)
        self.assertTrue(span.end_time < finish_time)

    def test_pipeline_immediate(self):
        async def pipeline_immediate():
            async with self.redis_client.pipeline() as pipeline:
                pipeline.set("a", 1)
                await pipeline.immediate_execute_command("SET", "b", 2)
                await pipeline.execute()

        async_call(pipeline_immediate())

        spans = self.memory_exporter.get_finished_spans()
        # expecting two separate spans here, rather than a
        # single span for the whole pipeline
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self._check_span(span, "SET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "SET ? ?"
        )

    def test_pipeline_immediate_traced_full_time(self):
        """Command should be traced for coroutine execution time, not creation time."""
        coro_created_time = None
        finish_time = None

        async def pipeline_simple():
            async with self.redis_client.pipeline(
                transaction=False
            ) as pipeline:
                nonlocal coro_created_time
                nonlocal finish_time
                pipeline.set("a", 1)

                # delay coroutine creation from coroutine execution
                coro = pipeline.immediate_execute_command("SET", "b", 2)
                coro_created_time = time_ns()
                await coro
                finish_time = time_ns()

        async_call(pipeline_simple())

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertTrue(span.start_time > coro_created_time)
        self.assertTrue(span.end_time < finish_time)

    def test_parent(self):
        """Ensure OpenTelemetry works with redis."""
        ot_tracer = trace.get_tracer("redis_svc")

        with ot_tracer.start_as_current_span("redis_get"):
            self.assertIsNone(async_call(self.redis_client.get("cheese")))

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        child_span, parent_span = spans[0], spans[1]

        # confirm the parenting
        self.assertIsNone(parent_span.parent)
        self.assertIs(child_span.parent, parent_span.get_span_context())

        self.assertEqual(parent_span.name, "redis_get")
        self.assertEqual(parent_span.instrumentation_info.name, "redis_svc")

        self.assertEqual(child_span.name, "GET")


class TestAsyncRedisClusterInstrument(TestBase):
    def setUp(self):
        super().setUp()
        self.redis_client = redis.asyncio.cluster.RedisCluster(
            host="localhost", port=7000
        )
        async_call(self.redis_client.flushall())
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

    def tearDown(self):
        super().tearDown()
        RedisInstrumentor().uninstrument()

    def _check_span(self, span, name):
        self.assertEqual(span.name, name)
        self.assertIs(span.status.status_code, trace.StatusCode.UNSET)

    def test_basics(self):
        self.assertIsNone(async_call(self.redis_client.get("cheese")))
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "GET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "GET ?"
        )
        self.assertEqual(span.attributes.get("db.redis.args_length"), 2)

    def test_execute_command_traced_full_time(self):
        """Command should be traced for coroutine execution time, not creation time."""
        coro_created_time = None
        finish_time = None

        async def pipeline_simple():
            nonlocal coro_created_time
            nonlocal finish_time

            # delay coroutine creation from coroutine execution
            coro = self.redis_client.get("foo")
            coro_created_time = time_ns()
            await coro
            finish_time = time_ns()

        async_call(pipeline_simple())

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertTrue(span.start_time > coro_created_time)
        self.assertTrue(span.end_time < finish_time)

    def test_pipeline_traced(self):
        async def pipeline_simple():
            async with self.redis_client.pipeline(
                transaction=False
            ) as pipeline:
                pipeline.set("blah", 32)
                pipeline.rpush("foo", "éé")
                pipeline.hgetall("xxx")
                await pipeline.execute()

        async_call(pipeline_simple())

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "SET RPUSH HGETALL")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT),
            "SET ? ?\nRPUSH ? ?\nHGETALL ?",
        )
        self.assertEqual(span.attributes.get("db.redis.pipeline_length"), 3)

    def test_pipeline_traced_full_time(self):
        """Command should be traced for coroutine execution time, not creation time."""
        coro_created_time = None
        finish_time = None

        async def pipeline_simple():
            async with self.redis_client.pipeline(
                transaction=False
            ) as pipeline:
                nonlocal coro_created_time
                nonlocal finish_time
                pipeline.set("blah", 32)
                pipeline.rpush("foo", "éé")
                pipeline.hgetall("xxx")

                # delay coroutine creation from coroutine execution
                coro = pipeline.execute()
                coro_created_time = time_ns()
                await coro
                finish_time = time_ns()

        async_call(pipeline_simple())

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertTrue(span.start_time > coro_created_time)
        self.assertTrue(span.end_time < finish_time)

    def test_parent(self):
        """Ensure OpenTelemetry works with redis."""
        ot_tracer = trace.get_tracer("redis_svc")

        with ot_tracer.start_as_current_span("redis_get"):
            self.assertIsNone(async_call(self.redis_client.get("cheese")))

        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        child_span, parent_span = spans[0], spans[1]

        # confirm the parenting
        self.assertIsNone(parent_span.parent)
        self.assertIs(child_span.parent, parent_span.get_span_context())

        self.assertEqual(parent_span.name, "redis_get")
        self.assertEqual(parent_span.instrumentation_info.name, "redis_svc")

        self.assertEqual(child_span.name, "GET")


class TestRedisDBIndexInstrument(TestBase):
    def setUp(self):
        super().setUp()
        self.redis_client = redis.Redis(port=6379, db=10)
        self.redis_client.flushall()
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)

    def tearDown(self):
        RedisInstrumentor().uninstrument()
        super().tearDown()

    def _check_span(self, span, name):
        self.assertEqual(span.name, name)
        self.assertIs(span.status.status_code, trace.StatusCode.UNSET)
        self.assertEqual(
            span.attributes[SpanAttributes.NET_PEER_NAME], "localhost"
        )
        self.assertEqual(span.attributes[SpanAttributes.NET_PEER_PORT], 6379)
        self.assertEqual(
            span.attributes[SpanAttributes.DB_REDIS_DATABASE_INDEX], 10
        )

    def test_get(self):
        self.assertIsNone(self.redis_client.get("foo"))
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self._check_span(span, "GET")
        self.assertEqual(
            span.attributes.get(SpanAttributes.DB_STATEMENT), "GET ?"
        )


class TestRedisearchInstrument(TestBase):
    def setUp(self):
        super().setUp()
        self.redis_client = redis.Redis(port=6379)
        self.redis_client.flushall()
        self.embedding_dim = 256
        RedisInstrumentor().instrument(tracer_provider=self.tracer_provider)
        self.prepare_data()
        self.create_index()

    def tearDown(self):
        RedisInstrumentor().uninstrument()
        super().tearDown()

    def prepare_data(self):
        try:
            self.redis_client.ft("idx:test_vss").dropindex(True)
        except ResponseError:
            print("No such index")
        item = {
            "name": "test",
            "value": "test_value",
            "embeddings": [0.1] * 256,
        }
        pipeline = self.redis_client.pipeline()
        pipeline.json().set("test:001", "$", item)
        res = pipeline.execute()
        assert False not in res

    def create_index(self):
        schema = (
            TextField("$.name", no_stem=True, as_name="name"),
            TextField("$.value", no_stem=True, as_name="value"),
            VectorField(
                "$.embeddings",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="vector",
            ),
        )
        definition = IndexDefinition(
            prefix=["test:"], index_type=IndexType.JSON
        )
        res = self.redis_client.ft("idx:test_vss").create_index(
            fields=schema, definition=definition
        )
        assert "OK" in str(res)

    def test_redis_create_index(self):
        spans = self.memory_exporter.get_finished_spans()
        span = next(
            span for span in spans if span.name == "redis.create_index"
        )
        assert "redis.create_index.fields" in span.attributes

    def test_redis_query(self):
        query = "@name:test"
        self.redis_client.ft("idx:test_vss").search(Query(query))

        spans = self.memory_exporter.get_finished_spans()
        span = next(span for span in spans if span.name == "redis.search")

        assert span.attributes.get("redis.search.query") == query
        assert span.attributes.get("redis.search.total") == 1
