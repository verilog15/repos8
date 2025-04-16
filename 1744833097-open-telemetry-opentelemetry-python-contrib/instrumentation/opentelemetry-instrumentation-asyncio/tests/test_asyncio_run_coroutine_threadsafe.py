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
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

# pylint: disable=no-name-in-module
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.asyncio.environment_variables import (
    OTEL_PYTHON_ASYNCIO_COROUTINE_NAMES_TO_TRACE,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import get_tracer


class TestRunCoroutineThreadSafe(TestBase):
    @patch.dict(
        "os.environ", {OTEL_PYTHON_ASYNCIO_COROUTINE_NAMES_TO_TRACE: "coro"}
    )
    def setUp(self):
        super().setUp()
        AsyncioInstrumentor().instrument()
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop.set_default_executor(self.executor)
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()

        self._tracer = get_tracer(
            __name__,
        )

    def tearDown(self):
        super().tearDown()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()

        AsyncioInstrumentor().uninstrument()

    def test_run_coroutine_threadsafe(self):
        async def coro():
            return 42

        future = asyncio.run_coroutine_threadsafe(coro(), self.loop)
        result = future.result(timeout=1)
        self.assertEqual(result, 42)
        spans = self.memory_exporter.get_finished_spans()
        assert spans
