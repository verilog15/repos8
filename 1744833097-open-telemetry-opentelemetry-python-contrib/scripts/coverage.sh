#!/bin/bash

set -e

function cov {
    pytest \
        --cov ${1} \
        --cov-append \
        --cov-branch \
        --cov-report='' \
        ${1}
}

coverage erase

cov instrumentation/opentelemetry-instrumentation-flask
cov instrumentation/opentelemetry-instrumentation-requests
cov instrumentation/opentelemetry-instrumentation-wsgi
cov instrumentation/opentelemetry-instrumentation-aiohttp-client
cov instrumentation/opentelemetry-instrumentation-asgi


coverage report --show-missing
coverage xml
