# Standalone DRES v2 Mock Server

This small local simulator implements the DRES Client API subset used by HCMAI:
login/logout, evaluation and current-task discovery, answer submission, and
result logging. It records validated requests in memory and returns an
operator-configured response. **It does not score answer meaning and does not
prove that a real DRES/VBS scorer would accept an answer.**

> **LOCAL MOCK — DO NOT EXPOSE PUBLICLY.** The `/__test/*` controls and `/monitor`
> dashboard have no authentication. Keep the default bind on loopback. The
> `X-DRES-Mock: true` header is an identification marker, not an access control.

The mock is an independent Python package. It never imports `hcmai`, registers
with the HCMAI FastAPI app, or starts an HCMAI process. Its sessions, configured
task, response scenarios, and captured records exist only in memory and reset
when the process restarts.

## Run directly on Windows

Use Python 3.11 or newer:

```powershell
cd dres-mock-server
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.venv\Scripts\dres-mock-server.exe
```

Open [http://127.0.0.1:8090/monitor](http://127.0.0.1:8090/monitor) to inspect
traffic and change the active task or the next response. Open
[http://127.0.0.1:8090/docs](http://127.0.0.1:8090/docs) to send requests from
Swagger UI. The server disables Uvicorn access logging because DRES clients
send their session token in a query parameter.

The harmless default accounts are `member-1` / `password-1` and `member-2` /
`password-2`. Change them with `DRES_MOCK_USERS_JSON` in the package-local
`.env` file before starting the server. Passwords are stored as `SecretStr` in
settings and are never returned by the test-control API.

## Run with Docker

From the repository root:

```bash
docker build -t dres-mock-server ./dres-mock-server
docker run --rm -p 127.0.0.1:8090:8090 dres-mock-server
```

The image runs as a non-root user and binds Uvicorn to `0.0.0.0` inside the
container. The documented host mapping remains on `127.0.0.1`, so the
unauthenticated test controls are reachable only from the local machine by
default. Do not change the host mapping to a public interface.

## Login and discover the task

For curl examples, replace `$SESSION` below with the returned `sessionId`.
Never commit a session value to Git:

```bash
curl -s -X POST http://127.0.0.1:8090/api/v2/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"member-1","password":"password-1"}'
```

The response has the DRES `ApiUser` shape and includes `sessionId`. Invalid
credentials return HTTP 401 with a DRES `ErrorStatus`. Use the token as the
`session` query parameter for protected requests:

```bash
curl -s 'http://127.0.0.1:8090/api/v2/client/evaluation/list?session='$SESSION
curl -s 'http://127.0.0.1:8090/api/v2/client/evaluation/currentTask/eval-vbs-local?session='$SESSION
```

The current-task response includes `name`, `taskGroup`, `taskType`, and
`duration`; it deliberately has no `taskId`, matching the DRES client contract.
Submission `taskName` is optional because DRES may infer the current task. If
supplied, it must match the active task configured in the mock.

## Submit KIS, VQA, and AVS answers

Every submission uses one `answerSets` array. The mock preserves all answer
order and exact media names/timestamps. For temporal answers, `start` and `end`
are integer milliseconds; point answers use the same value for both endpoints.

KIS temporal point:

```bash
curl -s -X POST 'http://127.0.0.1:8090/api/v2/submit/eval-vbs-local?session='$SESSION \
  -H 'Content-Type: application/json' \
  -d '{"answerSets":[{"taskName":"KIS task","answers":[{"mediaItemName":"00001","start":1234,"end":1234}]}]}'
```

VQA text answer (first switch the active task to VQA using the dashboard or the
task control API below):

```bash
curl -s -X POST 'http://127.0.0.1:8090/api/v2/submit/eval-vbs-local?session='$SESSION \
  -H 'Content-Type: application/json' \
  -d '{"answerSets":[{"taskName":"VQA task","answers":[{"text":"three"}]}]}'
```

AVS sends multiple ordered temporal answers in one answer set and one request
(first switch the active task to AVS):

```bash
curl -s -X POST 'http://127.0.0.1:8090/api/v2/submit/eval-vbs-local?session='$SESSION \
  -H 'Content-Type: application/json' \
  -d '{"answerSets":[{"taskName":"AVS task","answers":[{"mediaItemName":"00002","start":2000,"end":2000},{"mediaItemName":"00001","start":8500,"end":9100}]}]}'
```

The default response is HTTP 200 with `status: true`, verdict
`INDETERMINATE`, and description `mock accepted`. That verdict is a safe mock
default, not a semantic judgment.

## Log a result set

The mock accepts the DRES `QueryResultLog` shape and retains event/result order,
integer timestamps, empty `resultSetAvailability`, ranks, and exact media names:

```bash
curl -s -X POST 'http://127.0.0.1:8090/api/v2/log/result/eval-vbs-local?session='$SESSION \
  -H 'Content-Type: application/json' \
  -d '{
    "timestamp": 1800000000000,
    "sortType": "list",
    "resultSetAvailability": "",
    "results": [
      {"rank": 1, "answer": {"mediaItemName": "00001", "start": 1234, "end": 1234}},
      {"rank": 2, "answer": {"mediaItemName": "00002", "start": 5678, "end": 5678}}
    ],
    "events": [{
      "timestamp": 1800000000000,
      "category": "TEXT",
      "type": "SEARCH",
      "value": "person running"
    }]
  }'
```

A successful log returns only `status` and `description`. Logging the results
does not mean the mock evaluated or ranked their semantic quality.

## Configure one-shot scenarios

All controls are also available in `/monitor`. The test APIs are unauthenticated
and local-only. Each scenario is consumed only by the next authenticated,
well-formed request for that operation. A task mismatch is recorded as HTTP 412
but leaves the configured one-shot submission scenario pending.

Configure HTTP 202 with a `WRONG` verdict for the next submission:

```bash
curl -s -X PUT http://127.0.0.1:8090/__test/scenario/submission \
  -H 'Content-Type: application/json' \
  -d '{"statusCode":202,"verdict":"WRONG","delayMs":0,"malformedBody":false,"description":"queued for judging"}'
```

Configure a definitive HTTP 200 `WRONG` verdict:

```bash
curl -s -X PUT http://127.0.0.1:8090/__test/scenario/submission \
  -H 'Content-Type: application/json' \
  -d '{"statusCode":200,"verdict":"WRONG","description":"operator-selected test verdict"}'
```

Configure an HTTP 401 response for the next submission (useful for testing the
HCMAI client's expired-session retry):

```bash
curl -s -X PUT http://127.0.0.1:8090/__test/scenario/submission \
  -H 'Content-Type: application/json' \
  -d '{"statusCode":401,"verdict":"INDETERMINATE","description":"expired session"}'
```

Configure an HTTP 412 response, or let the mock produce one by sending an
explicit stale `taskName` or `taskId`:

```bash
curl -s -X PUT http://127.0.0.1:8090/__test/scenario/submission \
  -H 'Content-Type: application/json' \
  -d '{"statusCode":412,"verdict":"INDETERMINATE","description":"submission rejected"}'
```

To exercise a client timeout, ask the mock to wait five seconds before
responding. The request is captured and the scenario consumed before that
wait, so inspect `/__test/state` even if the caller times out:

```bash
curl -s -X PUT http://127.0.0.1:8090/__test/scenario/submission \
  -H 'Content-Type: application/json' \
  -d '{"statusCode":200,"verdict":"INDETERMINATE","delayMs":5000,"description":"slow response"}'
```

Return a deliberately incomplete HTTP 200/202 body to exercise a client's
malformed-response or `UNKNOWN` path:

```bash
curl -s -X PUT http://127.0.0.1:8090/__test/scenario/submission \
  -H 'Content-Type: application/json' \
  -d '{"statusCode":200,"verdict":"INDETERMINATE","malformedBody":true,"description":"incomplete response"}'
```

Switch the active task from KIS to AVS:

```bash
curl -s -X PUT http://127.0.0.1:8090/__test/task \
  -H 'Content-Type: application/json' \
  -d '{"active":true,"name":"AVS task","taskGroup":"AVS","taskType":"AVS","duration":300}'
```

Inspect received requests and the response the mock intended to return:

```bash
curl -s http://127.0.0.1:8090/__test/state
```

The monitor and state endpoint show captured payloads, actor, task name,
timestamps, HTTP outcome, configured delay, and verdict where applicable. They
show transport behavior only; they do not prove that a real VBS scorer accepts
the answer. `POST /__test/reset` clears captured records and sessions and
restores the default KIS task and response scenarios.

## Point HCMAI at the mock

For a local black-box run, set runtime environment variables in the HCMAI
process:

```dotenv
HCMAI_DRES_BASE_URL=http://127.0.0.1:8090
HCMAI_DRES_EVALUATION_ID=eval-vbs-local
HCMAI_DRES_USERS_JSON={"member-1":{"username":"member-1","password":"password-1"}}
```

This is runtime configuration only. HCMAI and the mock remain separate
processes and packages; neither imports, mounts, or starts the other.

## Validation

From this directory, install the isolated development extras and run:

```powershell
.venv\Scripts\python -m pytest -q
```

The suite uses a fresh app/state object per test and does not install the root
HCMAI package.
