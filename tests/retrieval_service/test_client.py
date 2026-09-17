"""Tests for retrieval client configuration and status mapping."""

from unittest.mock import Mock
import grpc
import pytest

from hcmai.retrieval_service.config import RetrievalClientSettings
from hcmai.retrieval_service.errors import map_rpc_error


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str = "error") -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


def _rpc_error(code: grpc.StatusCode, details: str = "error") -> grpc.RpcError:
    return _FakeRpcError(code, details)



def test_settings_default_to_loopback_service() -> None:
    settings = RetrievalClientSettings.from_env({})
    assert settings.target == "127.0.0.1:8002"
    assert settings.timeout_seconds == 120
    assert settings.health_timeout_seconds == 1
    assert settings.max_message_bytes == 64 * 1024 * 1024


def test_settings_reads_environment_variables() -> None:
    env = {
        "HCMAI_RETRIEVAL_TARGET": "127.0.0.1:9000",
        "HCMAI_RETRIEVAL_TIMEOUT_SECONDS": "60",
        "HCMAI_RETRIEVAL_HEALTH_TIMEOUT_SECONDS": "2.5",
        "HCMAI_RETRIEVAL_MAX_MESSAGE_MIB": "128",
    }
    settings = RetrievalClientSettings.from_env(env)
    assert settings.target == "127.0.0.1:9000"
    assert settings.timeout_seconds == 60.0
    assert settings.health_timeout_seconds == 2.5
    assert settings.max_message_bytes == 128 * 1024 * 1024


@pytest.mark.parametrize("target", ["http://127.0.0.1:8002", "https://127.0.0.1:8002"])
def test_settings_rejects_scheme_in_target(target: str) -> None:
    with pytest.raises(ValueError, match="scheme"):
        RetrievalClientSettings.from_env({"HCMAI_RETRIEVAL_TARGET": target})


@pytest.mark.parametrize("timeout", ["0", "-5", "abc"])
def test_settings_rejects_invalid_timeout(timeout: str) -> None:
    with pytest.raises(ValueError, match="timeout"):
        RetrievalClientSettings.from_env({"HCMAI_RETRIEVAL_TIMEOUT_SECONDS": timeout})


@pytest.mark.parametrize("mib", ["0", "-1", "513"])
def test_settings_rejects_out_of_bounds_message_limits(mib: str) -> None:
    with pytest.raises(ValueError, match="between 1 MiB and 512 MiB"):
        RetrievalClientSettings.from_env({"HCMAI_RETRIEVAL_MAX_MESSAGE_MIB": mib})


@pytest.mark.parametrize(
    "code",
    [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED],
)
def test_REQ_011_connectivity_codes_are_unavailable(code: grpc.StatusCode) -> None:
    assert map_rpc_error(_rpc_error(code)).category == "unavailable"


def test_REQ_011_status_code_mapping_categories() -> None:
    assert map_rpc_error(_rpc_error(grpc.StatusCode.INVALID_ARGUMENT)).category == "invalid_argument"
    assert map_rpc_error(_rpc_error(grpc.StatusCode.NOT_FOUND)).category == "not_found"
    assert map_rpc_error(_rpc_error(grpc.StatusCode.RESOURCE_EXHAUSTED)).category == "resource_exhausted"
    assert map_rpc_error(_rpc_error(grpc.StatusCode.INTERNAL)).category == "protocol"
    assert map_rpc_error(_rpc_error(grpc.StatusCode.DATA_LOSS)).category == "protocol"
