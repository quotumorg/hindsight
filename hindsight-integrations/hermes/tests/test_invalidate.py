"""Recall identifies a fact; curation retires that exact fact in one writable bank."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import hindsight_hermes as plugin
from conftest import FakeClient


@pytest.mark.parametrize("mode", ["hybrid", "tools", "context"])
def test_invalidate_schema_follows_memory_mode(provider, mode):
    instance, _ = provider({"memory_mode": mode})
    schemas = {schema["name"]: schema for schema in instance.get_tool_schemas()}
    if mode == "context":
        assert schemas == {}
    else:
        assert set(schemas) == set(instance._TOOL_HANDLERS)
        params = schemas["hindsight_invalidate"]["parameters"]
        assert params["required"] == ["memory_id"]
        assert set(params["properties"]) == {"memory_id", "reason", "bank_id"}
    instance.shutdown()


def test_invalidate_is_enabled_by_default(provider):
    instance, _ = provider()
    assert "hindsight_invalidate" in {schema["name"] for schema in instance.get_tool_schemas()}
    instance.shutdown()


@pytest.mark.parametrize("tool", list(plugin.HindsightMemoryProvider._TOOL_HANDLERS))
def test_context_mode_rejects_tool_dispatch(provider, tool):
    instance, fake = provider({"memory_mode": "context"})
    result = instance.handle_tool_call(tool, {"memory_id": "porto", "query": "Porto", "content": "Porto"})
    assert result == "ERROR: Memory tools are unavailable in context mode"
    assert fake.updates == fake.recalls == fake.reflects == fake.retains == []
    instance.shutdown()


@pytest.mark.parametrize("reason", [None, "User moved to Lisbon; Porto is incorrect as the current city.", ""])
def test_invalidate_dispatches_one_curation_request(provider, reason):
    instance, fake = provider({"bank_id": "team", "additional_banks": ["mirror"]})
    args = {"memory_id": "porto"}
    if reason is not None:
        args["reason"] = reason
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", args))

    assert result == {"result": "Memory invalidated (kept for audit, excluded from recall)."}
    assert len(fake.updates) == 1
    call = fake.updates[0]
    assert call["bank_id"] == "team" and call["memory_id"] == "porto"
    assert call["_request_timeout"] == float(instance._timeout)
    body = {"state": "invalidated", "resolve_entities": True}
    if reason is not None:
        body["reason"] = reason
    assert call["update_memory_request"].to_dict() == body
    assert fake.retains == []
    instance.shutdown()


def test_invalidate_uses_the_current_session_bank(provider):
    instance, fake = provider({"bank_id": "fallback", "bank_id_template": "hermes-{session}"})
    instance.on_session_switch("session-2")
    instance.handle_tool_call("hindsight_invalidate", {"memory_id": "porto"})
    assert [call["bank_id"] for call in fake.updates] == ["hermes-session-2"]
    instance.shutdown()


def test_invalidate_uses_the_bank_env_override(provider):
    from conftest import SECRETS

    SECRETS["HINDSIGHT_BANK_ID"] = "env-bank"
    instance, fake = provider({"bank_id": "config-bank"})
    instance.handle_tool_call("hindsight_invalidate", {"memory_id": "porto"})
    assert [call["bank_id"] for call in fake.updates] == ["env-bank"]
    instance.shutdown()


@pytest.mark.parametrize("bank_id", ["vault", "unconfigured"])
def test_invalidate_rejects_read_only_and_unconfigured_banks(provider, bank_id):
    instance, fake = provider({"bank_id": "team", "recall_additional_banks": ["vault"]})
    result = instance.handle_tool_call("hindsight_invalidate", {"memory_id": "porto", "bank_id": bank_id})
    assert result == f"ERROR: Failed to invalidate memory: Bank '{bank_id}' is not writable by this provider"
    assert fake.updates == []
    instance.shutdown()


def test_invalidate_targets_an_explicit_writable_bank_without_fanout(provider):
    instance, fake = provider(
        {"bank_id": "team", "additional_banks": ["shared"], "recall_additional_banks": ["shared"]}
    )
    instance.handle_tool_call("hindsight_invalidate", {"memory_id": "porto", "bank_id": "shared"})
    assert [call["bank_id"] for call in fake.updates] == ["shared"]
    instance.shutdown()


def test_invalidate_requires_a_memory_id(provider):
    instance, fake = provider()
    assert instance.handle_tool_call("hindsight_invalidate", {}) == "ERROR: Missing required parameter: memory_id"
    assert fake.updates == []
    instance.shutdown()


@pytest.mark.parametrize("error", ["not found", "observations are derived", "forbidden", "server error", "timeout"])
def test_invalidate_errors_are_logged_and_reported(provider, caplog, error):
    from hindsight_client_api.exceptions import ApiException

    class FailingClient(FakeClient):
        async def _update_memory(self, **kwargs):
            if error == "timeout":
                raise TimeoutError()
            status = {"not found": 404, "observations are derived": 400, "forbidden": 403, "server error": 500}[error]
            raise ApiException(status=status, reason=error)

    instance, _ = provider(client=FailingClient())
    result = instance.handle_tool_call("hindsight_invalidate", {"memory_id": "porto"})
    assert result.startswith("ERROR: Failed to invalidate memory:")
    assert ("TimeoutError" if error == "timeout" else error) in result
    assert "hindsight_invalidate failed" in caplog.text
    assert "Memory invalidated" not in result
    instance.shutdown()


@pytest.mark.parametrize("types", [[], "world", ["unknown"], [None]])
def test_recall_rejects_invalid_type_overrides(provider, types):
    instance, fake = provider()
    result = instance.handle_tool_call("hindsight_recall", {"query": "Porto", "types": types})
    assert result.startswith("ERROR: Failed to search memory: types must be a non-empty list")
    assert fake.recalls == []
    instance.shutdown()


def test_recall_exposes_raw_fact_ids_without_changing_defaults(provider):
    from hindsight_client_api.models.recall_response import RecallResponse
    from hindsight_client_api.models.recall_result import RecallResult

    class CityClient(FakeClient):
        async def arecall(self, **kwargs):
            self.recalls.append(kwargs)
            if kwargs["types"] == ["world", "experience"]:
                return RecallResponse(results=[RecallResult(id="porto", text="User lives in Porto.", type="world")])
            return RecallResponse(results=[RecallResult(id="obs", text="User lives in Porto.", type="observation")])

    instance, fake = provider({"recall_sync": True}, client=CityClient())
    result = json.loads(
        instance.handle_tool_call("hindsight_recall", {"query": "Porto", "types": ["world", "experience"]})
    )
    assert result == {"result": "1. [memory_id=porto, bank_id=hermes, type=world] User lives in Porto."}
    instance.handle_tool_call("hindsight_invalidate", {"memory_id": "porto", "reason": "Moved to Lisbon"})
    assert fake.updates[0]["memory_id"] == "porto"
    # The recording fake does not implement server curation; this checks that a
    # one-call override cannot change the next automatic recall's types or text format.
    assert "- User lives in Porto." in instance.prefetch("Porto")
    assert instance._recall_types == ["observation"]
    assert [call["types"] for call in fake.recalls] == [["world", "experience"], ["observation"]]
    assert fake.recalls[0]["budget"] == fake.recalls[1]["budget"] == instance._budget
    assert fake.recalls[0]["max_tokens"] == fake.recalls[1]["max_tokens"] == instance._recall_max_tokens
    instance.shutdown()


def test_recall_keeps_bank_attribution_ranking_and_deduplication(provider):
    from hindsight_client_api.models.recall_response import RecallResponse
    from hindsight_client_api.models.recall_result import RecallResult

    class BankClient(FakeClient):
        async def arecall(self, **kwargs):
            self.recalls.append(kwargs)
            bank = kwargs["bank_id"]
            texts = {"team": ["first", "duplicate"], "shared": ["duplicate", "shared fact"], "vault": ["vault fact"]}
            return RecallResponse(
                results=[RecallResult(id=f"{bank}-{i}", text=text, type="world") for i, text in enumerate(texts[bank])]
            )

    instance, fake = provider(
        {"bank_id": "team", "additional_banks": ["shared"], "recall_additional_banks": ["vault"]}, client=BankClient()
    )
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q", "types": ["world"]}))
    assert result["result"].splitlines() == [
        "1. [memory_id=team-0, bank_id=team, type=world] first",
        "2. [memory_id=team-1, bank_id=team, type=world] duplicate",
        "3. [memory_id=shared-1, bank_id=shared, type=world] shared fact",
        "4. [memory_id=vault-0, bank_id=vault, type=world] vault fact",
    ]
    assert [call["types"] for call in fake.recalls] == [["world"]] * 3
    instance.shutdown()


@pytest.mark.parametrize("mode", ["cloud", "local_external", "local_embedded"])
@pytest.mark.parametrize("with_reason", [False, True])
def test_invalidate_sdk_serializes_patch_with_auth_and_bank(provider, monkeypatch, mode, with_reason):
    """Use the real SDK down to its transport boundary; no socket or server is needed."""
    from hindsight_client_api.rest import RESTResponse
    from hindsight_hermes.embedded import LocalRuntimeStatus

    instance, _ = provider({"bank_id": "team/alpha", "api_url": "https://hs.test/prefix", "timeout": 17})
    instance._mode = mode
    if mode == "local_embedded":
        profile_env = plugin._embedded_profile_env_path(instance._config)
        profile_env.parent.mkdir(parents=True, exist_ok=True)
        profile_env.write_text("HINDSIGHT_API_TENANT_API_KEY=tenant-test-key\n")
        monkeypatch.setattr(plugin, "_check_local_runtime", lambda: LocalRuntimeStatus(available=True))
        monkeypatch.setattr(plugin, "_start_daemon", lambda *a: "http://127.0.0.1:54321")
        monkeypatch.setattr(instance, "_announce_slow_first_start", lambda *a: None)
        client = instance._new_embedded_client()
        base_url, key = "http://127.0.0.1:54321", "tenant-test-key"
    else:
        client = plugin.HindsightMemoryProvider._new_cloud_client(instance)
        base_url, key = "https://hs.test/prefix", "test-key"
    instance._client = client
    sent = []

    async def request(method, url, **kwargs):
        sent.append({"method": method, "url": url, **kwargs})
        return RESTResponse(
            SimpleNamespace(
                status=200,
                reason="OK",
                headers={"content-type": "application/json"},
                read=AsyncMock(return_value=b'{"id":"porto","state":"invalidated"}'),
            )
        )

    monkeypatch.setattr(client.memory.api_client.rest_client, "request", request)
    args = {"memory_id": "porto"}
    if with_reason:
        args["reason"] = "Moved to Lisbon"
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", args))
    assert "Memory invalidated" in result["result"]
    assert len(sent) == 1
    call = sent[0]
    assert call["method"] == "PATCH"
    assert call["url"] == f"{base_url}/v1/default/banks/team%2Falpha/memories/porto"
    expected = {"state": "invalidated", "resolve_entities": True}
    if with_reason:
        expected["reason"] = "Moved to Lisbon"
    assert call["body"] == expected
    assert call["headers"]["Authorization"] == f"Bearer {key}"
    assert call["_request_timeout"] == 17.0
    instance.shutdown()
