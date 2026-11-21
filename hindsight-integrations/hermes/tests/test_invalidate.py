"""Recall identifies a fact; curation retires that exact fact in one writable bank."""

import json
import threading
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


@pytest.mark.parametrize("reason", [None, "User moved to Lisbon; Porto is incorrect as the current city.", ""])
def test_invalidate_dispatches_one_curation_request(provider, reason):
    instance, fake = provider({"bank_id": "team", "additional_banks": ["mirror"]})
    args = {"memory_id": "porto"}
    if reason is not None:
        args["reason"] = reason
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", args))

    assert result == {
        "result": "Memory state=invalidated: User lives in Porto.\n"
        "Copies in other writable banks need their own hindsight_invalidate call."
    }
    assert len(fake.updates) == 1
    call = fake.updates[0]
    assert call["bank_id"] == "team" and call["memory_id"] == "porto"
    assert call["_request_timeout"] == float(instance._timeout)
    body = {"state": "invalidated", "resolve_entities": True}
    body["reason"] = reason or None
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
    assert result == {"result": "1. [memory_id=porto, type=world] User lives in Porto."}
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


def test_explicit_types_keep_bank_attribution_ranking_and_each_copy(provider):
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
        "3. [memory_id=shared-0, bank_id=shared, type=world] duplicate",
        "4. [memory_id=shared-1, bank_id=shared, type=world] shared fact",
        "5. [memory_id=vault-0, bank_id=vault, type=world] vault fact",
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
                read=AsyncMock(return_value=b'{"id":"porto","state":"invalidated","text":"User lives in Porto."}'),
            )
        )

    monkeypatch.setattr(client.memory.api_client.rest_client, "request", request)
    args = {"memory_id": "porto"}
    if with_reason:
        args["reason"] = "Moved to Lisbon"
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", args))
    assert "Memory state=invalidated: User lives in Porto." in result["result"]
    assert len(sent) == 1
    call = sent[0]
    assert call["method"] == "PATCH"
    assert call["url"] == f"{base_url}/v1/default/banks/team%2Falpha/memories/porto"
    expected = {"state": "invalidated", "resolve_entities": True, "reason": None}
    if with_reason:
        expected["reason"] = "Moved to Lisbon"
    assert call["body"] == expected
    assert call["headers"]["Authorization"] == f"Bearer {key}"
    assert call["_request_timeout"] == 17.0
    instance.shutdown()


RAW_ID = "4bd2133b-7137-4568-a1c5-789520f026e4"
OBSERVATION_ID = "c17a399a-57ca-427c-89f7-a4fd2bc68e4c"


@pytest.mark.parametrize("bank_id", [None, ""])
def test_null_or_empty_bank_defaults_to_primary(provider, bank_id):
    instance, fake = provider({"bank_id": "team"})
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": RAW_ID, "bank_id": bank_id}))
    assert result["result"].startswith("Memory state=invalidated:")
    assert [call["bank_id"] for call in fake.updates] == ["team"]
    instance.shutdown()


@pytest.mark.parametrize("explicit_types", [False, True])
def test_multi_bank_copies_can_be_curated_individually(provider, explicit_types):
    from hindsight_client_api.models.recall_response import RecallResponse
    from hindsight_client_api.models.recall_result import RecallResult

    ids = {"team": RAW_ID, "mirror": "04028fc7-5766-4004-984e-5ded8064e42c"}

    class CopiesClient(FakeClient):
        async def arecall(self, **kwargs):
            self.recalls.append(kwargs)
            return RecallResponse(
                results=[RecallResult(id=ids[kwargs["bank_id"]], text="User lives in Porto.", type="world")]
            )

    instance, fake = provider({"bank_id": "team", "additional_banks": ["mirror"]}, client=CopiesClient())
    args = {"query": "Porto"}
    if explicit_types:
        args["types"] = ["world", "experience"]
    result = json.loads(instance.handle_tool_call("hindsight_recall", args))
    if explicit_types:
        assert result["result"].splitlines() == [
            f"1. [memory_id={ids['team']}, bank_id=team, type=world] User lives in Porto.",
            f"2. [memory_id={ids['mirror']}, bank_id=mirror, type=world] User lives in Porto.",
        ]
        for bank, memory_id in ids.items():
            response = json.loads(
                instance.handle_tool_call("hindsight_invalidate", {"memory_id": memory_id, "bank_id": bank})
            )
            assert "other writable banks need their own" in response["result"]
        assert [call["bank_id"] for call in fake.updates] == ["team", "mirror"]
        assert [call["memory_id"] for call in fake.updates] == list(ids.values())
    else:
        assert result == {"result": "1. User lives in Porto."}
        assert fake.updates == []
    instance.shutdown()


@pytest.mark.parametrize("types", [None, ["world"]])
def test_single_bank_labels_are_opt_in(provider, types):
    instance, _ = provider(client=FakeClient(recall_texts=["User lives in Porto."]))
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "Porto", "types": types}))
    assert result == {
        "result": "1. User lives in Porto."
        if types is None
        else "1. [memory_id=memory-1, type=observation] User lives in Porto."
    }
    instance.shutdown()


def test_invalidation_discards_completed_prefetch_and_indicator(provider):
    instance, _ = provider(client=FakeClient(recall_texts=["User lives in Porto."]))
    instance._finish_prefetch("- User lives in Porto.", 1)
    assert instance.recall_status() is not None
    instance.queue_prefetch("Porto")
    instance._prefetch_thread.join(timeout=2)
    assert not instance._prefetch_thread.is_alive()
    assert instance._prefetch_result == "- User lives in Porto."
    generation = instance._prefetch_generation

    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": RAW_ID}))

    assert result["result"].startswith("Memory state=invalidated:")
    assert instance._prefetch_generation == generation + 1
    assert instance._prefetch_result == "" and instance._prefetch_count == 0
    assert not instance._last_recall_returned and instance._last_recall_count == 0
    assert instance.recall_status() is None
    assert instance.prefetch("next turn") == ""
    instance.shutdown()


def test_invalidation_discards_a_delayed_prefetch_worker(provider, monkeypatch):
    instance, _ = provider(client=FakeClient(recall_texts=["User lives in Porto."]))
    original = instance._do_recall
    started, release = threading.Event(), threading.Event()

    def delayed_recall(query: str) -> tuple[str, int]:
        response = original(query)
        started.set()
        assert release.wait(timeout=5)
        return response

    monkeypatch.setattr(instance, "_do_recall", delayed_recall)
    instance.queue_prefetch("Porto")
    try:
        assert started.wait(timeout=2)
        generation = instance._prefetch_generation
        result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": RAW_ID}))
        assert result["result"].startswith("Memory state=invalidated:")
        assert instance._prefetch_generation == generation + 1
    finally:
        release.set()
        instance._prefetch_thread.join(timeout=2)
    assert not instance._prefetch_thread.is_alive()
    assert instance._prefetch_result == "" and instance._prefetch_count == 0
    assert instance.prefetch("next turn") == ""
    assert instance.recall_status() is None
    instance.shutdown()


def test_observation_rejection_preserves_cached_context(provider):
    from hindsight_client_api.exceptions import ApiException

    class ObservationClient(FakeClient):
        async def _update_memory(self, **kwargs):
            self.updates.append(kwargs)
            assert kwargs["memory_id"] == OBSERVATION_ID
            raise ApiException(status=400, reason="Observations are derived and cannot be curated")

    instance, fake = provider(client=ObservationClient())
    instance._prefetch_result, instance._prefetch_count = "- still valid context", 1
    instance._finish_prefetch("- still valid context", 1)
    generation = instance._prefetch_generation
    result = instance.handle_tool_call("hindsight_invalidate", {"memory_id": OBSERVATION_ID})
    assert result.startswith("ERROR: Failed to invalidate memory:")
    assert "Observations are derived" in result
    assert len(fake.updates) == 1
    assert instance._prefetch_generation == generation
    assert instance._prefetch_result == "- still valid context" and instance._prefetch_count == 1
    assert instance.recall_status().count == 1
    instance.shutdown()


@pytest.mark.parametrize("lose_response", [False, True])
def test_invalidation_replays_after_commit_or_response_loss(provider, lose_response):
    class ReplayClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.invalidated_ids: set[str] = set()

        async def _update_memory(self, **kwargs):
            response = await super()._update_memory(**kwargs)
            self.invalidated_ids.add(kwargs["memory_id"])
            if lose_response and len(self.updates) == 1:
                raise TimeoutError("Response lost after commit")
            return response

    instance, fake = provider(client=ReplayClient())
    args = {"memory_id": RAW_ID, "reason": "Moved to Lisbon"}
    first = instance.handle_tool_call("hindsight_invalidate", args)
    assert fake.invalidated_ids == {RAW_ID}
    if lose_response:
        assert first == "ERROR: Failed to invalidate memory: Response lost after commit"
    else:
        assert json.loads(first)["result"].startswith("Memory state=invalidated:")
    second = json.loads(instance.handle_tool_call("hindsight_invalidate", args))
    assert second["result"].startswith("Memory state=invalidated:")
    assert fake.invalidated_ids == {RAW_ID}
    assert len(fake.updates) == 2
    assert fake.updates[0]["update_memory_request"].to_dict() == fake.updates[1]["update_memory_request"].to_dict()
    assert fake.retains == []
    instance.shutdown()


def test_invalidation_reports_server_state_and_text(provider):
    class ServerResultClient(FakeClient):
        async def _update_memory(self, **kwargs):
            await super()._update_memory(**kwargs)
            return {"id": RAW_ID, "state": "valid", "text": "The server returned the latest edited text."}

    instance, _ = provider(client=ServerResultClient())
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": RAW_ID}))
    assert result == {
        "result": "Memory state=valid: The server returned the latest edited text.\n"
        "Copies in other writable banks need their own hindsight_invalidate call."
    }
    instance.shutdown()


def test_tool_descriptions_limit_the_invalidation_claim():
    description = plugin.INVALIDATE_SCHEMA["description"]
    assert "Archives this raw fact and removes observations linked to it." in description
    assert "when automatic consolidation is enabled" in description
    assert "Other copies or later retains may still assert the same information." in description
    assert "restore through the server curation API, not this tool" in description
    assert "other writable banks need their own call" in description
    assert "the retained source document and its chunks still hold the original text" in description
    assert "retired, not erased" in description
    assert "With explicit types" in plugin.RECALL_SCHEMA["description"]
    assert "multiple banks are searched" in plugin.RECALL_SCHEMA["description"]
