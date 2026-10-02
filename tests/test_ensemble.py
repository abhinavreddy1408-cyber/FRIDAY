from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from friday.ensemble import (
    CANONICAL_MODEL_MAP,
    ENSEMBLE_KEYS,
    EnsembleRunner,
    call_model,
    check_safety,
    get_ensemble_config,
    resolve_api_key,
)
from friday.llm import LLMBrain


@pytest.mark.asyncio
async def test_call_model_success():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello World"}}]
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        res = await call_model(
            model_id="qwen2.5-coder:3b-instruct-q4_K_M",
            ensemble_name="coding",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert res == "Hello World"


@pytest.mark.asyncio
async def test_call_model_ollama_auth(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "ollama")
    mock_resp_success = MagicMock(spec=httpx.Response)
    mock_resp_success.status_code = 200
    mock_resp_success.json.return_value = {
        "choices": [{"message": {"content": "Success after auth retry"}}]
    }
    mock_resp_success.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp_success)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        res = await call_model(
            model_id="qwen2.5-coder:3b-instruct-q4_K_M",
            ensemble_name="coding",
            messages=[{"role": "user", "content": "hi"}],
            role="primary_coder",
        )
        assert res == "Success after auth retry"
        assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_call_model_exception():
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=httpx.RequestError("Network down"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        res = await call_model(
            model_id="qwen2.5-coder:3b-instruct-q4_K_M",
            ensemble_name="coding",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert res is None


@pytest.mark.asyncio
@patch("friday.ensemble.call_canonical_model")
async def test_check_safety_safe(mock_canonical):
    mock_canonical.return_value = "safe"
    is_safe = await check_safety("some output")
    assert is_safe is True
    assert mock_canonical.call_count == 2


@pytest.mark.asyncio
@patch("friday.ensemble.call_canonical_model")
async def test_check_safety_unsafe(mock_canonical):
    mock_canonical.side_effect = ["unsafe", "unsafe text"]
    is_safe = await check_safety("some output")
    assert is_safe is False


@pytest.mark.asyncio
@patch("friday.ensemble.call_canonical_model")
async def test_check_safety_one_failed_one_safe(mock_canonical):
    mock_canonical.side_effect = ["safe", None]
    is_safe = await check_safety("some output")
    assert is_safe is True


@pytest.mark.asyncio
@patch("friday.ensemble.call_canonical_model")
async def test_check_safety_both_fail_open(mock_canonical):
    mock_canonical.side_effect = [None, None]
    is_safe = await check_safety("some output")
    assert is_safe is True



def test_ensemble_registry_has_no_literal_api_keys():
    assert get_ensemble_config("research").name == "search"
    secret_prefix = "nv" + "api-"
    assert all(secret_prefix not in value for value in ENSEMBLE_KEYS.values())


def test_role_key_resolution_and_provided_alias(monkeypatch):
    assert (
        resolve_api_key(
            "coding",
            "qwen2.5-coder:3b-instruct-q4_K_M",
            role="primary_coder",
        )
        == "ollama"
    )

    assert (
        resolve_api_key(
            "coding",
            "qwen2.5-coder:3b-instruct-q4_K_M",
            role="fast_coder",
        )
        == "ollama"
    )


def test_canonical_model_map_has_pdf_models():
    assert len(CANONICAL_MODEL_MAP) == 22
    assert CANONICAL_MODEL_MAP["coding_flagship"] == "qwen2.5-coder:3b-instruct-q4_K_M"
    assert CANONICAL_MODEL_MAP["safety_classify_12b"] == "qwen2.5:1.5b-instruct-q4_K_M"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_coding_majority(mock_safety, mock_call):
    """Primary coding model succeeds on first call."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if model_id == "qwen2.5-coder:3b-instruct-q4_K_M":
            return "python_code_A"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("coding", [{"role": "user", "content": "write quicksort"}], "write quicksort")
    assert res == "python_code_A"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_coding_split(mock_safety, mock_call):
    """Primary coding model fails, fallback succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        role = kwargs.get("role")
        if role == "primary":
            return None  # primary fails
        elif role == "fallback":
            return "code_B"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("coding", [{"role": "user", "content": "write quicksort"}], "write quicksort")
    assert res == "code_B"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_reasoning(mock_safety, mock_call):
    """Primary reasoning model (kimi-k2.6) succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if model_id == "qwen2.5:3b-instruct-q4_K_M" and kwargs.get("role") == "primary":
            return "Reasoning response from primary"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("reasoning", [{"role": "user", "content": "solve riddle"}], "solve riddle")
    assert res == "Reasoning response from primary"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_search(mock_safety, mock_call):
    """Primary search model (kimi-k2.6) succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if kwargs.get("role") == "primary":
            return "Search results from primary"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("search", [{"role": "user", "content": "weather in Paris"}], "weather in Paris")
    assert res == "Search results from primary"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_brainstorming(mock_safety, mock_call):
    """Primary brainstorming model (kimi-k2.6) succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if kwargs.get("role") == "primary":
            return "Creative brainstorming ideas"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("brainstorming", [{"role": "user", "content": "app ideas"}], "app ideas")
    assert res == "Creative brainstorming ideas"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_document(mock_safety, mock_call):
    """Primary document model (nemotron-super-49b) succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if model_id == "qwen2.5:3b-instruct-q4_K_M" and kwargs.get("role") == "primary":
            return "Draft A"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("document", [{"role": "user", "content": "write cover letter"}], "write cover letter")
    assert res == "Draft A"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_file_analysis(mock_safety, mock_call):
    """Primary file_analysis model (nemotron-super-49b) succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if model_id == "qwen2.5:3b-instruct-q4_K_M" and kwargs.get("role") == "primary":
            return "Extracted invoice facts"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("file_analysis", [{"role": "user", "content": "analyze invoice"}], "analyze invoice")
    assert res == "Extracted invoice facts"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_screen_fast(mock_safety, mock_call):
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if model_id == "qwen2.5:3b-instruct-q4_K_M":
            return "Fast screen assessment. confidence: high. Click button."
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("screen_interaction", [{"role": "user", "content": "find next button"}], "find next button")
    assert res == "Fast screen assessment. confidence: high. Click button."


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_screen_detailed(mock_safety, mock_call):
    """Primary screen model fails, fallback succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        role = kwargs.get("role")
        if role == "primary":
            return None  # primary fails
        elif role == "fallback":
            return "Detailed screen analysis from fallback."
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("screen_interaction", [{"role": "user", "content": "find next button"}], "find next button")
    assert res == "Detailed screen analysis from fallback."


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_cascade_fallback_tier1(mock_safety, mock_call):
    """Unknown type defaults to cascade_medium; primary succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        if kwargs.get("role") == "primary":
            return "Valid cascade response"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("unknown_type", [{"role": "user", "content": "hello"}], "hello")
    assert res == "Valid cascade response"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_cascade_fallback_tier2(mock_safety, mock_call):
    """Unknown type defaults to cascade_medium; primary fails, fallback succeeds."""
    mock_safety.return_value = True

    def side_effect(model_id, ensemble_name, messages, *args, **kwargs):
        role = kwargs.get("role")
        if role == "primary":
            return None
        elif role == "fallback":
            return "Good fallback response"
        return None

    mock_call.side_effect = side_effect

    res = await EnsembleRunner.run_ensemble("unknown_type", [{"role": "user", "content": "hello"}], "hello")
    assert res == "Good fallback response"


@pytest.mark.asyncio
@patch("friday.ensemble.call_model")
@patch("friday.ensemble.check_safety")
async def test_run_ensemble_safety_blocked(mock_safety, mock_call):
    mock_safety.return_value = False
    mock_call.return_value = "Unsafe output"

    res = await EnsembleRunner.run_ensemble("coding", [{"role": "user", "content": "exploit details"}], "exploit details")
    assert res == "My apologies, Sir. The response was filtered for safety reasons."


@patch("friday.ensemble.EnsembleRunner.run_ensemble")
def test_llm_brain_query_ensemble(mock_run):
    mock_run.return_value = "Ensemble output"

    brain = LLMBrain()
    brain._ready = True
    brain._client = MagicMock()

    res = brain.query("test prompt", ensemble_type="coding", add_to_history=False)
    assert res.text == "Ensemble output"
    assert res.success is True
