import asyncio
from unittest import mock

import pytest
from app.tools.registry import Tool, ToolInput, ToolOutput, ToolRegistry, ToolSafetyError


@pytest.mark.asyncio
async def test_tool_allowlist_allows_dir():
    registry = ToolRegistry()
    res = await registry.execute("shell", {"command": "dir"})
    assert res.output is not None


@pytest.mark.asyncio
async def test_tool_rejects_mutation_without_confirm():
    registry = ToolRegistry()
    with pytest.raises(ToolSafetyError):
        await registry.execute("shell", {"command": "rm -rf /"})


@pytest.mark.asyncio
async def test_recall_and_store_fact_tool(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DB_URL", db_url)
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore

    registry = ToolRegistry()
    await registry.execute("store_fact", {"text": "The Eiffel Tower is in Paris", "tags": ["location"]}, confirm=lambda _: True)
    result = await registry.execute("recall", {"query": "Where is Eiffel?"})
    assert any("paris" in r.lower() for r in result.results)


@pytest.mark.asyncio
async def test_tool_timeout():
    class SlowInput(ToolInput):
        pass

    class SlowOutput(ToolOutput):
        done: bool = True

    async def slow_handler(data: SlowInput, timeout=None):
        await asyncio.sleep(1)
        return SlowOutput()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="slow",
            description="slow tool",
            input_model=SlowInput,
            output_model=SlowOutput,
            handler=slow_handler,
            mutating=False,
            timeout=0.01,
        )
    )
    with pytest.raises(ToolSafetyError):
        await registry.execute("slow", {})


@pytest.mark.asyncio
async def test_audit_log_for_mutating(monkeypatch):
    from loguru import logger

    events = []
    logger.add(events.append, filter=lambda rec: rec["extra"].get("audit") is True)

    registry = ToolRegistry()
    await registry.execute("store_fact", {"text": "mutating fact"}, confirm=lambda _: True)
    assert any("mutating_tool" in str(e) for e in events)


@pytest.mark.asyncio
async def test_requires_approval_blocks():
    registry = ToolRegistry(confirm_mutating=False)
    with pytest.raises(ToolSafetyError):
        await registry.execute("store_fact", {"text": "blocked fact"})


@pytest.mark.asyncio
async def test_read_write_list_file_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore

    registry = ToolRegistry()
    # write without confirm should fail
    with pytest.raises(ToolSafetyError):
        await registry.execute("write_file", {"path": "foo.txt", "content": "hi"})
    # with confirm succeeds
    await registry.execute("write_file", {"path": "foo.txt", "content": "hi"}, confirm=lambda _: True)
    read = await registry.execute("read_file", {"path": "foo.txt"})
    assert read.content == "hi"
    listing = await registry.execute("list_directory", {"path": "."})
    assert "foo.txt" in listing.entries or "foo.txt/" in listing.entries


@pytest.mark.asyncio
async def test_run_terminal_command_approval_and_capture(monkeypatch):
    registry = ToolRegistry()
    with pytest.raises(ToolSafetyError):
        await registry.execute("run_terminal_command", {"command": "echo hi"})
    with mock.patch("subprocess.run") as mrun:
        mrun.return_value = mock.Mock(stdout="out", stderr="err", returncode=0)
        res = await registry.execute("run_terminal_command", {"command": "echo hi"}, confirm=lambda _: True)
        assert res.stdout == "out"
        assert res.stderr == "err"


@pytest.mark.asyncio
async def test_system_status(monkeypatch):
    with mock.patch("psutil.cpu_percent", return_value=12.3), mock.patch(
        "psutil.virtual_memory"
    ) as mvm, mock.patch("psutil.sensors_battery") as mbat:
        mvm.return_value = mock.Mock(percent=45.6)
        mbat.return_value = mock.Mock(percent=78.9)
        registry = ToolRegistry()
        res = await registry.execute("system_status", {})
        assert res.cpu_percent == 12.3
        assert res.memory_percent == 45.6
        assert res.battery_percent == 78.9


@pytest.mark.asyncio
async def test_open_application_allowlist(monkeypatch):
    with mock.patch("subprocess.Popen") as mp:
        registry = ToolRegistry()
        await registry.execute("open_application", {"app": "code"}, confirm=lambda _: True)
        mp.assert_called()


@pytest.mark.asyncio
async def test_capture_screen_tool_encodes_base64():
    registry = ToolRegistry()

    fake_image = b"\xff\xd8\xff\xdbdummyjpeg"
    with mock.patch.object(
        registry,
        "_capture_screen_bytes",
        return_value=(fake_image, 1024, 640, "image/jpeg"),
    ):
        result = await registry.execute("capture_screen", {"max_width": 1024, "quality": 70})
    assert result.mime_type == "image/jpeg"
    assert result.width == 1024
    assert result.height == 640
    import base64

    assert base64.b64decode(result.image_base64) == fake_image
