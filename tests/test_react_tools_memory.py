from friday.memory import VectorMemory
from friday.react_loop import ReActLoop
from friday.reasoning import generate_cot_trace
from friday.tools.tool_executor import ToolExecutor


class FakeEmbeddingModel:
    def encode(self, text):
        lowered = text.lower()
        return [
            1.0 if any(word in lowered for word in ("friday", "planner", "chain")) else 0.0,
            1.0 if any(word in lowered for word in ("pizza", "recipe", "food")) else 0.0,
            1.0,
        ]


def test_cot_trace_output_format():
    trace = generate_cot_trace(
        "coding:requires_code_execution",
        {
            "text": "write merge sort",
            "category": "coding",
            "assumptions": ["Use Python unless specified."],
            "tool_hints": ["code_execution"],
            "risk_level": "low",
        },
    )

    assert trace
    assert all(item.startswith("thinking:") for item in trace)
    assert "code_execution" in " ".join(trace)


def test_react_loop_short_circuits_without_tool():
    loop = ReActLoop()
    result = loop.run("explain what a planner does")

    assert result.final_answer == "explain what a planner does"
    assert result.iterations == 1
    assert result.forced_final is False


def test_react_loop_terminates_after_max_iterations():
    def reasoner(prompt, observations):
        return {"final": False, "tool": "calendar_api", "query": "schedule soon"}

    loop = ReActLoop(reasoner=reasoner, max_iterations=5)
    result = loop.run("schedule soon")

    assert result.iterations == 2
    assert result.forced_final is True
    assert "failed after retry" in result.final_answer


def test_tool_executor_code_exec_success():
    result = ToolExecutor().execute("code_exec", "calculate 2 + 3 * 4")

    assert result["tool"] == "code_exec"
    assert result["success"] is True
    assert result["result"] == "14"


def test_tool_executor_calendar_failure_retries_once():
    result = ToolExecutor().execute("calendar_api", "schedule meeting soon")

    assert result["tool"] == "calendar_api"
    assert result["success"] is False
    assert result["attempt"] == 2


def test_tool_executor_knowledge_base_uses_vector_memory():
    memory = VectorMemory(model=FakeEmbeddingModel())
    memory.store("Friday planner uses chain-linked reasoning", tags=["friday"])
    memory.store("Pizza dough needs flour", tags=["food"])

    result = ToolExecutor().execute("knowledge_base", "Friday chain planner", memory=memory)

    assert result["success"] is True
    assert "Friday planner" in result["result"][0]


def test_vector_memory_retrieval_accuracy():
    memory = VectorMemory(model=FakeEmbeddingModel())
    memory.store("Friday planner uses chain-linked reasoning", tags=["architecture"])
    memory.store("Pizza dough needs flour and yeast", tags=["cooking"])

    matches = memory.retrieve_relevant("How does the Friday planner work?", top_k=1)

    assert matches[0]["text"] == "Friday planner uses chain-linked reasoning"
    assert matches[0]["score"] > 0.8
