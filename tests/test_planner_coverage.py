import time
from unittest.mock import MagicMock, patch
import pytest

from friday.planner import (
    Intent,
    RoutingDecision,
    PlannerDecision,
    PlanStep,
    ExecutionPlan,
    CommandRecord,
    ActionPlanner,
    _ConversationContext,
    _ProactiveSuggester,
)
from friday.reasoning import (
    AutonomousReasoningResult,
    AutonomousReasoner,
    generate_cot_trace,
)

# ── Planner Decision & Plan Models ───────────────────────────────────────────

def test_planner_decision_properties():
    dec_ignore = PlannerDecision(routing=RoutingDecision.IGNORE, intent=Intent.UNKNOWN, text="", confidence=0.0)
    assert dec_ignore.is_actionable is False
    assert dec_ignore.needs_llm is False

    dec_llm = PlannerDecision(routing=RoutingDecision.QUERY_LLM, intent=Intent.LLM_QUERY, text="hello", confidence=1.0)
    assert dec_llm.is_actionable is True
    assert dec_llm.needs_llm is True

    dec_goal = PlannerDecision(routing=RoutingDecision.CREATE_GOAL, intent=Intent.GOAL_CREATE, text="plan", confidence=1.0)
    assert dec_goal.needs_llm is True

    dec_delegate = PlannerDecision(routing=RoutingDecision.DELEGATE, intent=Intent.LOCAL_ACTION, text="test", confidence=1.0)
    assert dec_delegate.needs_llm is True

    # Test string representation
    rep = str(dec_llm)
    assert "[QUERY_LLM]" in rep
    assert "intent=LLM_QUERY" in rep
    assert "conf=100%" in rep
    assert "text='hello'" in rep

def test_execution_plan_has_agents():
    plan_empty = ExecutionPlan(steps=())
    assert plan_empty.has_agents is False

    step = PlanStep(agent="CODER", task_category="coding", ensemble_type="coding", prompt="write code")
    plan_with_step = ExecutionPlan(steps=(step,))
    assert plan_with_step.has_agents is True

# ── Conversation Context ──────────────────────────────────────────────────────

def test_conversation_context():
    context = _ConversationContext(timeout=2)
    assert context.is_follow_up() is False
    assert context.last_intent is None
    assert context.last_topic == ""
    assert context.turn_count == 0
    assert context.time_since_last == float("inf")

    context.record("open chrome browser", Intent.LOCAL_ACTION)
    assert context.is_follow_up() is True
    assert context.last_intent == Intent.LOCAL_ACTION
    assert context.last_topic == "open chrome browser"
    assert context.turn_count == 1
    assert context.time_since_last < 2

    # Sleep past timeout
    time.sleep(2.1)
    assert context.is_follow_up() is False

    # Reset
    context.record("hello", Intent.LLM_QUERY)
    context.reset()
    assert context.last_intent is None
    assert context.last_topic == ""
    assert context.turn_count == 0

def test_conversation_context_topic_extraction():
    # Test _extract_topic with filler words
    text = "can you please open the chrome browser today"
    topic = _ConversationContext._extract_topic(text)
    # fillers: can, you, please, the
    assert topic == "open chrome browser today"

# ── Proactive Suggester ────────────────────────────────────────────────────────

def test_proactive_suggester():
    suggester = _ProactiveSuggester()
    # Set cooldown to 1 second for testing
    suggester._COOLDOWN = 1

    # Triggers pattern suggestions
    res = suggester.suggest("open chrome for me", Intent.LOCAL_ACTION)
    assert res == "Want me to search for something while Chrome loads?"

    # Cool down check
    res_cooldown = suggester.suggest("open chrome for me", Intent.LOCAL_ACTION)
    assert res_cooldown is None

    # Wait for cooldown to pass
    time.sleep(1.1)

    # Trigger another suggestion
    res2 = suggester.suggest("open vscode please", Intent.LOCAL_ACTION)
    assert res2 == "Should I open the last project you were working on?"

    time.sleep(1.1)
    # Learning based suggestion
    mock_learning = MagicMock()
    mock_learning.suggest_shortcuts.return_value = [("frequent command", "fq")]
    res_learn = suggester.suggest("frequent command now", Intent.LOCAL_ACTION, learning=mock_learning)
    assert "Want me to register 'fq' as a shortcut?" in res_learn

    # No matches
    time.sleep(1.1)
    mock_learning.suggest_shortcuts.return_value = []
    res_none = suggester.suggest("do random things", Intent.LOCAL_ACTION, learning=mock_learning)
    assert res_none is None

# ── Action Planner ────────────────────────────────────────────────────────────

def test_action_planner_classify_intent():
    planner = ActionPlanner()
    assert planner.classify_intent("apply for internships today") == "goal"
    assert planner.classify_intent("help me plan a trip") == "goal"
    assert planner.classify_intent("what is the weather today?") == "question"
    assert planner.classify_intent("hello Friday!") == "chitchat"
    assert planner.classify_intent("good morning") == "chitchat"
    assert planner.classify_intent("open visual studio code") == "command"

def test_action_planner_decide_edges():
    planner = ActionPlanner()
    
    # Empty inputs
    assert planner.decide("").routing == RoutingDecision.IGNORE
    assert planner.decide("   ").routing == RoutingDecision.IGNORE

    # Wake-word-only input
    dec_wake = planner.decide("friday")
    assert dec_wake.routing == RoutingDecision.IGNORE
    assert dec_wake.confidence == 1.0

    # Shortcut resolution with learning engine
    mock_learning = MagicMock()
    mock_learning.resolve_shortcut.return_value = "volume up"
    planner_with_shortcut = ActionPlanner(learning=mock_learning)
    dec_shortcut = planner_with_shortcut.decide("vup")
    assert dec_shortcut.routing == RoutingDecision.EXECUTE_ACTION
    assert dec_shortcut.text == "volume up"

    # Shortcut resolution returns original when no learning
    planner_no_shortcut = ActionPlanner(learning=None)
    dec_no_shortcut = planner_no_shortcut.decide("vup")
    assert dec_no_shortcut.routing == RoutingDecision.QUERY_LLM

def test_action_planner_decide_escalation():
    # Query LLM with ensemble hint triggers DELEGATE
    planner = ActionPlanner()
    # Mock decide context to contain ensemble_hint
    # We patch inside decide to inject context["ensemble_hint"]
    with patch.object(ActionPlanner, "_classify") as mock_classify:
        mock_classify.return_value = PlannerDecision(
            routing=RoutingDecision.QUERY_LLM,
            intent=Intent.LLM_QUERY,
            text="write some Python script",
            confidence=0.8,
            context={"ensemble_hint": "coding"}
        )
        dec = planner.decide("write some Python script")
        assert dec.routing == RoutingDecision.DELEGATE
        assert dec.context["specialist"] == "CODER"

def test_action_planner_decide_with_ensemble_hint():
    planner = ActionPlanner()
    
    # Passes through if routing is not QUERY_LLM
    with patch.object(ActionPlanner, "decide") as mock_decide:
        mock_decide.return_value = PlannerDecision(
            routing=RoutingDecision.EXECUTE_ACTION,
            intent=Intent.LOCAL_ACTION,
            text="volume up",
            confidence=0.9
        )
        dec = planner.decide_with_ensemble_hint("volume up", ensemble_hint="coding")
        assert dec.routing == RoutingDecision.EXECUTE_ACTION

    # Escalates to DELEGATE when routing is QUERY_LLM
    with patch.object(ActionPlanner, "decide") as mock_decide:
        mock_decide.return_value = PlannerDecision(
            routing=RoutingDecision.QUERY_LLM,
            intent=Intent.LLM_QUERY,
            text="write coding script",
            confidence=0.5,
            duration=0.1
        )
        dec = planner.decide_with_ensemble_hint("write coding script", ensemble_hint="coding")
        assert dec.routing == RoutingDecision.DELEGATE
        assert dec.context["specialist"] == "CODER"
        assert dec.context["ensemble_hint"] == "coding"

def test_action_planner_decide_with_suggestion():
    planner = ActionPlanner()
    dec, sugg = planner.decide_with_suggestion("open chrome")
    assert dec.routing == RoutingDecision.EXECUTE_ACTION
    assert sugg == "Want me to search for something while Chrome loads?"

def test_action_planner_create_execution_plan():
    planner = ActionPlanner()
    
    # Local bypass routes
    dec_action = PlannerDecision(routing=RoutingDecision.EXECUTE_ACTION, intent=Intent.LOCAL_ACTION, text="volume up", confidence=1.0)
    exec_plan = planner.create_execution_plan(None, decision=dec_action)
    assert exec_plan.local_bypass is True
    assert exec_plan.steps == ()

    # Mixed category planning
    class DummyInput:
        task_category = "mixed_task"
        metadata = {"mixed_categories": ["coding", "search", "document"]}
        structured_prompt = "code search and doc"

    plan = planner.create_execution_plan(DummyInput())
    assert plan.local_bypass is False
    assert len(plan.steps) == 3
    assert plan.steps[0].agent == "CODER"
    assert plan.steps[0].task_category == "coding"
    assert plan.steps[0].ensemble_type == "coding"
    assert plan.steps[0].prompt == "code search and doc"
    assert plan.steps[0].read_only is False

    assert plan.steps[1].agent == "RESEARCH"
    assert plan.steps[1].task_category == "search"
    assert plan.steps[1].ensemble_type == "search"
    assert plan.steps[1].read_only is True
    assert plan.steps[1].dependencies == ()

    assert plan.steps[2].agent == "PRODUCTIVITY"
    assert plan.steps[2].task_category == "document"
    assert plan.steps[2].ensemble_type == "document"
    assert plan.steps[2].read_only is False
    assert plan.steps[2].dependencies == ("CODER",)

def test_action_planner_history_and_retries():
    planner = ActionPlanner()
    
    # Notify executed with empty history does nothing
    planner.notify_executed("test", "success")
    assert planner.get_last_command() is None
    assert planner.get_retry_decision() is None

    # Normal execution recording
    dec = planner.decide("open chrome")
    assert planner.get_last_command().text == "open chrome"
    
    # Update result
    planner.notify_executed("open chrome", "success outcome")
    assert planner.get_last_command().result == "success outcome"

    # Retry decision
    retry_dec = planner.get_retry_decision()
    assert retry_dec.routing == RoutingDecision.EXECUTE_ACTION
    assert retry_dec.text == "open chrome"

    # Context resets
    planner.reset_context()
    assert planner.in_conversation is False

    # Properties
    assert planner.conversation_turn == 0
    assert planner.context_topic == ""
    assert planner.time_since_last_command == float("inf")

    # History size limits
    planner._max_history = 2
    planner.decide("open chrome 1")
    planner.decide("open chrome 2")
    planner.decide("open chrome 3")
    assert len(planner._history) == 2

def test_action_planner_decide_follow_up_flow():
    planner = ActionPlanner(timeout=15)
    
    # Record a command to establish follow-up context
    planner.decide("open chrome")
    assert planner.in_conversation is True

    # High-confidence local action wins in follow-up window
    dec_local = planner.decide("volume up")
    assert dec_local.routing == RoutingDecision.EXECUTE_ACTION
    assert dec_local.intent == Intent.LOCAL_ACTION

    # Conversational follow-up otherwise
    dec_conv = planner.decide("and do some search")
    assert dec_conv.routing == RoutingDecision.QUERY_LLM
    assert dec_conv.intent == Intent.FOLLOW_UP
    assert dec_conv.context["follow_up"] is True
    assert dec_conv.context["topic"] == "open chrome"

# ── Autonomous Reasoner ───────────────────────────────────────────────────────

def test_autonomous_reasoner_missing_and_clarification():
    from types import SimpleNamespace
    reasoner = AutonomousReasoner()

    # Schedule intent missing time and participants
    res1 = reasoner.analyze("schedule a meeting", SimpleNamespace(task_category="productivity"))
    assert "time" in res1.missing_information
    assert "participants" in res1.missing_information
    assert res1.clarification_question == "Who should I include, and what exact time should I use?"

    # Schedule intent with time but missing participants
    res2 = reasoner.analyze("schedule a meeting at 2:00pm", SimpleNamespace(task_category="productivity"))
    assert "time" not in res2.missing_information
    assert "participants" in res2.missing_information
    assert res2.clarification_question == "Who should I include?"

    # Schedule intent with participant but missing time
    res3 = reasoner.analyze("schedule a meeting with John", SimpleNamespace(task_category="productivity"))
    assert "time" in res3.missing_information
    assert "participants" not in res3.missing_information
    assert res3.clarification_question == "What exact time should I schedule it for?"

    # Plan trip travel intents missing duration and budget
    res4 = reasoner.analyze("plan a trip to France", SimpleNamespace(task_category="productivity"))
    assert "duration" in res4.missing_information
    assert "budget" in res4.missing_information

    # Short vague target inputs
    res5 = reasoner.analyze("delete it", SimpleNamespace(task_category="productivity"))
    assert "target" in res5.missing_information
    assert res5.clarification_question == "What exactly should I apply that to?"

    # Document improve missing source text
    res6 = reasoner.analyze("improve", SimpleNamespace(task_category="document"))
    assert "source text" in res6.missing_information
    assert res6.clarification_question == "Send me the text you want improved, and I will revise it."

    # General missing details fallback question
    res7 = reasoner.analyze("plan a trip", SimpleNamespace(task_category="productivity"))
    assert res7.clarification_question == "I need one detail before I continue: duration, budget."

def test_autonomous_reasoner_assumptions_and_risks():
    from types import SimpleNamespace
    reasoner = AutonomousReasoner()

    # Current info assumption
    res1 = reasoner.analyze("what is the weather today?", SimpleNamespace(task_category="search"))
    assert "The request may require current information." in res1.assumptions
    assert res1.tool_hints == ("web_search",)

    # Coding python default assumption
    res2 = reasoner.analyze("write code to sort list", SimpleNamespace(task_category="coding"))
    assert "Use Python unless the user specifies another language." in res2.assumptions
    assert res2.tool_hints == ("code_execution",)

    # Productivity assumption
    res3 = reasoner.analyze("organise files", SimpleNamespace(task_category="productivity"))
    assert "Do not perform external actions until required details are clear." in res3.assumptions

    # High risk destructive action
    res4 = reasoner.analyze("delete all project databases", SimpleNamespace(task_category="productivity"))
    assert res4.risk_level == "high"
    assert res4.requires_confirmation is True
    # Verify clarification is targeting confirmation
    assert res4.clarification_question == "Before I take that action, confirm the exact target and say whether I should proceed."

    # Medium risk action
    res5 = reasoner.analyze("email the report", SimpleNamespace(task_category="productivity"))
    assert res5.risk_level == "medium"
    assert res5.requires_confirmation is True
    assert "email" in res5.tool_hints

def test_autonomous_reasoner_memory_recalls():
    from types import SimpleNamespace
    reasoner = AutonomousReasoner()

    # Memory is None
    res_no_mem = reasoner.analyze("hello", SimpleNamespace(task_category="general"), memory=None)
    assert res_no_mem.relevant_memory == ()

    # Memory throws exception
    mock_mem_fail = MagicMock()
    mock_mem_fail.recall_relevant.side_effect = Exception("DB failure")
    mock_mem_fail.build_context_summary.side_effect = Exception("DB failure 2")
    res_mem_fail = reasoner.analyze("hello", SimpleNamespace(task_category="general"), memory=mock_mem_fail)
    assert res_mem_fail.relevant_memory == ()

    # Memory success recall
    mock_mem_success = MagicMock()
    mock_entry = MagicMock()
    mock_entry.content = "Ground truth content"
    mock_mem_success.recall_relevant.return_value = [(mock_entry, 0.4)]
    res_mem_success = reasoner.analyze("hello", SimpleNamespace(task_category="general"), memory=mock_mem_success)
    assert res_mem_success.relevant_memory == ("Ground truth content",)
    assert "Relevant memory is available for prompt grounding." in res_mem_success.notes

    # Memory high distance fallback to context summary
    mock_mem_success.recall_relevant.return_value = [(mock_entry, 0.9)]
    mock_mem_success.build_context_summary.return_value = "System context summary"
    res_mem_summary = reasoner.analyze("hello", SimpleNamespace(task_category="general"), memory=mock_mem_success)
    assert res_mem_summary.relevant_memory == ("System context summary",)

def test_autonomous_reasoner_notes():
    from types import SimpleNamespace
    reasoner = AutonomousReasoner()
    
    # General brief notes
    res = reasoner.analyze("hi", SimpleNamespace(task_category="general"))
    assert "Keep response brief unless the user asks for depth." in res.notes

# ── COT Trace Generation ──────────────────────────────────────────────────────

def test_generate_cot_trace_edges():
    # Value error checks
    with pytest.raises(ValueError, match="requires a non-empty intent and context dict"):
        generate_cot_trace("", {})
    with pytest.raises(ValueError, match="requires a non-empty intent and context dict"):
        generate_cot_trace("intent", None)

    # Empty inputs trace
    trace_empty = generate_cot_trace("general", {"text": ""})
    assert "thinking: preserve the user's original goal" in trace_empty

    # Custom context values
    ctx = {
        "text": "email draft summary",
        "category": "productivity",
        "missing_information": ["time"],
        "assumptions": ["assume default email"],
        "tool_hints": ["email", "calendar"],
        "risk_level": "medium",
    }
    trace = generate_cot_trace("productivity:requires_email", ctx)
    assert "thinking: identify the request as productivity intent" in trace
    assert "thinking: preserve the user's original goal: email draft summary" in trace
    assert "thinking: ask for missing details before execution: time" in trace
    assert "thinking: use bounded assumptions: assume default email" in trace
    assert "thinking: gather tool observations from: email, calendar" in trace
    assert "thinking: apply medium risk guardrails before action" in trace
    assert "thinking: verify the result against the original request before delivery" in trace
