from friday.voice_to_prompt import VoiceToPromptAgent


def test_voice_to_prompt_sets_coding_ensemble_metadata():
    result = VoiceToPromptAgent().process("Friday, write a Python function")

    assert result.cleaned_transcript.startswith("FRIDAY")
    assert result.task_category == "coding"
    assert result.ensemble_type == "coding"
    assert result.target_agent == "CODER"
    assert result.priority == 3
    assert result.needs_clarification is False


def test_voice_to_prompt_detects_mixed_task_categories():
    result = VoiceToPromptAgent().process(
        "Friday, search NVIDIA models and also write a Python script"
    )

    assert result.task_category == "mixed_task"
    assert result.ensemble_type == "reasoning"
    assert result.metadata["mixed_categories"][:2] == ["search", "coding"]
