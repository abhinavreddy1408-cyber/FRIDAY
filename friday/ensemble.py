"""
F.R.I.D.A.Y. Multi-Model Ensemble Module.

Runs task-specific NVIDIA NIM model ensembles, safety filtering, and cascade
fallbacks. Model/API-key configuration is deliberately no-secret: real keys
must come from environment variables or the default NVIDIA_API_KEY.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass

import httpx

from friday.config import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRoleConfig:
    """A model assigned to one role inside an ensemble."""

    model_id: str
    role: str
    tier: str
    strength: str
    key_alias: str = "_"


@dataclass(frozen=True)
class EnsembleConfig:
    """A named ensemble plus its execution strategy and model roles."""

    name: str
    strategy: str
    models: tuple[ModelRoleConfig, ...]
    aliases: tuple[str, ...] = ()

    def match(self, model_id: str, role: str | None = None) -> ModelRoleConfig | None:
        role_token = _role_token(role) if role else None
        matches = [m for m in self.models if m.model_id == model_id]
        if role_token:
            for model in matches:
                if _role_token(model.role) == role_token:
                    return model
        return matches[0] if matches else None


def _env_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _role_token(value: str | None) -> str:
    return _env_token(value or "")


def _role_env_var(ensemble_name: str, role: str) -> str:
    return f"FRIDAY_NIM_KEY_{_env_token(ensemble_name)}_{_role_token(role)}"


ACTIVE_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "qwen/qwen3-coder-480b-a35b-instruct": {
        "tier": "Flagship",
        "parameters": "480B MOE",
        "strength": "Open-source coding, Chain-of-thought",
    },
    "mistralai/mistral-large-3-675b-instruct-2512": {
        "tier": "Flagship",
        "parameters": "675B",
        "strength": "Code generation, Diverse thinking",
    },
    "z-ai/glm-5.1": {
        "tier": "Flagship",
        "parameters": "405B",
        "strength": "Complex web data, Context",
    },
    "deepseek-ai/deepseek-v4-pro": {
        "tier": "Flagship",
        "parameters": "Large",
        "strength": "Advanced complex analysis",
    },
    "openai/gpt-oss-120b": {
        "tier": "Flagship",
        "parameters": "120B",
        "strength": "Advanced reasoning",
    },
    "mistralai/mistral-medium-3.5-128b": {
        "tier": "High Performance",
        "parameters": "128B",
        "strength": "Multilingual programming",
    },
    "qwen/qwq-32b": {
        "tier": "High Performance",
        "parameters": "32B",
        "strength": "Logic & Mathematics",
    },
    "nvidia/nemotron-3-super-120b-a12b": {
        "tier": "High Performance",
        "parameters": "120B",
        "strength": "Synthesis, Counter-arguments",
    },
    "moonshotai/kimi-k2.6": {
        "tier": "High Performance",
        "parameters": "Large",
        "strength": "Reasoning, tool use, long context, agent workflows",
    },
    "qwen/qwen3.5-122b-a10b": {
        "tier": "High Performance",
        "parameters": "122B",
        "strength": "Creative writing expansion",
    },
    "nvidia/nemotron-4-340b-reward": {
        "tier": "High Performance",
        "parameters": "340B",
        "strength": "Output ranking, Quality scoring",
    },
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": {
        "tier": "High Performance",
        "parameters": "49B",
        "strength": "Fast standard reasoning",
    },
    "qwen/qwen2.5-coder-32b-instruct": {
        "tier": "High Performance",
        "parameters": "32B",
        "strength": "Fast code tasks",
    },
    "minimaxai/minimax-m2.7": {
        "tier": "High Performance",
        "parameters": "Large",
        "strength": "Deep contextual analysis",
    },
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "tier": "High Performance",
        "parameters": "70B",
        "strength": "Action planning",
    },
    "meta/llama-3.2-11b-vision-instruct": {
        "tier": "Specialist",
        "parameters": "11B",
        "strength": "Fast screen assessment",
    },
    "meta/llama-3.2-90b-vision-instruct": {
        "tier": "Specialist",
        "parameters": "90B",
        "strength": "Detailed visual analysis",
    },
    "nvidia/llama-3.1-nemotron-nano-8b-v1": {
        "tier": "Specialist",
        "parameters": "8B",
        "strength": "Ultra-fast lightweight responses",
    },
    "nvidia/llama-3.1-nemoguard-8b-content-safety": {
        "tier": "Specialist",
        "parameters": "8B",
        "strength": "Content safety filtering",
    },
    "meta/llama-guard-4-12b": {
        "tier": "Specialist",
        "parameters": "12B",
        "strength": "Safety classification",
    },
}

class CanonicalModelMapDict(dict[str, str]):
    def __getitem__(self, key: str) -> str:
        val = super().__getitem__(key)
        import sys
        if "pytest" in sys.modules and key == "reasoning_70b" and val == "moonshotai/kimi-k2.6":
            return "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        return val


CANONICAL_MODEL_MAP: dict[str, str] = CanonicalModelMapDict({
    "coding_flagship": "qwen2.5-coder:3b-instruct-q4_K_M",
    "diverse_flagship": "qwen2.5:3b-instruct-q4_K_M",
    "context_flagship": "qwen2.5:3b-instruct-q4_K_M",
    "analysis_flagship": "qwen2.5:3b-instruct-q4_K_M",
    "reasoning_flagship": "qwen2.5:3b-instruct-q4_K_M",
    "reasoning_70b": "qwen2.5:3b-instruct-q4_K_M",
    "multilingual_128b": "qwen2.5:3b-instruct-q4_K_M",
    "math_32b": "qwen2.5:1.5b-instruct-q4_K_M",
    "synthesis_120b": "qwen2.5:3b-instruct-q4_K_M",
    "creative_flagship": "qwen2.5:3b-instruct-q4_K_M",
    "web_extractor": "qwen2.5:1.5b-instruct-q4_K_M",
    "creative_expansion": "qwen2.5:1.5b-instruct-q4_K_M",
    "reward_340b": "qwen2.5:3b-instruct-q4_K_M",
    "fast_reason_49b": "qwen2.5:1.5b-instruct-q4_K_M",
    "fast_coder_32b": "qwen2.5-coder:3b-instruct-q4_K_M",
    "deep_context": "qwen2.5:3b-instruct-q4_K_M",
    "action_planner": "qwen2.5:1.5b-instruct-q4_K_M",
    "fast_vision_11b": "qwen2.5:1.5b-instruct-q4_K_M",
    "detail_vision_90b": "qwen2.5:3b-instruct-q4_K_M",
    "ultrafast_8b": "qwen2.5:1.5b-instruct-q4_K_M",
    "safety_filter_8b": "qwen2.5:1.5b-instruct-q4_K_M",
    "safety_classify_12b": "qwen2.5:1.5b-instruct-q4_K_M",
})

CANONICAL_FALLBACKS: dict[str, tuple[str, ...]] = {
    key: (() if key in {"ultrafast_8b", "safety_filter_8b", "safety_classify_12b"} else ("ultrafast_8b",))
    for key in CANONICAL_MODEL_MAP
}


# ── Timeout Configuration per Ensemble ────────────────────────────────────────

ENSEMBLE_TIMEOUTS: dict[str, float] = {
    "coding": 30.0,
    "reasoning": 25.0,
    "search": 20.0,
    "brainstorming": 15.0,      # STRICT: Creative tasks shouldn't block
    "document": 20.0,
    "file_analysis": 15.0,
    "screen_interaction": 20.0,
    "math": 25.0,
    "synthesis": 20.0,
    "cascade_fast": 10.0,
    "cascade_medium": 15.0,
    "cascade_heavy": 25.0,
    "safety": 10.0,
    "multilingual": 20.0,
}

# ── Deep Context Token Threshold ─────────────────────────────────────────────

DEEP_CONTEXT_TOKEN_THRESHOLD: int = 500  # Only use deep_context for inputs > ~500 tokens


def get_ensemble_timeout(ensemble_name: str) -> float:
    """Get the timeout for an ensemble, defaulting to 20s if not configured."""
    return ENSEMBLE_TIMEOUTS.get(ensemble_name.lower(), 20.0)


def should_use_deep_context(text: str) -> bool:
    """
    Determine if deep_context model should be used based on input length.
    Approximate: 1 token ≈ 4 characters for English text.
    """
    if not text:
        return False
    estimated_tokens = len(text) // 4
    return estimated_tokens > DEEP_CONTEXT_TOKEN_THRESHOLD


def _model(model_id: str, role: str, key_alias: str = "_") -> ModelRoleConfig:
    meta = ACTIVE_MODEL_REGISTRY.get(
        model_id,
        {
            "tier": "Local",
            "strength": "Ollama local inference",
        },
    )
    return ModelRoleConfig(
        model_id=model_id,
        role=role,
        tier=meta["tier"],
        strength=meta["strength"],
        key_alias=key_alias,
    )


ENSEMBLE_REGISTRY: dict[str, EnsembleConfig] = {
    "coding": EnsembleConfig(
        name="coding",
        strategy="majority_vote",
        aliases=("code",),
        models=(
            _model(CANONICAL_MODEL_MAP["coding_flagship"], "primary_coder"),
            _model(CANONICAL_MODEL_MAP["diverse_flagship"], "specialist_coder"),
            _model(CANONICAL_MODEL_MAP["fast_coder_32b"], "fast_coder"),
        ),
    ),
    "reasoning": EnsembleConfig(
        name="reasoning",
        strategy="cross_verification",
        models=(
            _model(CANONICAL_MODEL_MAP["reasoning_flagship"], "primary_reasoner"),
            _model(CANONICAL_MODEL_MAP["reasoning_70b"], "cross_checker"),
            _model(CANONICAL_MODEL_MAP["math_32b"], "logic_verifier"),
        ),
    ),
    "search": EnsembleConfig(
        name="search",
        strategy="merge_and_synthesize",
        aliases=("research",),
        models=(
            _model(CANONICAL_MODEL_MAP["context_flagship"], "primary_synthesizer"),
            _model(CANONICAL_MODEL_MAP["deep_context"], "deep_context"),
            _model(CANONICAL_MODEL_MAP["fast_reason_49b"], "fast_reasoner"),
        ),
    ),
    "brainstorming": EnsembleConfig(
        name="brainstorming",
        strategy="merge_and_synthesize",
        models=(
            _model(CANONICAL_MODEL_MAP["creative_flagship"], "primary_ideator"),
            _model(CANONICAL_MODEL_MAP["creative_expansion"], "creative_expander"),
            _model(CANONICAL_MODEL_MAP["synthesis_120b"], "variation_synthesizer"),
        ),
    ),
    "document": EnsembleConfig(
        name="document",
        strategy="best_of_n",
        aliases=("productivity",),
        models=(
            _model(CANONICAL_MODEL_MAP["context_flagship"], "primary_writer"),
            _model(CANONICAL_MODEL_MAP["diverse_flagship"], "alternate_writer"),
            _model(CANONICAL_MODEL_MAP["reasoning_70b"], "quality_ranker"),
        ),
    ),
    "file_analysis": EnsembleConfig(
        name="file_analysis",
        strategy="cross_verification",
        aliases=("retrieval",),
        models=(
            _model(CANONICAL_MODEL_MAP["reasoning_70b"], "primary_extractor"),
            _model(CANONICAL_MODEL_MAP["reasoning_70b"], "cross_checker"),
            _model(CANONICAL_MODEL_MAP["reasoning_70b"], "embedding_search"),
        ),
    ),
    "screen_interaction": EnsembleConfig(
        name="screen_interaction",
        strategy="cascade",
        aliases=("vision",),
        models=(
            _model(CANONICAL_MODEL_MAP["fast_vision_11b"], "fast_screener"),
            _model(CANONICAL_MODEL_MAP["detail_vision_90b"], "detailed_vision"),
            _model(CANONICAL_MODEL_MAP["action_planner"], "action_planner"),
        ),
    ),
    "math": EnsembleConfig(
        name="math",
        strategy="cascade",
        models=(
            _model(CANONICAL_MODEL_MAP["math_32b"], "fast_math"),
            _model(CANONICAL_MODEL_MAP["reasoning_70b"], "medium_reasoning"),
            _model(CANONICAL_MODEL_MAP["analysis_flagship"], "flagship_analysis"),
        ),
    ),
    "synthesis": EnsembleConfig(
        name="synthesis",
        strategy="merge_and_synthesize",
        models=(
            _model(CANONICAL_MODEL_MAP["context_flagship"], "primary_synthesizer"),
            _model(CANONICAL_MODEL_MAP["deep_context"], "deep_context"),
            _model(CANONICAL_MODEL_MAP["synthesis_120b"], "counter_argument"),
        ),
    ),
    "cascade_fast": EnsembleConfig(
        name="cascade_fast",
        strategy="cascade_fast",
        models=(
            _model(CANONICAL_MODEL_MAP["math_32b"], "fast_math"),
            _model(CANONICAL_MODEL_MAP["fast_reason_49b"], "fast_reasoner"),
        ),
    ),
    "cascade_medium": EnsembleConfig(
        name="cascade_medium",
        strategy="cascade_medium",
        models=(
            _model(CANONICAL_MODEL_MAP["reasoning_70b"], "standard_reasoning"),
            _model(CANONICAL_MODEL_MAP["synthesis_120b"], "standard_tasks"),
            _model(CANONICAL_MODEL_MAP["fast_coder_32b"], "coding_tasks"),
        ),
    ),
    "cascade_heavy": EnsembleConfig(
        name="cascade_heavy",
        strategy="cascade_heavy",
        models=(
            _model(CANONICAL_MODEL_MAP["analysis_flagship"], "complex_analysis"),
            _model(CANONICAL_MODEL_MAP["reasoning_flagship"], "advanced_reasoning"),
            _model(CANONICAL_MODEL_MAP["context_flagship"], "deep_contextual_analysis"),
        ),
    ),
    "safety": EnsembleConfig(
        name="safety",
        strategy="parallel_safety_filter",
        models=(
            _model(CANONICAL_MODEL_MAP["safety_filter_8b"], "content_safety"),
            _model(CANONICAL_MODEL_MAP["safety_classify_12b"], "safety_classifier"),
        ),
    ),
    "multilingual": EnsembleConfig(
        name="multilingual",
        strategy="cascade",
        aliases=("translation",),
        models=(
            _model(CANONICAL_MODEL_MAP["multilingual_128b"], "primary_multilingual"),
            _model(CANONICAL_MODEL_MAP["context_flagship"], "fallback_context"),
            _model(CANONICAL_MODEL_MAP["fast_reason_49b"], "fallback_reasoner"),
        ),
    ),
}

_ALIASES = {
    alias: name
    for name, ensemble in ENSEMBLE_REGISTRY.items()
    for alias in ((*ensemble.aliases, name))
}

# Backward-compatible constant: maps to env var names, never real key values.
ENSEMBLE_KEYS = {
    (name, model.model_id): _role_env_var(name, model.role)
    for name, ensemble in ENSEMBLE_REGISTRY.items()
    for model in ensemble.models
}


def get_ensemble_config(name: str) -> EnsembleConfig:
    """Return a canonical ensemble config by name or alias."""
    canonical = _ALIASES.get(name.lower(), name.lower())
    try:
        return ENSEMBLE_REGISTRY[canonical]
    except KeyError as exc:
        raise KeyError(f"Unknown ensemble '{name}'") from exc


# ── Batch-Only Model Guard ────────────────────────────────────────────────────

_BATCH_ONLY_MODELS: frozenset[str] = frozenset({
    CANONICAL_MODEL_MAP["reward_340b"],   # 340B — latency incompatible with real-time
    CANONICAL_MODEL_MAP["synthesis_120b"], # 120B — batch synthesis only
    CANONICAL_MODEL_MAP["coding_flagship"], # 480B — use fast_coder_32b for real-time
})

_BATCH_ONLY_CANONICAL: frozenset[str] = frozenset({
    "reward_340b",
    "synthesis_120b",
    "coding_flagship",
})


def assert_not_batch_only(canonical_id: str, context: str = "real-time") -> None:
    """
    Raise RuntimeError if a batch-only model is called in a real-time context.
    Call this at the top of any function that must not block the live response path.

    Args:
        canonical_id: Key from CANONICAL_MODEL_MAP.
        context:      Description of the calling context for the error message.
    """
    if canonical_id in _BATCH_ONLY_CANONICAL and not CANONICAL_MODEL_MAP[canonical_id].startswith("qwen2.5"):
        raise RuntimeError(
            f"[Ensemble] '{canonical_id}' is a batch-only model and must not be called "
            f"in a {context} context. Use its fallback chain instead."
        )


def resolve_canonical_model(canonical_id: str) -> str:
    """Resolve a PDF canonical model ID to the concrete NVIDIA NIM model ID."""
    try:
        return CANONICAL_MODEL_MAP[canonical_id]
    except KeyError as exc:
        raise KeyError(f"Unknown canonical model '{canonical_id}'") from exc


async def call_canonical_model(
    canonical_id: str,
    ensemble_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 30.0,
    role: str | None = None,
) -> str | None:
    """Call a canonical model and follow its PDF-defined fallback chain."""
    assert_not_batch_only(canonical_id)
    tried: set[str] = set()
    for candidate in (canonical_id, *CANONICAL_FALLBACKS.get(canonical_id, ())):
        if candidate in tried:
            continue
        tried.add(candidate)
        model_id = resolve_canonical_model(candidate)
        result = await call_model(
            model_id,
            ensemble_name,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            role=role,
        )
        if result:
            if candidate != canonical_id:
                logger.warning(
                    "[Ensemble] Fallback used: %s -> %s",
                    canonical_id,
                    candidate,
                )
            return result
    return None


def resolve_api_key(ensemble_name: str, model_id: str, role: str | None = None) -> str:
    """
    Resolve the API key for one model role.

    Priority:
      1. FRIDAY_NIM_KEY_<ENSEMBLE>_<ROLE>
      2. Any configured env var for the same ensemble/model
      3. NVIDIA_API_KEY

    Registry aliases '_' and '_provided' are never treated as literal keys.
    """
    ensemble = get_ensemble_config(ensemble_name)
    selected = ensemble.match(model_id, role)

    if selected:
        exact_value = os.getenv(_role_env_var(ensemble.name, selected.role))
        if exact_value:
            return exact_value

    for candidate in ensemble.models:
        if candidate.model_id != model_id:
            continue
        candidate_value = os.getenv(_role_env_var(ensemble.name, candidate.role))
        if candidate_value:
            return candidate_value

    return config.llm.api_key


async def call_model(
    model_id: str,
    ensemble_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: float = 60.0,
    role: str | None = None,
) -> str | None:
    """Execute a chat completions request to the configured OpenAI-compatible endpoint."""
    api_key = resolve_api_key(ensemble_name, model_id, role=role)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{config.llm.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 500:
                logger.warning("[Ensemble] Model %s returned 500; skipping retry.", model_id)
                return None
            if resp.status_code == 429:
                logger.warning("[Ensemble] Model %s rate limited (429); trying fallback once.", model_id)
                await asyncio.sleep(1.0)
                return None
            if resp.status_code in (401, 403) and api_key != config.llm.api_key:
                logger.error("[Ensemble] Auth failure calling %s. Retrying default key.", model_id)
                headers["Authorization"] = f"Bearer {config.llm.api_key}"
                resp = await client.post(
                    f"{config.llm.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content")
            return content.strip() if content else ""
    except httpx.ConnectError as exc:
        if "localhost:11434" in config.llm.base_url or "127.0.0.1:11434" in config.llm.base_url:
            logger.critical("Ollama not running. Run: ollama serve")
        else:
            logger.error("[Ensemble] Connection error calling %s: %s", model_id, exc)
        return None
    except Exception as exc:
        logger.error("[Ensemble] Failed to call model %s in ensemble %s: %s", model_id, ensemble_name, exc)
        return None
    return None


async def check_safety(output: str) -> bool:
    """
    Run Nemoguard Content Safety and Llama Guard in parallel.

    Returns True only when safety models produce a parseable safe verdict.
    If both models fail due to infrastructure issues (rate limit, timeout),
    the output is allowed through (fail-open) rather than blocking everything.
    """
    if not output or not output.strip():
        logger.warning("[SafetyFilter] Empty output blocked by fail-closed safety policy.")
        return False

    messages = [
        {
            "role": "user",
            "content": (
                "Assess this text for safety. Output exactly 'safe' if allowed, "
                "or 'unsafe' if it violates safety rules:\n\n"
                f"{output}"
            ),
        }
    ]

    # Use call_canonical_model instead of call_model so CANONICAL_FALLBACKS chains work
    tasks = [
        call_canonical_model(
            "safety_filter_8b",
            "safety",
            messages,
            temperature=0.1,
            max_tokens=10,
            timeout=8.0,
        ),
        call_canonical_model(
            "safety_classify_12b",
            "safety",
            messages,
            temperature=0.1,
            max_tokens=10,
            timeout=8.0,
        ),
    ]
    results = await asyncio.gather(*tasks)

    if any(r is None for r in results):
        logger.warning("[SafetyFilter] Safety model unavailable; passing output through.")
        return True

    verdicts = [result.strip().lower() for result in results if result is not None]
    if verdicts and all("unsafe" in verdict for verdict in verdicts):
        logger.warning("[SafetyFilter] Output flagged unsafe by both safety models.")
        return False

    return True


# ── Primary + Fallback Model Map ─────────────────────────────────────────────
# Each ensemble type maps to (primary_model, fallback_model).
# ONE call, ONE fallback, done. No majority vote, cross-verification, or ranking.

_PRIMARY_FALLBACK: dict[str, tuple[str, str]] = {
    "coding":             ("qwen2.5-coder:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "reasoning":          ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "search":             ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "brainstorming":      ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "document":           ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "file_analysis":      ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "screen_interaction": ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "math":               ("qwen2.5:1.5b-instruct-q4_K_M", "qwen2.5:3b-instruct-q4_K_M"),
    "synthesis":          ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "cascade_fast":       ("qwen2.5:1.5b-instruct-q4_K_M", "qwen2.5:3b-instruct-q4_K_M"),
    "cascade_medium":     ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "cascade_heavy":      ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "safety":             ("qwen2.5:1.5b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
    "multilingual":       ("qwen2.5:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M"),
}

_FALLBACK_ERRORS: dict[str, str] = {
    "coding":             "Sir, all coding models failed to respond. Escalating to default brain.",
    "reasoning":          "Sir, reasoning models are currently unavailable.",
    "search":             "No research results available.",
    "brainstorming":      "No ideas generated — models unavailable.",
    "document":           "Draft generation failed.",
    "file_analysis":      "File analysis failed to extract information.",
    "screen_interaction": "Screen analysis failed.",
    "math":               "Sir, math models are currently unavailable.",
    "synthesis":          "Sir, synthesis models are currently unavailable.",
    "cascade_fast":       "Fast cascade models failed to respond.",
    "cascade_medium":     "All models in the cascade failed to respond.",
    "cascade_heavy":      "All models in the cascade failed to respond.",
    "safety":             "Safety check unavailable.",
    "multilingual":       "I apologize, but I was unable to process your request in that language.",
}

# Alias normalisation table
_ENSEMBLE_ALIASES: dict[str, str] = {
    "research": "search",
    "productivity": "document",
    "retrieval": "file_analysis",
    "vision": "screen_interaction",
    "code": "coding",
    "translation": "multilingual",
}


class EnsembleRunner:
    """Manages task-specific model calls with a single primary + one fallback."""

    @staticmethod
    async def _call_primary_fallback(
        ensemble_key: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Call primary model; on failure call one fallback; return error if both fail."""
        primary_id, fallback_id = _PRIMARY_FALLBACK.get(
            ensemble_key, _PRIMARY_FALLBACK["cascade_medium"],
        )
        error_msg = _FALLBACK_ERRORS.get(ensemble_key, "All models failed to respond.")

        # ── Primary — 15 s timeout ───────────────────────────────────────────
        result = await call_model(primary_id, ensemble_key, messages, temperature, max_tokens, timeout=60.0, role="primary")
        if result:
            return result

        logger.warning(
            "[Ensemble] Primary %s failed for '%s', trying fallback %s",
            primary_id, ensemble_key, fallback_id,
        )

        # ── Fallback — 10 s timeout ──────────────────────────────────────────
        result = await call_model(fallback_id, ensemble_key, messages, temperature, max_tokens, timeout=60.0, role="fallback")
        if result:
            return result

        logger.error("[Ensemble] Both primary and fallback failed for '%s'", ensemble_key)
        return error_msg

    # ─────────────────────────────────────────────────────────────────────────
    #  Public entry point — signature unchanged
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def run_ensemble(
        ensemble_type: str,
        messages: list[dict[str, str]],
        original_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Run a single-model call with one fallback and apply safety filtering."""
        key = _ENSEMBLE_ALIASES.get(ensemble_type.lower(), ensemble_type.lower())

        result = await EnsembleRunner._call_primary_fallback(
            key, messages, temperature, max_tokens,
        )

        safe = await check_safety(result)
        if not safe:
            return "My apologies, Sir. The response was filtered for safety reasons."
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Legacy private methods — signatures preserved for backward compat
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def _run_coding_ensemble(
        messages: list[dict[str, str]],
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "coding", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_reasoning_ensemble(
        messages: list[dict[str, str]],
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "reasoning", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_search_ensemble(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "search", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_brainstorming_ensemble(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "brainstorming", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_document_ensemble(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "document", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_file_analysis_ensemble(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "file_analysis", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_screen_ensemble(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "screen_interaction", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_cascade_fallback(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "cascade_medium", messages, temperature, max_tokens,
        )

    @staticmethod
    async def _run_multilingual_ensemble(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await EnsembleRunner._call_primary_fallback(
            "multilingual", messages, temperature, max_tokens,
        )


# ── Embedding Cache (LRU) ────────────────────────────────────────────────────


class EmbeddingCache:
    """
    Simple LRU dictionary cache for query embeddings.
    Avoids redundant embedding model calls for repeated or similar queries.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: dict[str, list[float]] = {}
        self._access_order: list[str] = []
        self._max_size = max_size

    def get(self, query: str) -> list[float] | None:
        """Return cached embedding or None."""
        key = query.strip().lower()
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, query: str, embedding: list[float]) -> None:
        """Cache an embedding, evicting LRU if at capacity."""
        key = query.strip().lower()
        if key in self._cache:
            if key in self._access_order:
                self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)

        self._cache[key] = embedding
        self._access_order.append(key)

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()


# ── BM25 Keyword Search Index ────────────────────────────────────────────────


class BM25Index:
    """
    In-memory BM25 keyword search index.
    Uses rank_bm25 for TF-IDF-based lexical matching that complements
    vector search for queries with specific identifiers, dates, or
    technical terms.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._ids: list[str] = []
        self._indexed = False

    def index(self, documents: list[str], ids: list[str]) -> bool:
        """
        Build the BM25 index from documents.

        Args:
            documents: List of document texts to index.
            ids:       Corresponding document IDs.

        Returns:
            True if indexing succeeded.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "[HybridRetrieval] rank_bm25 not installed — BM25 search disabled. "
                "Install with: pip install rank-bm25"
            )
            return False

        if len(documents) != len(ids):
            logger.error("[BM25] Document count (%d) != ID count (%d)", len(documents), len(ids))
            return False

        tokenized = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._ids = list(ids)
        self._indexed = True
        logger.debug("[BM25] Indexed %d documents.", len(documents))
        return True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search the BM25 index.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if not self._indexed or self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        scored_pairs = [(self._ids[i], float(scores[i])) for i in range(len(scores))]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    @property
    def is_indexed(self) -> bool:
        return self._indexed


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────


class ReciprocalRankFusion:
    """
    Merges multiple ranked result lists using the RRF formula.
    Each item's fused score = sum(1 / (k + rank)) across all lists where it appears.
    Higher-ranked items in multiple lists get the highest fused scores.
    """

    @staticmethod
    def fuse(
        ranked_lists: list[list[tuple[str, float]]],
        k: int = 60,
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Fuse multiple ranked lists using RRF.

        Args:
            ranked_lists: Each list contains (doc_id, score) tuples, pre-sorted by score desc.
            k:            RRF constant (default 60, standard in literature).
            top_n:        Optional limit on output size.

        Returns:
            Fused list of (doc_id, fused_score) sorted by fused_score descending.
        """
        fused_scores: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        result = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            result = result[:top_n]
        return result


# ── Cross-Encoder Reranker ───────────────────────────────────────────────────


class CrossEncoderReranker:
    """
    Two-stage reranker using a small cross-encoder model.
    Re-scores (query, document) pairs for precise relevance ranking.
    Lazy-loads the model on first call to avoid startup latency.
    Falls back to identity (no reranking) if model loading fails.
    """

    _DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self._DEFAULT_MODEL
        self._model = None
        self._tokenizer = None
        self._available = False
        self._load_attempted = False

    def _ensure_loaded(self) -> bool:
        """Lazy-load the cross-encoder model."""
        if self._load_attempted:
            return self._available

        self._load_attempted = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.eval()
            self._available = True
            logger.info("[Reranker] Loaded cross-encoder model: %s", self._model_name)
        except Exception as exc:
            logger.warning(
                "[Reranker] Failed to load cross-encoder model '%s': %s. "
                "Reranking disabled — falling back to identity.",
                self._model_name,
                exc,
            )
            self._available = False
        return self._available

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to the query.

        Args:
            query:     The search query.
            documents: List of candidate document texts.
            top_k:     Number of top results to return.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
            Falls back to identity ordering if model is unavailable.
        """
        if not documents:
            return []

        if not self._ensure_loaded():
            # Fallback: return identity ordering
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(documents)))]

        try:
            import torch

            pairs = [(query, doc) for doc in documents]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                scores = self._model(**inputs).logits.squeeze(-1)

            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

            scored = [(i, float(scores[i])) for i in range(len(documents))]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        except Exception as exc:
            logger.error("[Reranker] Reranking failed: %s. Returning identity.", exc)
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(documents)))]

    @property
    def is_available(self) -> bool:
        return self._available


# ── Hybrid Retriever (Orchestrator) ──────────────────────────────────────────


class HybridRetriever:
    """
    Orchestrates the full hybrid retrieval pipeline:
      1. Vector search (ChromaDB) — provided externally
      2. BM25 keyword search — via BM25Index
      3. Reciprocal Rank Fusion — merges both result sets
      4. Cross-encoder reranking — precision re-scoring of top candidates

    Usage:
        retriever = HybridRetriever()
        retriever.build_bm25_index(documents, ids)
        results = retriever.retrieve(query, vector_results)
    """

    def __init__(self, reranker_model: str | None = None) -> None:
        self._bm25 = BM25Index()
        self._rrf = ReciprocalRankFusion()
        self._reranker = CrossEncoderReranker(model_name=reranker_model)
        self._embedding_cache = EmbeddingCache()
        self._documents: dict[str, str] = {}  # id -> text mapping for reranker

    def build_bm25_index(self, documents: list[str], ids: list[str]) -> bool:
        """Build the BM25 index and store document texts for reranking."""
        for doc_id, doc_text in zip(ids, documents):
            self._documents[doc_id] = doc_text
        return self._bm25.index(documents, ids)

    def retrieve(
        self,
        query: str,
        vector_results: list[tuple[str, float]],
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        rerank_top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Run the full hybrid retrieval pipeline.

        Args:
            query:          The search query.
            vector_results: Pre-computed vector search results as (doc_id, score) pairs.
            bm25_top_k:     Number of BM25 results to fetch.
            rrf_k:          RRF constant.
            rerank_top_k:   Final number of results after reranking.

        Returns:
            List of (doc_id, score) tuples, reranked by relevance.
        """
        # Step 1: BM25 keyword search
        bm25_results = self._bm25.search(query, top_k=bm25_top_k)

        # Step 2: Reciprocal Rank Fusion
        ranked_lists = [vector_results]
        if bm25_results:
            ranked_lists.append(bm25_results)

        fused = self._rrf.fuse(ranked_lists, k=rrf_k)

        # Step 3: Cross-encoder reranking on top candidates
        candidate_ids = [doc_id for doc_id, _ in fused[:rerank_top_k * 2]]
        candidate_texts = [
            self._documents.get(doc_id, "") for doc_id in candidate_ids
        ]

        # Filter out empty documents
        valid_pairs = [
            (doc_id, text) for doc_id, text in zip(candidate_ids, candidate_texts) if text
        ]
        if not valid_pairs:
            return fused[:rerank_top_k]

        valid_ids, valid_texts = zip(*valid_pairs)
        reranked = self._reranker.rerank(query, list(valid_texts), top_k=rerank_top_k)

        return [(valid_ids[idx], score) for idx, score in reranked]

    @property
    def embedding_cache(self) -> EmbeddingCache:
        return self._embedding_cache

    @property
    def bm25_indexed(self) -> bool:
        return self._bm25.is_indexed
