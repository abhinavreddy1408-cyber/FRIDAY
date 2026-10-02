# mypy: ignore-errors
"""
Unit and integration tests for the new components introduced in the architectural overhaul:
1. phase1.py — ConfidenceGate and IntentClassifier
2. ensemble.py — BM25Index, ReciprocalRankFusion, CrossEncoderReranker, HybridRetriever
3. memory/manager.py — WorkingMemory, EpisodicMemory, SemanticFactStore, MemoryHygiene
4. safety.py — GroundingVerifier, ToolOutputValidator
"""

import math
import time
import sqlite3
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from friday.phase1 import (
    ConfidenceGate,
    IntentClassifier,
    IntentCategory,
    Phase1Gate,
    RouteDecision,
    TriageResult,
)
from friday.ensemble import (
    BM25Index,
    ReciprocalRankFusion,
    CrossEncoderReranker,
    HybridRetriever,
    EmbeddingCache,
)
from friday.memory.manager import (
    WorkingMemory,
    EpisodicMemory,
    SemanticFactStore,
    StructuredFact,
    MemoryHygiene,
    MemoryManager,
    MemoryEntry,
    MemoryCategory,
    MemoryTier,
)
from friday.safety import (
    GroundingVerifier,
    ToolOutputValidator,
    GroundingReport,
    ValidationResult,
    VerificationReport,
    VerificationChain,
    SafetyCheck,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. PHASE 1 TESTS: ConfidenceGate and IntentClassifier
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase1Overhaul:
    def test_confidence_gate_empty(self):
        gated, resp, conf = ConfidenceGate.gate("")
        assert gated is False
        assert resp is None
        
        gated, resp, conf = ConfidenceGate.gate("    ")
        assert gated is False
        assert resp is None

    def test_confidence_gate_greetings(self):
        greetings = ["hello", "hi", "hey friday", "good morning", "yo", "howdy"]
        for g in greetings:
            gated, resp, conf = ConfidenceGate.gate(g)
            assert gated is True
            assert resp == "Hello! How can I assist you today, Sir?"
            assert conf == "high"

    def test_confidence_gate_acknowledgements(self):
        acks = ["ok", "okay", "got it", "thanks", "thank you", "sure", "alright", "cool", "nice", "great", "yep", "yes", "no", "nope"]
        for a in acks:
            gated, resp, conf = ConfidenceGate.gate(a)
            assert gated is True
            assert resp == "Understood, Sir. Standing by."
            assert conf == "high"

    def test_confidence_gate_status(self):
        statuses = ["are you ready", "are you online", "what's your status", "how fast are you"]
        for s in statuses:
            gated, resp, conf = ConfidenceGate.gate(s)
            assert gated is True
            assert resp == "All systems operational, Sir. Ready for your command."
            assert conf == "high"

    def test_confidence_gate_time_date(self):
        queries = ["what's the time today", "what is the date", "what is the day now"]
        for q in queries:
            gated, resp, conf = ConfidenceGate.gate(q)
            assert gated is True
            assert "currently" in resp
            assert conf == "high"

    def test_confidence_gate_simple_math(self):
        queries = [
            ("what is 2 + 2", "The answer is 4."),
            ("what's 5 * 10", "The answer is 50."),
            ("what is 20 / 4", "The answer is 5."),
            ("what is 2^3", "The answer is 8."),
            ("what's 3.5 + 1.5", "The answer is 5.")
        ]
        for q, expected in queries:
            gated, resp, conf = ConfidenceGate.gate(q)
            assert gated is True
            assert resp == expected
            assert conf == "high"
            
        # Error / division by zero should be bypassed gracefully
        gated, resp, conf = ConfidenceGate.gate("what is 2 / 0")
        assert gated is False
        assert resp is None

    def test_intent_classifier(self):
        # Empty
        assert IntentClassifier.classify("") == IntentCategory.CONVERSATIONAL
        
        # Retrieval
        retrievals = [
            "remember my name is Bob",
            "what did we discuss earlier?",
            "recall our previous conversation",
            "continue our discussion"
        ]
        for r in retrievals:
            assert IntentClassifier.classify(r) == IntentCategory.RETRIEVAL
            
        # Clarification
        clarifications = [
            "what do you mean by that?",
            "can you elaborate?",
            "explain this more"
        ]
        for c in clarifications:
            assert IntentClassifier.classify(c) == IntentCategory.CLARIFICATION
            
        # Task Execution
        tasks = [
            "create a new file",
            "deploy the app to production",
            "run python main.py",
            "delete this folder"
        ]
        for t in tasks:
            assert IntentClassifier.classify(t) == IntentCategory.TASK_EXECUTION
            
        # Procedural
        procedurals = [
            "how to learn python?",
            "steps to learn python",
            "guide for learning python"
        ]
        for p in procedurals:
            assert IntentClassifier.classify(p) == IntentCategory.PROCEDURAL
            
        # Default / Factual
        assert IntentClassifier.classify("Who is the president of France?") == IntentCategory.FACTUAL

    @pytest.mark.asyncio
    async def test_phase1_gate_route_integration(self):
        # Test routing a gated greeting
        with patch("friday.phase1.Phase1Gate.check_output_safety", AsyncMock(return_value=True)):
            triage = await Phase1Gate.route("hello")
            assert triage.route == RouteDecision.FAST_PATH
            assert triage.fast_response == "Hello! How can I assist you today, Sir?"
            assert triage.intent_category == IntentCategory.FACTUAL
            
        # Test routing a gated greeting that fails output safety
        with patch("friday.phase1.Phase1Gate.check_output_safety", AsyncMock(return_value=False)):
            triage = await Phase1Gate.route("hello")
            assert triage.route == RouteDecision.BLOCKED
            assert triage.safe is False
            assert "failed output safety" in triage.block_reason


# ──────────────────────────────────────────────────────────────────────────────
# 2. ENSEMBLE TESTS: BM25Index, ReciprocalRankFusion, CrossEncoderReranker, HybridRetriever
# ──────────────────────────────────────────────────────────────────────────────

class TestEnsembleOverhaul:
    def test_bm25_index_basic(self):
        idx = BM25Index()
        assert idx.is_indexed is False
        
        # Search when not indexed
        assert idx.search("hello") == []
        
        # Mismatched lengths
        assert idx.index(["doc1"], ["id1", "id2"]) is False
        
        # Index successfully
        docs = ["the quick brown fox", "jumps over the lazy dog", "the quick dog"]
        ids = ["doc1", "doc2", "doc3"]
        assert idx.index(docs, ids) is True
        assert idx.is_indexed is True
        
        # Search match (using "fox" to avoid negative IDF issue in small corpus)
        results = idx.search("fox", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"
        assert results[0][1] > 0.0

    def test_rrf_fusion(self):
        # Empty
        assert ReciprocalRankFusion.fuse([]) == []
        
        # Normal fuse
        list1 = [("doc1", 0.9), ("doc2", 0.8)]
        list2 = [("doc2", 0.95), ("doc3", 0.7)]
        # RRF formula: 1 / (60 + rank)
        fused = ReciprocalRankFusion.fuse([list1, list2], k=60, top_n=2)
        assert len(fused) == 2
        assert fused[0][0] == "doc2"  # Highest fused score because it appears in both
        assert fused[1][0] == "doc1"

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_cross_encoder_reranker_success(self, mock_model_load, mock_tokenizer_load):
        # Mock transformers
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_tokenizer_load.return_value = mock_tokenizer
        mock_model_load.return_value = mock_model
        
        mock_tokenizer.return_value = {"input_ids": "dummy"}
        
        # Mock model return value: torch tensor for logits
        import torch
        mock_logits = torch.tensor([1.5, 0.5])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        
        reranker = CrossEncoderReranker()
        assert reranker.is_available is False
        
        # First call loads the model
        scored = reranker.rerank("query", ["doc1", "doc2"], top_k=2)
        assert reranker.is_available is True
        assert len(scored) == 2
        assert scored[0] == (0, 1.5)
        assert scored[1] == (1, 0.5)

    def test_cross_encoder_reranker_fallback_load_failure(self):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=Exception("Failed to load")):
            reranker = CrossEncoderReranker()
            scored = reranker.rerank("query", ["doc1", "doc2"], top_k=2)
            assert reranker.is_available is False
            assert scored == [(0, 1.0), (1, 0.99)]  # Identity mapping fallback

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_cross_encoder_reranker_fallback_runtime_exception(self, mock_model_load, mock_tokenizer_load):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        mock_model_load.return_value = mock_model
        
        mock_tokenizer.side_effect = Exception("Runtime error")
        
        reranker = CrossEncoderReranker()
        scored = reranker.rerank("query", ["doc1", "doc2"], top_k=2)
        assert scored == [(0, 1.0), (1, 0.99)]  # Identity mapping fallback

    def test_cross_encoder_reranker_empty(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", []) == []

    @patch("friday.ensemble.CrossEncoderReranker.rerank")
    def test_hybrid_retriever(self, mock_rerank):
        mock_rerank.return_value = [(0, 0.95), (1, 0.8)]
        
        retriever = HybridRetriever()
        assert retriever.bm25_indexed is False
        
        # Build index
        docs = ["doc one text", "doc two text"]
        ids = ["id1", "id2"]
        assert retriever.build_bm25_index(docs, ids) is True
        assert retriever.bm25_indexed is True
        
        # Retrieve
        vector_results = [("id2", 0.9), ("id1", 0.5)]
        results = retriever.retrieve("one", vector_results)
        assert len(results) == 2
        assert results[0][0] in ["id1", "id2"]
        assert isinstance(retriever.embedding_cache, EmbeddingCache)

    def test_hybrid_retriever_empty_valid_pairs(self):
        retriever = HybridRetriever()
        # No documents indexed, so _documents is empty
        vector_results = [("id2", 0.9), ("id1", 0.5)]
        results = retriever.retrieve("one", vector_results, rerank_top_k=2)
        assert len(results) == 2
        assert results[0][0] == "id2"
        assert results[1][0] == "id1"


# ──────────────────────────────────────────────────────────────────────────────
# 3. TRIPARTITE MEMORY TESTS: WorkingMemory, EpisodicMemory, SemanticFactStore, MemoryHygiene
# ──────────────────────────────────────────────────────────────────────────────

class TestTripartiteMemory:
    def test_working_memory(self):
        wm = WorkingMemory()
        assert wm.has_state is False
        
        wm.set_goal("test goal")
        assert wm.get_goal() == "test goal"
        assert wm.has_state is True
        
        wm.set_plan(["step 1", "step 2"])
        assert wm.get_plan() == ["step 1", "step 2"]
        
        wm.add_observation("obs 1")
        wm.add_observation("obs 2")
        assert wm.get_observations() == ["obs 1", "obs 2"]
        
        wm.set_scratch("key", "val")
        assert wm.get_scratch("key") == "val"
        assert wm.get_scratch("missing", "default") == "default"
        
        # Compress without summarizer
        compressed = wm.compress()
        assert compressed == "obs 1\nobs 2"
        assert wm.get_observations() == []
        
        # Compress with custom summarizer
        wm.add_observation("obs 3")
        compressed_custom = wm.compress(lambda x: x.upper())
        assert compressed_custom == "OBS 3"
        
        # Clear
        wm.set_goal("goal")
        wm.clear()
        assert wm.has_state is False

    def test_episodic_memory(self):
        em = EpisodicMemory()
        assert em.is_available is False
        
        # Setup with unavailable semantic memory
        mock_semantic = MagicMock()
        mock_semantic._available = False
        assert em.setup(mock_semantic) is False
        
        # Link turns when unavailable
        assert em.store_turn("user", "assistant") is False
        assert em.recall_recent() == []
        assert em.recall_similar("query") == []
        
        # Setup with available semantic memory
        mock_semantic._available = True
        assert em.setup(mock_semantic) is True
        assert em.is_available is True
        
        # Store turn
        mock_semantic.store.return_value = True
        assert em.store_turn("hello", "hi", session_id="sess_1", turn_index=2) is True
        mock_semantic.store.assert_called_once()
        entry = mock_semantic.store.call_args[0][0]
        assert entry.content == "User: hello\nAssistant: hi"
        assert entry.metadata["session_id"] == "sess_1"
        assert entry.metadata["turn_index"] == "2"
        
        # Recall recent
        entry1 = MemoryEntry.create("User: a\nAssistant: b", MemoryCategory.CONVERSATION, MemoryTier.SEMANTIC, {"session_id": "sess_1"})
        entry2 = MemoryEntry.create("User: c\nAssistant: d", MemoryCategory.CONVERSATION, MemoryTier.SEMANTIC, {"session_id": "sess_2"})
        mock_semantic.search.return_value = [(entry1, 0.1), (entry2, 0.2)]
        
        recent = em.recall_recent("sess_1", limit=1)
        assert len(recent) == 1
        assert recent[0].metadata["session_id"] == "sess_1"
        
        # Recall similar
        similar = em.recall_similar("hello", limit=2)
        assert len(similar) == 2

    @pytest.fixture
    def fact_store(self, tmp_path):
        db_file = tmp_path / "test_memory.db"
        from friday.config import config
        original_db = config.storage.memory_db
        object.__setattr__(config.storage, "memory_db", str(db_file))
        try:
            store = SemanticFactStore()
            store.setup()
            yield store
        finally:
            object.__setattr__(config.storage, "memory_db", original_db)

    def test_semantic_fact_store(self, fact_store):
        assert fact_store.count() == 0
        
        # Store initial fact
        assert fact_store.store_fact("sky", "is", "blue", confidence=0.8, source="user") is True
        assert fact_store.count() == 1
        
        # Try storing fact with lower confidence -> should fail/skip
        assert fact_store.store_fact("sky", "is", "green", confidence=0.5, source="user") is False
        facts = fact_store.query_facts("sky", "is")
        assert len(facts) == 1
        assert facts[0].value == "blue"
        assert facts[0].confidence == 0.8
        
        # Store fact with higher/equal confidence -> should update
        assert fact_store.store_fact("sky", "is", "dark blue", confidence=0.9, source="system") is True
        facts = fact_store.query_facts("sky", "is")
        assert len(facts) == 1
        assert facts[0].value == "dark blue"
        assert facts[0].confidence == 0.9
        assert facts[0].source == "system"
        
        # Update fact
        assert fact_store.update_fact("sky", "is", "black", new_confidence=0.95) is True
        facts = fact_store.query_facts("sky", "is")
        assert len(facts) == 1
        assert facts[0].value == "black"
        assert facts[0].confidence == 0.95
        
        # Update fact without new confidence
        assert fact_store.update_fact("sky", "is", "gray") is True
        facts = fact_store.query_facts("sky", "is")
        assert len(facts) == 1
        assert facts[0].value == "gray"
        assert facts[0].confidence == 0.95  # Retained
        
        # Query with only subject, only predicate, both, or none
        facts_sub = fact_store.query_facts(subject="sky")
        assert len(facts_sub) == 1
        
        facts_pred = fact_store.query_facts(predicate="is")
        assert len(facts_pred) == 1
        
        facts_none = fact_store.query_facts()
        assert len(facts_none) == 1

    def test_semantic_fact_store_setup_failure(self):
        with patch("sqlite3.connect", side_effect=sqlite3.Error("Mock DB Error")):
            store = SemanticFactStore()
            assert store.setup() is False
            assert store.count() == 0

    def test_semantic_fact_store_operations_failure(self, fact_store):
        with patch.object(fact_store, "_connection", side_effect=sqlite3.Error("Mock error")):
            assert fact_store.store_fact("a", "b", "c") is False
            assert fact_store.query_facts() == []
            assert fact_store.update_fact("a", "b", "c") is False

    def test_memory_hygiene_calculate_vitality(self):
        now = time.time()
        entry_fresh = MemoryEntry.create(
            "test content",
            MemoryCategory.GOAL,
            MemoryTier.LONG_TERM,
            metadata={"access_count": "5"}
        )
        v_fresh = MemoryHygiene.calculate_vitality(entry_fresh)
        assert 0.0 <= v_fresh <= 1.0
        
        entry_old = MemoryEntry.create(
            "test content",
            MemoryCategory.CONVERSATION,
            MemoryTier.LONG_TERM,
            metadata={"access_count": "1"}
        )
        object.__setattr__(entry_old, "timestamp", now - 30 * 86400.0)
        v_old = MemoryHygiene.calculate_vitality(entry_old)
        assert v_old < v_fresh

    def test_memory_hygiene_prune_low_vitality(self):
        long_term = MagicMock()
        semantic = MagicMock()
        semantic._available = True
        
        entry_high = MemoryEntry.create("High vitality", MemoryCategory.GOAL, MemoryTier.LONG_TERM)
        entry_low = MemoryEntry.create("Low vitality", MemoryCategory.CONVERSATION, MemoryTier.LONG_TERM)
        object.__setattr__(entry_low, "timestamp", time.time() - 100 * 86400.0)
        
        long_term.recall.return_value = [entry_high, entry_low]
        
        mock_conn = MagicMock()
        long_term._connection.return_value.__enter__.return_value = mock_conn
        
        hygiene = MemoryHygiene(long_term, semantic)
        pruned_count = hygiene.prune_low_vitality(threshold=0.3)
        
        assert pruned_count == 1
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args[0]
        assert "DELETE FROM memories WHERE id IN" in args[0]
        assert entry_low.id in args[1]
        semantic._collection.delete.assert_called_once_with(ids=[entry_low.id])

    def test_memory_hygiene_prune_low_vitality_db_failure(self):
        long_term = MagicMock()
        semantic = MagicMock()
        
        entry_low = MemoryEntry.create("Low vitality", MemoryCategory.CONVERSATION, MemoryTier.LONG_TERM)
        object.__setattr__(entry_low, "timestamp", time.time() - 100 * 86400.0)
        long_term.recall.return_value = [entry_low]
        
        long_term._connection.side_effect = sqlite3.Error("DB error")
        
        hygiene = MemoryHygiene(long_term, semantic)
        assert hygiene.prune_low_vitality(threshold=0.3) == 0

    def test_memory_hygiene_consolidate_related(self):
        long_term = MagicMock()
        semantic = MagicMock()
        semantic._available = True
        
        entry_high = MemoryEntry.create("Duplicate text", MemoryCategory.GOAL, MemoryTier.LONG_TERM)
        entry_low = MemoryEntry.create("Duplicate text", MemoryCategory.CONVERSATION, MemoryTier.LONG_TERM)
        object.__setattr__(entry_low, "timestamp", time.time() - 100 * 86400.0)
        
        long_term.recall.return_value = [entry_high, entry_low]
        
        def mock_search(query, limit):
            if query == entry_high.content:
                return [(entry_low, 0.01)]
            elif query == entry_low.content:
                return [(entry_high, 0.01)]
            return []
            
        semantic.search.side_effect = mock_search
        
        mock_conn = MagicMock()
        long_term._connection.return_value.__enter__.return_value = mock_conn
        
        hygiene = MemoryHygiene(long_term, semantic)
        consolidated_count = hygiene.consolidate_related(similarity_threshold=0.95)
        
        assert consolidated_count == 1
        mock_conn.execute.assert_called_once_with("DELETE FROM memories WHERE id = ?", (entry_low.id,))
        semantic._collection.delete.assert_called_once_with(ids=[entry_low.id])

    def test_memory_hygiene_consolidate_related_db_failure(self):
        long_term = MagicMock()
        semantic = MagicMock()
        semantic._available = True
        
        entry_high = MemoryEntry.create("Duplicate text", MemoryCategory.GOAL, MemoryTier.LONG_TERM)
        entry_low = MemoryEntry.create("Duplicate text", MemoryCategory.CONVERSATION, MemoryTier.LONG_TERM)
        object.__setattr__(entry_low, "timestamp", time.time() - 100 * 86400.0)
        
        long_term.recall.return_value = [entry_high, entry_low]
        semantic.search.return_value = [(entry_low, 0.01)]
        
        long_term._connection.side_effect = sqlite3.Error("DB error")
        
        hygiene = MemoryHygiene(long_term, semantic)
        assert hygiene.consolidate_related(similarity_threshold=0.95) == 0

    def test_memory_hygiene_consolidate_semantic_unavailable(self):
        long_term = MagicMock()
        semantic = MagicMock()
        semantic._available = False
        hygiene = MemoryHygiene(long_term, semantic)
        assert hygiene.consolidate_related() == 0

    def test_memory_manager_integration(self):
        mgr = MemoryManager()
        assert mgr.is_ready is False
        
        assert mgr.setup() is True
        assert mgr.is_ready is True
        
        assert isinstance(mgr.working, WorkingMemory)
        assert isinstance(mgr.episodic, EpisodicMemory)
        assert isinstance(mgr.facts, SemanticFactStore)
        assert isinstance(mgr.hygiene, MemoryHygiene)
        
        stats = mgr.stats
        assert "working_memory_active" in stats
        assert "episodic_available" in stats
        assert "facts_count" in stats


# ──────────────────────────────────────────────────────────────────────────────
# 4. SAFETY & VERIFICATION TESTS: GroundingVerifier, ToolOutputValidator
# ──────────────────────────────────────────────────────────────────────────────

class TestSafetyVerificationOverhaul:
    @pytest.mark.asyncio
    @patch("friday.safety.call_model")
    async def test_grounding_verifier_grounded(self, mock_call):
        # Setup reasoning output format without digital source labels at the very end
        mock_call.return_value = (
            "GROUNDED\n"
            "- CITED: The sky is blue\n"
            "Confidence: 0.95"
        )
        
        report = await GroundingVerifier.verify_against_sources(
            response="The sky is blue.",
            sources=["The sky is blue."],
        )
        
        assert report is not None
        assert report.grounded is True
        assert report.confidence == 0.95
        assert len(report.citations) == 1
        assert report.citations[0]["claim"] == "The sky is blue"
        assert len(report.ungrounded_claims) == 0

    @pytest.mark.asyncio
    @patch("friday.safety.call_model")
    async def test_grounding_verifier_ungrounded(self, mock_call):
        mock_call.return_value = (
            "UNGROUNDED\n"
            "- UNGROUNDED: The moon is made of cheese\n"
            "- CITED: The sky is blue\n"
            "Confidence: 0.2"
        )
        
        report = await GroundingVerifier.verify_against_sources(
            response="The sky is blue. The moon is made of cheese.",
            sources=["The sky is blue."],
        )
        
        assert report is not None
        assert report.grounded is False
        assert report.confidence == 0.2
        assert len(report.ungrounded_claims) == 1
        assert report.ungrounded_claims[0] == "The moon is made of cheese"
        assert len(report.citations) == 1

    @pytest.mark.asyncio
    async def test_grounding_verifier_empty(self):
        assert await GroundingVerifier.verify_against_sources("", ["src"]) is None
        assert await GroundingVerifier.verify_against_sources("resp", []) is None

    @pytest.mark.asyncio
    @patch("friday.safety.call_model")
    async def test_grounding_verifier_no_llm_response(self, mock_call):
        mock_call.return_value = None
        report = await GroundingVerifier.verify_against_sources("resp", ["src"])
        assert report is None

    def test_tool_output_validator_none(self):
        res = ToolOutputValidator.validate("test_tool", None, None)
        assert res.valid is False
        assert "returned None" in res.errors[0]
        assert res.sanitized_output is None

    def test_tool_output_validator_error_string(self):
        res = ToolOutputValidator.validate("test_tool", None, "[ERROR] Something went wrong")
        assert res.valid is False
        assert "returned an error" in res.errors[0]
        
        res = ToolOutputValidator.validate("test_tool", None, "error: something went wrong")
        assert res.valid is False
        assert "returned an error" in res.errors[0]

    def test_tool_output_validator_type_conformance(self):
        schema = {"type": "integer"}
        # Valid integer
        res = ToolOutputValidator.validate("test_tool", schema, 42)
        assert res.valid is True
        
        # Invalid integer (string)
        res = ToolOutputValidator.validate("test_tool", schema, "42")
        assert res.valid is False
        assert "expected type" in res.errors[0]

    def test_tool_output_validator_required_keys(self):
        schema = {"required_keys": ["api_key", "endpoint"]}
        # Valid keys
        res = ToolOutputValidator.validate("test_tool", schema, {"api_key": "123", "endpoint": "http"})
        assert res.valid is True
        
        # Missing keys
        res = ToolOutputValidator.validate("test_tool", schema, {"api_key": "123"})
        assert res.valid is False
        assert "missing required keys" in res.errors[0]

    def test_tool_output_validator_redact_keys(self):
        schema = {"redact_keys": ["password", "token"]}
        output = {"password": "secret_pass", "token": "secret_tok", "user": "alice"}
        res = ToolOutputValidator.validate("test_tool", schema, output)
        assert res.valid is True
        assert res.sanitized_output["password"] == "[REDACTED]"
        assert res.sanitized_output["token"] == "[REDACTED]"
        assert res.sanitized_output["user"] == "alice"

    @pytest.mark.asyncio
    @patch("friday.safety.GroundingVerifier.verify_against_sources")
    async def test_verification_chain_with_grounding_and_tools(self, mock_verify_grounding):
        # Mock Grounding Report
        mock_report = GroundingReport(
            grounded=True,
            confidence=0.9,
            ungrounded_claims=[],
            citations=[{"claim": "cited", "source": "matched"}]
        )
        mock_verify_grounding.return_value = mock_report
        
        chain = VerificationChain()
        
        # Mock inner tasks
        async def mock_safety(output):
            return SafetyCheck("pass", False, False)
            
        async def mock_cross(inp, out):
            return (True, "Good")
            
        async def mock_quality(inp, out):
            return (0.9, "Good quality")
            
        chain._safety_filter = mock_safety
        chain._cross_verify = mock_cross
        chain._quality_score = mock_quality
        
        context = {
            "prompt": "Test input",
            "agent_results": [{"agent": "TestAgent", "output": "Test output claims."}],
            "retrieved_sources": ["Source 1 details"],
            "tool_results": [
                {
                    "name": "fetch_data",
                    "output": {"data": "secure"},
                    "expected_schema": {"type": "object"}
                }
            ]
        }
        
        res = await chain.verify(context)
        report = res.get("verification_report")
        assert report is not None
        assert report.grounding is not None
        assert report.grounding.grounded is True
        assert len(report.tool_validations) == 1
        assert report.tool_validations[0].valid is True
