"""
F.R.I.D.A.Y. Research Skill
============================
Deep background research: web scraping, PDF processing, and vector indexing.
Supports multiple sources, chunking strategies, and semantic memory integration.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

try:
    import PyPDF2

    _PYPDF2_AVAILABLE = True
except ImportError:
    _PYPDF2_AVAILABLE = False
    logging.warning("[Research] PyPDF2 not installed — PDF processing disabled.")

try:
    import chromadb  # noqa: F401

    _CHROMADB_AVAILABLE = True
except ImportError:
    _CHROMADB_AVAILABLE = False

from friday.memory import MemoryCategory, MemoryTier, get_memory

logger = logging.getLogger(__name__)

# ── Constants & Configuration ─────────────────────────────────────────────────

# Request settings
DEFAULT_TIMEOUT = 15
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) F.R.I.D.A.Y./5.0 (Research Agent)"

# Text processing
MIN_CHUNK_LENGTH = 50
MAX_CHUNK_LENGTH = 1000
OVERLAP_LENGTH = 50

# ── Data Models ───────────────────────────────────────────────────────────────


class ResearchStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class SourceType(Enum):
    WEB = "web"
    PDF = "pdf"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ResearchResult:
    """Standardized result structure for all research operations."""

    success: bool
    message: str
    data: dict[str, Any]
    status: ResearchStatus
    timestamp: float
    sources_processed: int
    chunks_indexed: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class SourceDocument:
    """Represents a single source document (web page, PDF, etc.)."""

    source_id: str
    source_type: SourceType
    url_or_path: str
    title: str
    content: str
    word_count: int
    retrieved_at: float
    error: str | None = None


@dataclass
class ResearchTask:
    """Represents a complete research task with multiple sources."""

    task_id: str
    query: str
    sources: list[str]
    pdf_paths: list[str]
    status: ResearchStatus
    created_at: float
    completed_at: float | None
    chunks_indexed: int
    error: str | None = None


# ── Research Manager ──────────────────────────────────────────────────────────


class ResearchManager:
    """
    Centralized manager for research operations.
    Handles web scraping, PDF processing, text chunking, and memory indexing.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._logger = logging.getLogger("ResearchManager")
        self._active_tasks: dict[str, ResearchTask] = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

        self._logger.info("[Research] Research manager initialized.")

    def _generate_source_id(self, content: str) -> str:
        """Generate a unique ID for a source based on content hash."""
        return hashlib.sha256(content.encode("utf-8")[:1000]).hexdigest()[:16]

    def _generate_task_id(self, query: str) -> str:
        """Generate a unique task ID."""
        timestamp = int(time.time() * 1000)
        return f"research_{timestamp}_{hashlib.md5(query.encode()).hexdigest()[:8]}"

    def _chunk_text(
        self,
        text: str,
        min_length: int = MIN_CHUNK_LENGTH,
        max_length: int = MAX_CHUNK_LENGTH,
        overlap: int = OVERLAP_LENGTH,
    ) -> list[str]:
        """
        Split text into semantic chunks for vector indexing.
        Uses paragraph boundaries when possible.
        """
        chunks = []

        # First split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_length:
                current_chunk += para + "\n\n"
            else:
                if len(current_chunk) >= min_length:
                    chunks.append(current_chunk.strip())

                # Handle oversized paragraphs
                if len(para) > max_length:
                    # Split long paragraphs by sentences
                    sentences = para.split(".")
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) <= max_length:
                            sub_chunk += sent + "."
                        else:
                            if len(sub_chunk) >= min_length:
                                chunks.append(sub_chunk.strip())
                            sub_chunk = sent + "."
                    if sub_chunk and len(sub_chunk) >= min_length:
                        chunks.append(sub_chunk.strip())
                else:
                    current_chunk = para + "\n\n"

        # Add remaining chunk
        if current_chunk and len(current_chunk.strip()) >= min_length:
            chunks.append(current_chunk.strip())

        return chunks

    # ── Web Scraping ──────────────────────────────────────────────────────────

    def scrape_url(self, url: str, timeout: int = DEFAULT_TIMEOUT) -> SourceDocument:
        """
        Fetch and clean text from a URL.
        Returns a SourceDocument with extracted content.
        """
        try:
            resp = self._session.get(url, timeout=timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts, styles, and navigation
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Extract title
            title = soup.title.string if soup.title else url

            # Extract main text
            text = soup.get_text(separator=" ")

            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove excessive whitespace
            text = " ".join(text.split())

            doc = SourceDocument(
                source_id=self._generate_source_id(text),
                source_type=SourceType.WEB,
                url_or_path=url,
                title=title,
                content=text,
                word_count=len(text.split()),
                retrieved_at=time.time(),
            )

            self._logger.info(f"[Research] Scraped {url}: {doc.word_count} words")
            return doc

        except Exception as e:
            self._logger.error(f"[Research] Scraping failed for {url}: {e}")
            return SourceDocument(
                source_id="",
                source_type=SourceType.WEB,
                url_or_path=url,
                title=url,
                content="",
                word_count=0,
                retrieved_at=time.time(),
                error=str(e),
            )

    # ── PDF Processing ────────────────────────────────────────────────────────

    def read_pdf(self, path: str) -> SourceDocument:
        """
        Extract text from a local PDF file.
        Returns a SourceDocument with extracted content.
        """
        if not _PYPDF2_AVAILABLE:
            return SourceDocument(
                source_id="",
                source_type=SourceType.PDF,
                url_or_path=path,
                title=Path(path).name,
                content="",
                word_count=0,
                retrieved_at=time.time(),
                error="PyPDF2 not installed",
            )

        try:
            pdf_path = Path(path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {path}")

            text = ""
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for _i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Clean whitespace
            text = " ".join(text.split())

            doc = SourceDocument(
                source_id=self._generate_source_id(text),
                source_type=SourceType.PDF,
                url_or_path=str(pdf_path.absolute()),
                title=pdf_path.name,
                content=text,
                word_count=len(text.split()),
                retrieved_at=time.time(),
            )

            self._logger.info(f"[Research] Read PDF {path}: {doc.word_count} words")
            return doc

        except Exception as e:
            self._logger.error(f"[Research] PDF read failed for {path}: {e}")
            return SourceDocument(
                source_id="",
                source_type=SourceType.PDF,
                url_or_path=path,
                title=Path(path).name if path else "Unknown",
                content="",
                word_count=0,
                retrieved_at=time.time(),
                error=str(e),
            )

    # ── Memory Indexing ───────────────────────────────────────────────────────

    def index_to_memory(self, documents: list[SourceDocument], query: str) -> int:
        """
        Index document content into semantic memory.
        Returns the number of chunks successfully indexed.
        """
        try:
            memory = get_memory()
            if not memory or not memory.is_ready:
                self._logger.warning("[Research] Memory not ready for indexing")
                return 0

            chunks_indexed = 0

            for doc in documents:
                if not doc.content or doc.error:
                    continue

                # Chunk the content
                chunks = self._chunk_text(doc.content)

                for i, chunk in enumerate(chunks):
                    memory.remember(
                        content=chunk,
                        category=MemoryCategory.FACT,
                        tier=MemoryTier.LONG_TERM,
                        metadata={
                            "source": "research_agent",
                            "source_id": doc.source_id,
                            "source_type": doc.source_type.value,
                            "url_or_path": doc.url_or_path,
                            "title": doc.title,
                            "query": query,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "word_count": doc.word_count,
                            "retrieved_at": doc.retrieved_at,
                        },
                        semantic=True,
                    )
                    chunks_indexed += 1

            self._logger.info(f"[Research] Indexed {chunks_indexed} chunks into semantic memory")
            return chunks_indexed

        except Exception as e:
            self._logger.error(f"[Research] Memory indexing failed: {e}")
            return 0

    # ── Research Execution ────────────────────────────────────────────────────

    def perform_research(
        self, query: str, urls: list[str] | None = None, pdf_paths: list[str] | None = None
    ) -> ResearchResult:
        """
        Execute a complete research task: gather data from all sources, then vectorize.
        This is a blocking operation - use start_background_research for async.
        """
        task_id = self._generate_task_id(query)
        start_time = time.time()

        with self._lock:
            task = ResearchTask(
                task_id=task_id,
                query=query,
                sources=urls or [],
                pdf_paths=pdf_paths or [],
                status=ResearchStatus.IN_PROGRESS,
                created_at=start_time,
                completed_at=None,
                chunks_indexed=0,
            )
            self._active_tasks[task_id] = task

        documents: list[SourceDocument] = []
        errors: list[str] = []

        try:
            # 1. Gather from URLs
            if urls:
                self._logger.info(f"[Research] Scraping {len(urls)} URLs...")
                for url in urls:
                    doc = self.scrape_url(url)
                    documents.append(doc)
                    if doc.error:
                        errors.append(f"URL {url}: {doc.error}")

            # 2. Gather from PDFs
            if pdf_paths:
                self._logger.info(f"[Research] Reading {len(pdf_paths)} PDFs...")
                for path in pdf_paths:
                    doc = self.read_pdf(path)
                    documents.append(doc)
                    if doc.error:
                        errors.append(f"PDF {path}: {doc.error}")

            # Filter successful documents
            successful_docs = [d for d in documents if not d.error and d.content]

            # 3. Vectorize into Memory
            chunks_indexed = 0
            if successful_docs:
                chunks_indexed = self.index_to_memory(successful_docs, query)

            # Determine status
            if chunks_indexed > 0 and not errors:
                status = ResearchStatus.COMPLETED
            elif chunks_indexed > 0 and errors:
                status = ResearchStatus.PARTIAL
            else:
                status = ResearchStatus.FAILED

            # Build message
            msg = f"Sir, I've completed the research on '{query}'. "
            msg += f"Processed {len(successful_docs)} sources, indexed {chunks_indexed} chunks."

            if errors:
                msg += f" {len(errors)} sources had errors."

            end_time = time.time()

            with self._lock:
                task.status = status
                task.completed_at = end_time
                task.chunks_indexed = chunks_indexed
                if errors:
                    task.error = "; ".join(errors)

            return ResearchResult(
                success=(status != ResearchStatus.FAILED),
                message=msg,
                data={
                    "task_id": task_id,
                    "query": query,
                    "sources_processed": len(successful_docs),
                    "total_sources": len(documents),
                    "chunks_indexed": chunks_indexed,
                    "errors": errors,
                    "duration_seconds": round(end_time - start_time, 2),
                    "documents": [asdict(d) for d in successful_docs[:5]],  # Limit for response
                },
                status=status,
                timestamp=end_time,
                sources_processed=len(successful_docs),
                chunks_indexed=chunks_indexed,
            )

        except Exception as e:
            self._logger.error(f"[Research] Research task failed: {e}")

            with self._lock:
                task.status = ResearchStatus.FAILED
                task.completed_at = time.time()
                task.error = str(e)

            return ResearchResult(
                success=False,
                message=f"Sir, the research task failed: {e}",
                data={"error": str(e)},
                status=ResearchStatus.FAILED,
                timestamp=time.time(),
                sources_processed=0,
                chunks_indexed=0,
            )

    def get_task_status(self, task_id: str) -> ResearchTask | None:
        """Get the status of a research task."""
        with self._lock:
            return self._active_tasks.get(task_id)

    def list_active_tasks(self) -> list[ResearchTask]:
        """List all active research tasks."""
        with self._lock:
            return [
                t
                for t in self._active_tasks.values()
                if t.status in (ResearchStatus.PENDING, ResearchStatus.IN_PROGRESS)
            ]

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove completed tasks older than specified hours."""
        removed = 0
        now = time.time()
        max_age_seconds = max_age_hours * 3600

        with self._lock:
            to_remove = []
            for task_id, task in self._active_tasks.items():
                if task.completed_at and (now - task.completed_at) > max_age_seconds:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._active_tasks[task_id]
                removed += 1

        if removed > 0:
            self._logger.info(f"[Research] Cleaned up {removed} old tasks")

        return removed


# ── Background Task Launcher ──────────────────────────────────────────────────

# Global research manager instance
_research_manager: ResearchManager | None = None
_active_threads: dict[str, threading.Thread] = {}


def get_research_manager() -> ResearchManager:
    """Get or create the global research manager singleton."""
    global _research_manager
    if _research_manager is None:
        _research_manager = ResearchManager()
    return _research_manager


def start_background_research(
    query: str, urls: list[str] | None = None, pdf_paths: list[str] | None = None
) -> dict[str, Any]:
    """
    Launch a research task in a background thread.
    Returns immediately with task ID for status tracking.
    """
    manager = get_research_manager()
    task_id = manager._generate_task_id(query)

    def _run():
        result = manager.perform_research(query, urls, pdf_paths)
        # Could notify user here via callback or event
        logger.info(f"[Research] Background task {task_id} completed: {result.status.value}")

    thread = threading.Thread(target=_run, daemon=True, name=f"research-{task_id}")
    thread.start()

    with manager._lock:
        _active_threads[task_id] = thread

    return {
        "success": True,
        "task_id": task_id,
        "message": f"I've initiated background research for '{query}', Sir. Task ID: {task_id[:12]}...",
        "estimated_time": "30-60 seconds per source",
    }


def get_research_status(task_id: str) -> dict[str, Any]:
    """Get the status of a background research task."""
    manager = get_research_manager()
    task = manager.get_task_status(task_id)

    if not task:
        return {"success": False, "message": f"Task {task_id} not found.", "status": "not_found"}

    return {
        "success": True,
        "task_id": task.task_id,
        "query": task.query,
        "status": task.status.value,
        "chunks_indexed": task.chunks_indexed,
        "created_at": datetime.fromtimestamp(task.created_at).strftime("%H:%M:%S"),
        "completed_at": datetime.fromtimestamp(task.completed_at).strftime("%H:%M:%S")
        if task.completed_at
        else None,
        "error": task.error,
    }


# ── Legacy Function Wrappers ──────────────────────────────────────────────────


def scrape_url(url: str) -> dict[str, Any]:
    """Legacy wrapper for URL scraping."""
    doc = get_research_manager().scrape_url(url)
    return {
        "success": not doc.error,
        "url": url,
        "title": doc.title,
        "word_count": doc.word_count,
        "error": doc.error,
        "content_preview": doc.content[:500] if doc.content else "",
    }


def read_pdf(path: str) -> dict[str, Any]:
    """Legacy wrapper for PDF reading."""
    doc = get_research_manager().read_pdf(path)
    return {
        "success": not doc.error,
        "path": path,
        "title": doc.title,
        "word_count": doc.word_count,
        "error": doc.error,
        "content_preview": doc.content[:500] if doc.content else "",
    }


def perform_research(
    query: str, urls: list[str] | None = None, pdf_paths: list[str] | None = None
) -> dict[str, Any]:
    """Legacy wrapper for research execution."""
    result = get_research_manager().perform_research(query, urls, pdf_paths)
    return result.to_dict()


# ── Test Suite ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test Research Skill functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. Research Skill Test Suite")
    print("=" * 60)

    research_mgr = ResearchManager()

    # ── Test 1: Web Scraping ─────────────────────────────────────────────────
    print("\n[TEST 1] Web Scraping...")
    # Using a safe, public test URL
    doc = research_mgr.scrape_url("https://example.com")
    print(f"   Success: {not doc.error}")
    print(f"   Title: {doc.title}")
    print(f"   Word Count: {doc.word_count}")
    if doc.error:
        print(f"   Error: {doc.error}")

    # ── Test 2: Text Chunking ────────────────────────────────────────────────
    print("\n[TEST 2] Text Chunking...")
    sample_text = "This is paragraph one.\n\nThis is paragraph two.\n\n" * 20
    chunks = research_mgr._chunk_text(sample_text)
    print(f"   Input Length: {len(sample_text)} chars")
    print(f"   Chunks Created: {len(chunks)}")
    print(
        f"   Avg Chunk Length: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0} chars"
    )

    # ── Test 3: PDF Reading (if test file exists) ────────────────────────────
    print("\n[TEST 3] PDF Reading...")
    test_pdf = Path("test_document.pdf")
    if test_pdf.exists():
        doc = research_mgr.read_pdf(str(test_pdf))
        print(f"   Success: {not doc.error}")
        print(f"   Word Count: {doc.word_count}")
    else:
        print("   ⚠️  Skipped (no test PDF found)")

    # ── Test 4: Research Task Creation ───────────────────────────────────────
    print("\n[TEST 4] Research Task Creation...")
    task_id = research_mgr._generate_task_id("Test research query")
    print(f"   Task ID: {task_id}")
    print("   Format: research_<timestamp>_<hash>")

    # ── Test 5: Background Research Launcher ─────────────────────────────────
    print("\n[TEST 5] Background Research Launcher...")
    result = start_background_research(
        query="Test background research", urls=["https://example.com"]
    )
    print(f"   Success: {result['success']}")
    print(f"   Task ID: {result.get('task_id', 'N/A')}")
    print(f"   Message: {result['message']}")

    # ── Test 6: Task Status Check ────────────────────────────────────────────
    print("\n[TEST 6] Task Status Check...")
    if "task_id" in result:
        status = get_research_status(result["task_id"])
        print(f"   Status: {status.get('status', 'unknown')}")
        print(f"   Query: {status.get('query', 'unknown')}")

    # ── Test 7: Cleanup Old Tasks ────────────────────────────────────────────
    print("\n[TEST 7] Cleanup Old Tasks...")
    removed = research_mgr.cleanup_old_tasks(max_age_hours=24)
    print(f"   Tasks Removed: {removed}")

    # ── Test 8: Active Tasks List ────────────────────────────────────────────
    print("\n[TEST 8] Active Tasks List...")
    active = research_mgr.list_active_tasks()
    print(f"   Active Tasks: {len(active)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Research Skill tests completed!")
    print("=" * 60)
