"""
Day 7 — Benchmark Evaluation Script (K4-L3B)
Runs 5 benchmark queries against the e-commerce policy knowledge base,
evaluates retrieval results, and outputs ket_qua_benchmark.txt.
"""

import re
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import RecursiveChunker, FixedSizeChunker, SentenceChunker
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent


class MarkdownHeaderChunker:
    """Chiến lược chia nhỏ tùy chỉnh theo tiêu đề/mục (Markdown Headings) của chính sách gốc."""

    def __init__(self, max_chunk_size: int = 500) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sections = re.split(r"(?=(?:^|\n)#{1,3}\s+)", text)
        chunks = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) > self.max_chunk_size:
                sub_parts = sec.split("\n\n")
                chunks.extend([p.strip() for p in sub_parts if p.strip()])
            else:
                chunks.append(sec)
        return chunks


def load_knowledge_base(data_dir: Path) -> EmbeddingStore:
    chunker = RecursiveChunker(chunk_size=500)
    docs = []

    for p in sorted(data_dir.glob("*.md")):
        content = p.read_text(encoding="utf-8")
        parts = content.split("---")
        body = "---".join(parts[2:]).strip() if len(parts) >= 3 else content.strip()
        fm = dict(re.findall(r"^(\w+):\s*(.+)$", parts[1], re.M)) if len(parts) >= 3 else {}
        doc_id = fm.get("doc_id", p.stem).strip().strip('"').strip("'")
        audience = fm.get("audience", "").strip().strip('"').strip("'")

        chunks = chunker.chunk(body)
        for i, c in enumerate(chunks):
            docs.append(
                Document(
                    id=f"{doc_id}#{i}",
                    content=c,
                    metadata={"doc_id": doc_id, "audience": audience, "source": str(p)},
                )
            )

    store = EmbeddingStore("benchmark_store")
    store.add_documents(docs)
    return store


def run_benchmark() -> str:
    data_dir = PROJECT_ROOT / "data" / "ecommerce"
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} not found.")

    store = load_knowledge_base(data_dir)

    queries = [
        (
            "Người mua thanh toán đơn hàng bằng Thẻ tín dụng/ghi nợ thì nhận được tiền hoàn trong bao lâu?",
            None,
            "7–14 ngày làm việc sau khi Shopee chấp nhận hoàn tiền, về đúng tài khoản thẻ đã thanh toán [1].",
        ),
        (
            "Đối với sản phẩm thực phẩm tươi sống và đông lạnh, thời gian tối đa để người mua gửi yêu cầu Trả hàng/Hoàn tiền là bao lâu?",
            None,
            "Trong vòng 24 giờ kể từ lúc đơn hàng được cập nhật trạng thái 'Giao hàng thành công' [1].",
        ),
        (
            "Người bán có thời hạn bao nhiêu ngày để gửi khiếu nại nếu không đồng ý với quyết định Hoàn tiền ngay của Shopee?",
            {"audience": "seller"},
            "Người bán phải gửi khiếu nại trong vòng 2 ngày kể từ khi Shopee gửi thông báo [1].",
        ),
        (
            "Khi người bán khiếu nại quyết định hoàn tiền ngay không yêu cầu trả hàng của Shopee, loại bằng chứng nào là bắt buộc phải cung cấp?",
            {"audience": "seller"},
            "Bắt buộc cung cấp Bằng chứng đóng gói (video ghi lại quá trình đóng gói sản phẩm) [1].",
        ),
        (
            "Trong các phương thức gửi hàng hoàn trả của Shopee, hình thức nào yêu cầu người mua phải thanh toán trước phí trả hàng?",
            {"audience": "buyer"},
            "Hình thức 'Tự sắp xếp' yêu cầu Người mua thanh toán trước, sau đó Shopee sẽ hỗ trợ hoàn lại [1].",
        ),
    ]

    lines = [
        "=== KẾT QUẢ BENCHMARK RETRIEVAL CÁ NHÂN ===",
        "Sinh viên: Lục Tiến Đạt - 2A202602969",
        "Nhóm: kingprotein",
        "Chiến lược sử dụng: RecursiveChunker (chunk_size=500)",
        "Backend embedding: MockEmbedder (Deterministic test backend)",
        "===========================================================\n",
    ]

    for idx, (q, f, ans) in enumerate(queries, 1):
        res = store.search_with_filter(q, top_k=3, metadata_filter=f) if f else store.search(q, top_k=3)
        lines.append(f"Câu hỏi {idx}: {q}")
        lines.append(f"Metadata filter: {f}")
        lines.append("Top-3 Chunks truy xuất được:")
        for rank, r in enumerate(res, 1):
            doc_id = r.get("metadata", {}).get("doc_id", "N/A")
            score = r.get("score", 0.0)
            snippet = r.get("content", "").replace("\n", " ")[:110]
            lines.append(f"  [{rank}] doc_id={doc_id} | score={score:.4f}")
            lines.append(f"      content: {snippet}...")
        lines.append(f"Câu trả lời chuẩn (Gold Answer): {ans}")
        lines.append("-" * 55)

    # A/B Test for Query 3
    lines.append("\n=== KẾT QUẢ A/B TEST: SO SÁNH CÓ VÀ KHÔNG CÓ METADATA FILTER ===")
    ab_query = "Người bán có thời hạn bao nhiêu ngày để gửi khiếu nại nếu không đồng ý với quyết định Hoàn tiền ngay của Shopee?"
    res_no_filter = store.search(ab_query, top_k=3)
    res_with_filter = store.search_with_filter(ab_query, top_k=3, metadata_filter={"audience": "seller"})

    lines.append(f"Câu hỏi test A/B: {ab_query}\n")
    lines.append("LẦN 1: KHÔNG DÙNG METADATA FILTER")
    for rank, r in enumerate(res_no_filter, 1):
        doc_id = r.get("metadata", {}).get("doc_id", "N/A")
        lines.append(
            f"  Top-{rank}: {doc_id} (audience={r.get('metadata', {}).get('audience')}) | score={r.get('score', 0):.4f}"
        )

    lines.append("\nLẦN 2: CÓ DÙNG METADATA_FILTER={'audience': 'seller'}")
    for rank, r in enumerate(res_with_filter, 1):
        doc_id = r.get("metadata", {}).get("doc_id", "N/A")
        lines.append(
            f"  Top-{rank}: {doc_id} (audience={r.get('metadata', {}).get('audience')}) | score={r.get('score', 0):.4f}"
        )

    result_text = "\n".join(lines)
    output_file = PROJECT_ROOT / "ket_qua_benchmark.txt"
    output_file.write_text(result_text, encoding="utf-8")
    return result_text


if __name__ == "__main__":
    output = run_benchmark()
    print("Benchmark completed successfully. Results written to ket_qua_benchmark.txt")
