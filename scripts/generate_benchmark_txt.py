import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import RecursiveChunker, FixedSizeChunker, SentenceChunker
from src.models import Document
from src.store import EmbeddingStore

D = Path("data/ecommerce")
chunker = RecursiveChunker(chunk_size=500)
docs = []

for p in sorted(D.glob("*.md")):
    content = p.read_text(encoding="utf-8")
    parts = content.split("---")
    body = "---".join(parts[2:]).strip() if len(parts) >= 3 else content.strip()
    fm = dict(re.findall(r"^(\w+):\s*(.+)$", parts[1], re.M)) if len(parts) >= 3 else {}
    doc_id = fm.get("doc_id", p.stem).strip().strip('"').strip("'")
    audience = fm.get("audience", "").strip().strip('"').strip("'")
    chunks = chunker.chunk(body)
    for i, c in enumerate(chunks):
        docs.append(Document(id=f"{doc_id}#{i}", content=c, metadata={"doc_id": doc_id, "audience": audience, "source": str(p)}))

store = EmbeddingStore("benchmark")
store.add_documents(docs)

queries = [
    ("Người mua thanh toán đơn hàng bằng Thẻ tín dụng/ghi nợ thì nhận được tiền hoàn trong bao lâu?", None),
    ("Đối với sản phẩm thực phẩm tươi sống và đông lạnh, thời gian tối đa để người mua gửi yêu cầu Trả hàng/Hoàn tiền là bao lâu?", None),
    ("Người bán có thời hạn bao nhiêu ngày để gửi khiếu nại nếu không đồng ý với quyết định Hoàn tiền ngay của Shopee?", {"audience": "seller"}),
    ("Khi người bán khiếu nại quyết định hoàn tiền ngay không yêu cầu trả hàng của Shopee, loại bằng chứng nào là bắt buộc phải cung cấp?", {"audience": "seller"}),
    ("Trong các phương thức gửi hàng hoàn trả của Shopee, hình thức nào yêu cầu người mua phải thanh toán trước phí trả hàng?", {"audience": "buyer"}),
]

lines = [
    "=== KẾT QUẢ BENCHMARK RETRIEVAL CÁ NHÂN ===",
    "Sinh viên: Lục Tiến Đạt - 2A202602969",
    "Nhóm: kingprotein",
    "Chiến lược sử dụng: RecursiveChunker (chunk_size=500)",
    "Backend embedding: MockEmbedder (lưu ý: số liệu điểm tương tự mang tính đại diện kiểm thử cấu trúc)",
    "===========================================================\n",
]

for idx, (q, f) in enumerate(queries, 1):
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
    lines.append("-" * 55)

# A/B Test for Query 3 (With vs Without filter)
lines.append("\n=== KẾT QUẢ A/B TEST: SO SÁNH CÓ VÀ KHÔNG CÓ METADATA FILTER ===")
ab_query = "Người bán có thời hạn bao nhiêu ngày để gửi khiếu nại nếu không đồng ý với quyết định Hoàn tiền ngay của Shopee?"
res_no_filter = store.search(ab_query, top_k=3)
res_with_filter = store.search_with_filter(ab_query, top_k=3, metadata_filter={"audience": "seller"})

lines.append(f"Câu hỏi test A/B: {ab_query}\n")
lines.append("LẦN 1: KHÔNG DÙNG METADATA FILTER")
for rank, r in enumerate(res_no_filter, 1):
    doc_id = r.get("metadata", {}).get("doc_id", "N/A")
    lines.append(f"  Top-{rank}: {doc_id} (audience={r.get('metadata', {}).get('audience')}) | score={r.get('score', 0):.4f}")

lines.append("\nLẦN 2: CÓ DÙNG METADATA_FILTER={'audience': 'seller'}")
for rank, r in enumerate(res_with_filter, 1):
    doc_id = r.get("metadata", {}).get("doc_id", "N/A")
    lines.append(f"  Top-{rank}: {doc_id} (audience={r.get('metadata', {}).get('audience')}) | score={r.get('score', 0):.4f}")

Path("ket_qua_benchmark.txt").write_text("\n".join(lines), encoding="utf-8")
print("Successfully written ket_qua_benchmark.txt!")
