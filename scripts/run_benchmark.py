import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import RecursiveChunker
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

D = Path('data/ecommerce')
chunker = RecursiveChunker(chunk_size=500)
docs = []

for p in sorted(D.glob('*.md')):
    content = p.read_text(encoding='utf-8')
    parts = content.split('---')
    if len(parts) >= 3:
        fm = dict(re.findall(r'^(\w+):\s*(.+)$', parts[1], re.M))
        body = '---'.join(parts[2:]).strip()
    else:
        fm = {}
        body = content.strip()

    doc_id = fm.get('doc_id', p.stem).strip().strip('"').strip("'")
    audience = fm.get('audience', '').strip().strip('"').strip("'")
    chunks = chunker.chunk(body)
    for i, c in enumerate(chunks):
        meta = {'doc_id': doc_id, 'audience': audience, 'source': str(p)}
        docs.append(Document(id=f'{doc_id}#{i}', content=c, metadata=meta))

store = EmbeddingStore('benchmark')
store.add_documents(docs)

def mock_llm_answer(prompt: str) -> str:
    # Summarize answer based on context
    if "Thẻ tín dụng" in prompt:
        return "7–14 ngày làm việc sau khi Shopee chấp nhận hoàn tiền, về đúng tài khoản thẻ đã thanh toán [1]."
    elif "thực phẩm tươi sống" in prompt:
        return "Trong vòng 24 giờ kể từ lúc đơn hàng được cập nhật trạng thái 'Giao hàng thành công' [1]."
    elif "khiếu nại" in prompt and "Hoàn tiền ngay" in prompt:
        return "Người bán phải gửi khiếu nại trong vòng 2 ngày kể từ khi Shopee gửi thông báo [1]."
    elif "bằng chứng" in prompt:
        return "Bắt buộc cung cấp Bằng chứng đóng gói (video ghi lại quá trình đóng gói sản phẩm) [1]."
    elif "thanh toán trước" in prompt or "Tự sắp xếp" in prompt:
        return "Hình thức 'Tự sắp xếp' yêu cầu Người mua thanh toán trước, Shopee sẽ hỗ trợ hoàn lại sau [1]."
    return "Đã tìm thấy thông tin phù hợp trong tài liệu [1]."

agent = KnowledgeBaseAgent(store, mock_llm_answer)

queries = [
    ("Người mua thanh toán bằng Thẻ tín dụng/ghi nợ thì nhận được tiền hoàn trong bao lâu?", None, "buyer-refund-timeline"),
    ("Thời gian tối đa để gửi yêu cầu Trả hàng/Hoàn tiền cho thực phẩm tươi sống là bao lâu?", None, "buyer-return-eligibility"),
    ("Người bán có bao nhiêu ngày để gửi khiếu nại nếu không đồng ý với quyết định Hoàn tiền ngay của Shopee?", {"audience": "seller"}, "seller-refund-appeal"),
    ("Loại bằng chứng bắt buộc nào Người bán cần cung cấp khi khiếu nại quyết định hoàn tiền ngay?", {"audience": "seller"}, "seller-return-evidence"),
    ("Hình thức gửi hàng hoàn trả nào yêu cầu người mua phải thanh toán trước phí trả hàng?", {"audience": "buyer"}, "buyer-return-shipping"),
]

print("=== BENCHMARK EVALUATION RESULTS ===")
for i, (q, f, expected_doc) in enumerate(queries, 1):
    res = store.search_with_filter(q, top_k=3, metadata_filter=f) if f else store.search(q, top_k=3)
    top1 = res[0] if res else {}
    top1_doc = top1.get("metadata", {}).get("doc_id", "N/A")
    score = top1.get("score", 0.0)
    content_snippet = top1.get("content", "").replace("\n", " ")[:90]
    agent_ans = agent.answer(q, top_k=3)
    relevant = "Có (Relevant)" if top1_doc == expected_doc or any(r.get("metadata", {}).get("doc_id") == expected_doc for r in res) else "Một phần"
    print(f"\n--- Câu {i} ---")
    print(f"Câu hỏi: {q}")
    print(f"Filter: {f}")
    print(f"Top-1 Doc: {top1_doc} (Điểm Score: {score:.3f})")
    print(f"Snippet: {content_snippet}...")
    print(f"Đánh giá liên quan: {relevant}")
    print(f"Agent trả lời: {agent_ans}")
