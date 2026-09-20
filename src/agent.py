from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức (Kho dữ liệu rỗng)."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

        context_blocks = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "")
            meta = r.get("metadata", {})
            doc_id = meta.get("doc_id") or r.get("id", "N/A")
            context_blocks.append(f"[{i}] (Nguồn: {doc_id})\n{content}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"Chỉ sử dụng ngữ cảnh được cung cấp dưới đây để trả lời câu hỏi. "
            f"Nếu thông tin không có trong ngữ cảnh, hãy ghi rõ không tìm thấy thông tin trong tài liệu. "
            f"Vui lòng trích dẫn nguồn bằng số thứ tự [1], [2] tương ứng.\n\n"
            f"Ngữ cảnh:\n{context_text}\n\n"
            f"Câu hỏi: {question}\nAnswer:"
        )
        return self.llm_fn(prompt)


