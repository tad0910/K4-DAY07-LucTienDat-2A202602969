# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lục Tiến Đạt
**Nhóm:** kingpro
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai đoạn văn bản có độ tương tự cosine cao thể hiện hai vector biểu diễn có hướng tương đồng nhau trong không gian vector (góc giữa hai vector tiến về 0, $cos(0) \approx 1$), chứng tỏ hai đoạn văn bản có ý nghĩa ngữ nghĩa (semantic meaning) rất giống nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Chính sách bảo hành sản phẩm điện tử là 12 tháng.
- Câu B: Thiết bị điện tử được bảo hành chính hãng trong thời hạn 1 năm.
- Tại sao tương đồng: Cả hai câu đều thể hiện cùng một nội dung về thời gian bảo hành sản phẩm điện tử (12 tháng = 1 năm) dù từ ngữ có sự khác nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Khách hàng được quyền đổi trả sản phẩm trong vòng 7 ngày.
- Câu B: Trái Đất quay xung quanh Mặt Trời theo quỹ đạo hình elip.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn không liên quan tới nhau (quy định mua sắm vs hiện tượng thiên văn).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid phụ thuộc lớn vào độ dài của vector (tức số lượng từ/ký tự của văn bản), khiến hai câu cùng ý nghĩa nhưng một câu dài và một câu ngắn lại có khoảng cách Euclid rất xa. Cosine similarity triệt tiêu ảnh hưởng của độ dài bằng cách chỉ tính góc giữa hai vector, phản ánh chuẩn xác độ tương đồng nội dung.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Công thức: $\text{Số lượng chunk} = \left\lceil \frac{\text{Độ dài} - \text{Overlap}}{\text{Chunk size} - \text{Overlap}} \right\rceil = \left\lceil \frac{10000 - 50}{500 - 50} \right\rceil = \left\lceil \frac{9950}{450} \right\rceil = \left\lceil 22.11 \right\rceil = 23$
> *Đáp án:* 23 chunks

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Phép tính mới: $\left\lceil \frac{10000 - 100}{500 - 100} \right\rceil = \left\lceil \frac{9900}{400} \right\rceil = 25$ chunks (tăng thêm 2 chunks). Tăng độ chồng chéo giúp duy trì liên kết ngữ cảnh giữa các ranh giới chunk, giảm nguy cơ cắt đứt một ý quan trọng hoặc một câu nằm ở điểm giao giữa 2 chunks.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex `(?<=\. |\! |\? |\.\n)` để phân tách các câu trong đoạn văn bản dựa trên các ranh giới câu chuẩn mà không làm mất dấu câu. Loại bỏ khoảng trắng thừa ở hai đầu câu và nhóm từng nhóm `max_sentences_per_chunk` câu lại với nhau bằng dấu cách để tạo ra các chunk hoàn chỉnh.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán duyệt qua danh sách các separator theo thứ tự ưu tiên giảm dần (`\n\n`, `\n`, `. `, ` `, `""`). Nếu độ dài văn bản nhỏ hơn `chunk_size` thì trả về chunk lập tức (base case); nếu lớn hơn thì tiến hành tách theo separator hiện tại và gom các phần lại nếu chưa quá `chunk_size`, đối với phần vượt kích thước sẽ gọi đệ quy `_split` với danh sách separator còn lại.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Duyệt qua danh sách `Document`, với mỗi document tạo bản ghi chuẩn hóa chứa `id`, `doc_id`, `content`, `metadata` và gọi hàm embedding để tạo vector lưu vào `self._store` (và ChromaDB nếu có). Khi tìm kiếm (`search`), embed câu query rồi tính điểm tích vô hướng (dot product) đối với vector của từng chunk, sau đó sắp xếp theo score giảm dần và lấy ra `top_k` kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Với `search_with_filter`, thực hiện lọc trước (pre-filtering) các bản ghi trong store dựa trên `metadata_filter` (kiểm tra tất cả các cặp key-value), sau đó mới chạy hàm tính similarity trên danh sách đã lọc. Với `delete_document`, lọc bỏ tất cả các chunk có `id` hoặc `doc_id` trùng với `doc_id` truyền vào khỏi `self._store` và trả về `True` nếu có ít nhất 1 chunk bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Gọi `store.search(question, top_k)` để truy xuất danh sách các chunk liên quan nhất. Nối nội dung các chunk này thành đoạn văn bản ngữ cảnh (context), sau đó tạo prompt theo mẫu: `Context:\n... \n\nQuestion: ... \nAnswer:` và chuyển prompt sang hàm `llm_fn` để tổng hợp câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= 42 passed in 0.22s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42


---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Khách hàng được quyền đổi trả hàng trong 7 ngày. | Sản phẩm có thể hoàn trả trong vòng 1 tuần. | cao | -0.020 | Sai (do MockEmbedder) |
| 2 | Thời gian bảo hành thiết bị điện tử là 12 tháng. | Sản phẩm điện tử được bảo hành 1 năm chính hãng. | cao | 0.268 | Đúng |
| 3 | Người bán phải đóng gói hàng cẩn thận. | Hàng hóa cần được đóng gói kỹ càng trước khi giao. | cao | 0.097 | Đúng |
| 4 | Khách hàng được quyền đổi trả hàng trong 7 ngày. | Mặt trời mọc ở hướng Đông và lặn ở hướng Tây. | thấp | 0.119 | Sai |
| 5 | Quy định dành cho người bán trên sàn TMĐT. | Hướng dẫn đăng ký tài khoản ngân hàng trực tuyến. | thấp | 0.011 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 1 và Cặp 4 cho kết quả bất ngờ khi hai câu cùng ý nghĩa (Cặp 1) lại có điểm thực tế âm (-0.020), còn hai câu khác biệt ngữ nghĩa (Cặp 4) lại có điểm tương đối cao hơn (0.119). Điều này xảy ra do bài test dùng `MockEmbedder` (băm MD5 ngẫu nhiên dựa trên chuỗi ký tự) nên không thể học và biểu diễn ngữ nghĩa (semantic representation) của ngôn ngữ. Đối với các mô hình Embedding thực sự (như SentenceTransformers hay OpenAI/Gemini), hai câu có cùng ý nghĩa sẽ luôn có vector hướng gần nhau và điểm Cosine Similarity rất cao.


---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người mua thanh toán đơn hàng bằng Thẻ tín dụng/ghi nợ thì nhận được tiền hoàn trong bao lâu? | Bảng 1 phương thức hoàn tiền và các mốc thời gian xử lý hoàn trả (`buyer-refund-timeline`) | 0.331 | Có | Thời gian nhận được tiền hoàn từ 7–14 ngày làm việc sau khi Shopee chấp nhận hoàn tiền. |
| 2 | Đối với sản phẩm thực phẩm tươi sống và đông lạnh, thời gian tối đa để người mua gửi yêu cầu Trả hàng/Hoàn tiền là bao lâu? | Quy định thời gian tiếp nhận yêu cầu trả hàng đối với thực phẩm tươi sống (`buyer-return-eligibility`) | 0.364 | Có | Trong vòng 24 giờ kể từ lúc đơn hàng được cập nhật trạng thái "Giao hàng thành công". |
| 3 | Người bán có thời hạn bao nhiêu ngày để gửi khiếu nại nếu không đồng ý với quyết định Hoàn tiền ngay của Shopee? | Tổng quan thời hạn khiếu nại và các bước phản hồi của Người bán (`seller-refund-appeal`) | 0.318 | Có | Người bán phải gửi khiếu nại trong vòng 2 ngày kể từ khi Shopee gửi thông báo hoàn tiền ngay. |
| 4 | Khi người bán khiếu nại quyết định hoàn tiền ngay không yêu cầu trả hàng của Shopee, loại bằng chứng nào là bắt buộc phải cung cấp? | Bảng quy định các loại bằng chứng khiếu nại bắt buộc đối với Người bán (`seller-return-evidence`) | 0.272 | Có | Bắt buộc cung cấp Bằng chứng đóng gói (video ghi lại quá trình đóng gói sản phẩm). |
| 5 | Trong các phương thức gửi hàng hoàn trả của Shopee, hình thức nào yêu cầu người mua phải thanh toán trước phí trả hàng? | Hướng dẫn các hình thức gửi hàng hoàn trả và quy định hoàn phí (`buyer-return-shipping`) | 0.344 | Có | Hình thức "Tự sắp xếp" yêu cầu Người mua thanh toán trước, sau đó Shopee sẽ hỗ trợ hoàn lại phí. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Việc áp dụng tiền lọc siêu dữ liệu (metadata pre-filtering với `audience: buyer/seller`) là yếu tố quyết định để loại bỏ hoàn toàn các tài liệu gây nhiễu giữa hai đối tượng người mua và người bán. Ngoài ra, việc dùng `RecursiveChunker` giúp các đoạn văn bản giữ trọn vẹn tiêu đề điều khoản, giúp agent trích dẫn nguồn cực kỳ chính xác.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |

