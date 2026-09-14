# VBS 2027 — Thiết kế feedback theo event và khoảng bằng chứng

- **Loại tài liệu:** Research & system design specification theo workflow `superpowers:brainstorming`.
- **Phiên bản:** 0.1 — hợp nhất thiết kế đã được người dùng duyệt.
- **Ngày:** 13/09/2026.
- **Nền hệ thống:** `src_hcmai_v20.zip` nguyên bản; không lấy prototype agent do assistant từng tự viết làm baseline.
- **Branch implementation bắt buộc:** `vbs`.
- **Mục tiêu submission của nhóm:** VBS trước 22/09/2026; hướng entity-aware của thành viên khác dành cho SOICT trước 05/10/2026.
- **Trạng thái nghiên cứu:** Hướng phát triển đã chốt; novelty và hiệu quả tương tác chưa được chứng minh.
- **Phạm vi bàn giao:** Đặc tả thiết kế, căn cứ, ranh giới tích hợp và đánh giá. Không phải implementation hay cam kết publish.

## 1. Phân công và tài liệu có thẩm quyền

| Chủ thể | Trách nhiệm |
|---|---|
| Người dùng và assistant cùng thảo luận | Hướng nghiên cứu, cơ chế interactive, tối ưu hệ thống bên dưới, phản biện contribution |
| Assistant | Research, đọc/audit code, đối chiếu prior work, đề xuất và phản biện; không tự implementation |
| Người dùng | Implementation, tích hợp và chạy hệ thống trên branch `vbs` |
| Thành viên phụ trách frontend | UI/UX, AnswerWorkspace và các thay đổi API/DRES được mô tả trong plan của thành viên đó |

Plan tích hợp có thẩm quyền là **`2026-09-12-vbs-2027-dres-answer-workspace.md`** do người dùng cung cấp. Tài liệu này bổ sung thiết kế exploration; không thay thế plan đó và không tự đổi schema submission, session hay workspace.

Tránh hướng entity-aware: không lấy entity binding, coreference, tracking/ReID hoặc identity consistency làm contribution VBS. Tìm người/vật theo nội dung query thông thường vẫn là chức năng retrieval.

## 2. Bài toán và task đích

VBS 2027 liệt kê KIS-V, KIS-T, KIS-C, AVS và VQA; không liệt kê TRAKE như task thi. KIS-C có manh mối được làm rõ trong quá trình hỏi/đáp; không đồng nghĩa bắt buộc xây chatbot. Thiết kế này ưu tiên KIS-T/KIS-C có diễn biến thời gian. Temporal alignment là công cụ nội bộ để tìm và kiểm tra đáp án. [CFP VBS 2027](https://videobrowsershowdown.org/call-for-papers/), [Task definitions](https://videobrowsershowdown.org/about-vbs/).

**Vấn đề được chọn:** Người dùng tìm được một phần bằng chứng hoặc một video có triển vọng, nhưng còn phải xác định đúng thời điểm và các sự kiện còn thiếu. Phản hồi relevant/irrelevant cho toàn ứng viên không biểu đạt rõ việc một khoảng đúng với event A nhưng không phù hợp với event B.

**Mục tiêu chức năng:** Giữ các điều kiện người dùng đã xác nhận, sửa từng phần chưa phù hợp và giải lại alignment; có thể hoàn tác hoặc quay lại kết quả toàn corpus.

**Giả thuyết nghiên cứu:** Feedback tích lũy theo event–khoảng có thể giảm công sức tìm đúng thời điểm so với cùng công cụ tìm cục bộ nhưng người dùng phải nhập lại điều kiện. Đây là giả thuyết, không phải kết quả hiện có.

## 3. Căn cứ từ benchmark của nhóm

Nguồn: `results_benchmarked_baselines.zip`, report và metrics per-query trong `artifacts/baselines/`. Bộ đánh giá có 30 query: 19 KIS, 9 QA, 2 TRAKE. Các run ghi `use_dense=true`, `use_bm25=false`; không đại diện cho toàn bộ cấu hình multimodal có thể bật của hệ thống.

### FF-MDP trên 19 query KIS

| Độ sâu | Có đúng video | Path có frame khớp mục tiêu trong ±5 giây |
|---|---:|---:|
| Top-1 | 12/19 | 8/19 |
| Top-5 | 15/19 | 13/19 |
| Top-10 | 17/19 | 16/19 |

Bốn query có đúng video top-1 nhưng path top-1 chưa khớp trong ±5 giây: `query-p2-14-kis.txt`, `query-p2-18-kis.txt`, `query-p2-3-kis.txt`, `query-p2-6-kis.txt`. Đây là nhóm phân tích failure cases, không phải toàn bộ tập đánh giá tương tác.

Các quan sát khác:

- TK-MDP và FF-MDP cùng KIS video R@1 = 63,16%; alignment mean trong run lần lượt 0,742 s và 0,139 s. Chưa có căn cứ ưu tiên lattice vì giả định sparse luôn nhanh.
- QA relaxed Path R@10: IEM 7/9, FF-MDP 5/9. Không áp strict chronology cho mọi câu hỏi.
- Exact matching gần như bằng 0. Cần phân biệt độ thưa frame, ground truth và tọa độ thời gian; không mặc định toàn bộ là lỗi decoder.
- TRAKE chỉ có hai query; kết quả này không chứng minh hiệu quả VBS.
- Latency là một lượt chạy mỗi phương pháp, chưa kiểm soát đầy đủ warm-up/cache/run order. Không dùng để claim tốc độ tổng quát.
- `dante_style_unary_dp` là tên artifact của baseline có objective unary monotonic liên quan DANTE, không phải reproduction toàn bộ DANTE.

**Quyết định engineering:** Giữ FF-MDP làm nền cho lát cắt đầu tiên; ưu tiên cải thiện quá trình tương tác sau retrieval. Không cần chờ benchmark lớn hơn để bắt đầu implementation, nhưng phải giới hạn claim theo bằng chứng.

## 4. Prior work và giới hạn novelty

| Nguồn | Điều đã xác nhận | Hệ quả với thiết kế |
|---|---|---|
| [Exquisitor VBS2025](https://link.springer.com/chapter/10.1007/978-981-96-2074-6_31) | Kết hợp conversational search, relevance feedback và hợp nhất kết quả; xác nhận từ abstract | Chat + feedback không đủ làm novelty |
| [Exquisitor VBS2026](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_27) | Abstract mô tả sequence-chain + RRF cho temporal query và in-video search cho QA | Tìm trong video hoặc temporal querying không mới tự thân |
| [Interactive VCMR using RL, ACM MM2022](https://ink.library.smu.edu.sg/sis_research/7506/) | Abstract mô tả điều hướng theo feedback, đánh giá TVR/DiDeMo | Dùng feedback chọn bước tiếp theo đã có tiền lệ; chưa đối chiếu chi tiết toàn văn |
| [Phân tích VBS2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8791088/) | Có temporal browsing, temporal queries và trường hợp tìm được video nhưng không hoàn thành tìm cảnh | Vấn đề video-to-moment có cơ sở thực tiễn; temporal context là baseline cần có |
| [Vgent](https://arxiv.org/html/2510.14032v1) | Structured verification/reasoning sau retrieval; ablation MLVU: GraphRAG 69,5, thêm SR 72,1 với Qwen2.5-VL | Verification có cơ sở nghiên cứu, nhưng không chứng minh checklist của nhóm cải thiện VBS; không theo entity graph của bài |
| [VideoSearch-R1](https://arxiv.org/html/2607.00446v1) | Iterative retrieval, verification và temporal grounding, có soft query refinement | Search → Verify → Search không phải claim mới riêng |

**Contribution dự kiến:** Cơ chế tương tác tích lũy phản hồi có phạm vi ngữ nghĩa và thời gian, giúp giữ phần bằng chứng đã xác nhận và sửa phần còn thiếu.

Không claim: “first”, thuật toán DP mới, k-best decoding mới, entity-aware retrieval, hoặc cải thiện mọi task VBS. Chưa có đủ bằng chứng để nói prior systems không hỗ trợ cơ chế tương đương. Việc chưa truy cập full text không được diễn giải thành limitation của họ.

## 5. Phạm vi bản đầu

### Bao gồm

- Query có danh sách event thời gian rõ ràng.
- Một nhánh cục bộ đang hoạt động trong một video.
- Giữ kết quả toàn corpus để quay lại.
- Xác nhận khoảng cho một event, bác bỏ khoảng đối với một event, trạng thái chưa rõ.
- Giải lại toàn bộ FF-MDP dưới điều kiện hiện tại.
- Hiển thị điều kiện được giữ và vị trí frame thay đổi.
- Hoàn tác, quản lý phiên bản decomposition và tái sử dụng score matrix.

### Ngoài phạm vi bản đầu

- Cây nhiều nhánh giả thuyết, tự động điều hướng nhiều vòng, reinforcement learning.
- Agent tự xác nhận hoặc loại ứng viên.
- Entity-aware, crop verification, arbitrary clip sampling, tự sinh đáp án VQA.
- Incremental prefix/suffix DP, true k-best paths, decoder mới.
- Tái thiết kế UI, thay AnswerWorkspace hoặc DRES adapter.

Agent là hướng mở rộng phục vụ diễn giải và kiểm tra; không bắt buộc để kiểm chứng cơ chế feedback. Query một cảnh vẫn tìm và duyệt bình thường, không ép tách nhiều event.

## 6. Các khái niệm và invariants

Đây là nội dung trạng thái cần quản lý, không yêu cầu tạo một DTO/class mới cho mỗi hàng.

| Khái niệm | Ý nghĩa |
|---|---|
| Query gốc | Nhu cầu người dùng, được giữ khi khám phá cục bộ |
| Danh sách event có phiên bản | Các retrieval unit có thứ tự, đang được người dùng xem |
| Nhánh cục bộ | Giả thuyết khám phá trong một video; không đồng nghĩa video đích đã đúng |
| Khoảng xác nhận | Miền thời gian được phép cho một event trong nhánh |
| Khoảng bác bỏ | Miền thời gian không dùng cho event được chỉ định |
| Lịch sử thao tác | Cho phép khôi phục bộ điều kiện trước đó |
| Path trước/hiện tại | Dùng so sánh thay đổi trong cùng phiên bản event |

Invariants:

1. Feedback gắn với đúng event, phiên bản decomposition, video và khoảng thời gian.
2. Dùng `event_index` zero-based hiện có, kèm phiên bản; không tự đưa thêm hệ entity/event identity song song.
3. “Chưa thấy” khác “đã xác nhận không phù hợp”. Không suy negative video từ vài frame thiếu bằng chứng.
4. Một event bị bác bỏ ở một khoảng không làm khoảng đó mất giá trị cho event khác.
5. Xác nhận khoảng không có nghĩa ghim một frame bất biến bên trong khoảng.
6. Không tự nới điều kiện khi giải không thành công.
7. Score matrix gốc không bị feedback của một người dùng sửa tại chỗ.
8. Exploration revision độc lập với workspace revision.
9. Không xem việc mở/xem một frame là xác nhận relevance.

## 7. Semantics của thao tác

Quy ước thời gian cho exploration: integer milliseconds, khoảng đóng `[start_ms, end_ms]`, `0 <= start_ms <= end_ms`. Cho phép hai đầu bằng nhau; nếu không có frame index tương ứng, báo đúng trạng thái, không tự snap. Đây không thay đổi interval semantics của DRES.

| Thao tác | Tác động |
|---|---|
| Tìm quanh mốc | Chọn phạm vi khám phá bằng giới hạn thời gian rõ ràng; không tự tạo xác nhận event |
| Xác nhận A trong `[a,b]` | A chỉ được chọn frame trong khoảng đó ở nhánh hiện tại |
| Xác nhận lại A | Thay khoảng xác nhận A trước đó; lịch sử cho phép hoàn tác |
| Bác bỏ `[c,d]` cho B | Hợp khoảng này vào các khoảng loại của B |
| Chưa đủ bằng chứng cho B | Không thay mask; người dùng xem thêm ngữ cảnh |
| Hoàn tác | Khôi phục bộ điều kiện trước đó và giải lại |
| Bỏ nhánh cục bộ | Quay về kết quả toàn corpus đã giữ; điều kiện nhánh không tự lan ra toàn corpus |
| Đổi danh sách event | Feedback cũ không tự áp theo index; cần đối chiếu lại trước khi tiếp tục áp dụng |

“Trước/sau mốc” phải thể hiện rõ hướng tìm và phạm vi được chọn. Không ngầm dùng một cửa sổ cố định rồi coi đó là toàn video. Xác nhận A trước B chỉ áp dụng khi quan hệ thời gian thuộc query hoặc được người dùng yêu cầu.

## 8. Thiết kế thuật toán trên v20

### 8.1 Các điểm tái sử dụng

| File/thành phần trong repository v20 | Vai trò |
|---|---|
| `src/hcmai/orchestration/workflows/kis.py` | Planner, gọi temporal service và materialize KIS |
| `src/hcmai/temporal/planner.py` | Quy tắc tách/gộp câu thành event; không quản lý phiên bản qua các lượt |
| `src/hcmai/retrieval/retriever/video_scores.py` | `VideoEventScores`: matrix event × frame và canonical metadata |
| `src/hcmai/retrieval/evidence/hybrid.py` | Scoring/fusion; giữ quy ước chuẩn hóa nhất quán |
| `src/hcmai/orchestration/workflows/temporal_search.py` | Scoring, validate identity, decode và materialize |
| `src/hcmai/temporal/dp.py` | Prefix-max monotonic DP, `DPPath`, `AlignedPath` |

Các path trên là vị trí audit trong source của dự án, không phải schema mới.

### 8.2 Mask theo event–frame

Với event e, frame j ở timestamp t_j, đặt M[e,j] là điều kiện hợp lệ:

- Frame thuộc video và phạm vi nhánh đang xét.
- Nếu e có xác nhận, t_j thuộc khoảng xác nhận của e.
- t_j không thuộc bất kỳ khoảng bác bỏ nào dành cho e.

Giữ score gốc S; tính biến đổi điểm như baseline rồi áp mask để cấm trạng thái DP không hợp lệ. Không dùng model để tự tăng/giảm điểm thay cho phản hồi rõ ràng.

**Cảnh báo correctness từ audit:** `align_video()` có `clip(scores, 0, ...) ** event_power`. Đưa `-inf` vào trước đó có thể biến trạng thái bị cấm thành 0. `cluster_starts()` cũng không nên nhận mask được mã hóa bằng `-inf`. Tính clustering trên điểm gốc/đã biến đổi theo baseline, sau đó áp mask ở DP.

### 8.3 Giữ objective và giải lại toàn bộ

- Giữ tất cả hàng event trong DP, kể cả event đã xác nhận.
- Giới hạn miền hợp lệ của event đã xác nhận; không xóa event đó khỏi chuỗi.
- Giữ lambda_gap và cấu hình baseline để so sánh có kiểm soát.
- Với ràng buộc độc lập theo event–frame, recurrence prefix-max vẫn O(E × F). Đây là phân tích recurrence; không bao gồm scoring, sorting endpoint và materialization, không phải latency đo mới.
- Strict order hiện tại là tăng cột frame; không tự đổi sang partial order/simultaneity. Thuộc tính đồng thời cần gắn vào event phù hợp thay vì thêm một bước thời gian giả.
- Path score chỉ so sánh có kiểm soát dưới cùng query/scoring semantics; không trình bày là xác suất đúng.

### 8.4 Thay đổi path và giới hạn alternatives

So sánh bằng canonical frame IDs/timestamps theo event position trong cùng phiên bản event. Hiển thị riêng điều kiện giữ nguyên và frame đại diện thay đổi. Nếu A được xác nhận 01:20–01:25, A chuyển từ 01:21 sang 01:23 vẫn tôn trọng xác nhận.

Decoder hiện giữ một predecessor tốt nhất mỗi trạng thái và chọn các endpoint khác nhau; không phải true k-best paths. Bản đầu chỉ yêu cầu path tốt nhất còn hợp lệ sau phản hồi, không hứa liệt kê mọi giả thuyết.

## 9. Tái sử dụng scoring và phiên bản

`TemporalSearchService.search()` hiện chấm điểm trước mỗi decode. Khi chỉ thay feedback mà event/scoring không đổi, tái sử dụng score matrix của video đang mở và giải lại DP.

Cache cần phân biệt video, nội dung/phiên bản event, retrieval rewrite, nguồn scoring và cấu hình/index liên quan. Thay đổi một thành phần làm score không còn tương đương thì không dùng lại cache đó. Không xây cache toàn corpus cho mọi phiên ở lát cắt đầu.

Không chuẩn hóa lại điểm theo cửa sổ mới rồi so trực tiếp với điểm toàn corpus. Mask không sửa cached matrix dùng chung. TTL/giới hạn dung lượng là quyết định deployment của người implement, không phải contribution và chưa được chọn bằng benchmark.

Khi danh sách event đổi, giữ feedback cũ như lịch sử để đối chiếu; không âm thầm chuyển nó sang event khác. Không tự chạy LLM remapping rồi coi kết quả là xác nhận của người dùng.

## 10. Luồng xuyên suốt

Ví dụ tự tạo: “Một người mở tủ, lấy chiếc cốc rồi đặt lên bàn.”

1. Tìm toàn corpus; trả ứng viên và các mốc temporal.
2. Người dùng mở một video, xác nhận “mở tủ” trong 01:20–01:25.
3. Tạo nhánh cục bộ; giải lại chuỗi, giữ miền hợp lệ của event mở tủ.
4. Người dùng bác bỏ 01:35–01:40 đối với “đặt cốc”.
5. Chỉ loại khoảng đó cho event đặt cốc; giải lại trên score đã có.
6. Hiển thị path mới, những frame đổi và điều kiện được giữ.
7. Người dùng hoàn tác nếu cần, hoặc chọn thời điểm để mở dialog đáp án.

Nếu nhánh không thành công: gỡ/sửa điều kiện hoặc quay lại toàn corpus. Không tự loại video vì nhánh thất bại.

## 11. Trạng thái kết quả và lỗi

Các tên dưới là nhãn ngữ nghĩa của đặc tả, chưa phải tên enum hoặc HTTP status bắt buộc.

| Trạng thái | Ý nghĩa | Cách phản hồi |
|---|---|---|
| Có path | Có chuỗi hợp lệ dưới điều kiện và decoder hiện tại | Trả canonical path và phần thay đổi |
| Điều kiện mâu thuẫn | Có thể xác định điều kiện không thể đồng thời thỏa mãn, ví dụ xác nhận bị loại hết | Giữ điều kiện để người dùng sửa/hoàn tác; không tự nới |
| Không có frame trong miền | Miền đã chọn không có frame index phù hợp | Báo thiếu dữ liệu frame cho miền; không snap |
| Không có path hợp lệ | Miền có frame nhưng không dựng được chuỗi theo order/clustering/cấu hình hiện tại | Không suy ra video sai; cho phép sửa điều kiện hoặc mở rộng |
| Phiên bản event không khớp | Feedback thuộc decomposition khác | Không áp feedback; đối chiếu lại |
| Dữ liệu/scoring không sẵn sàng | Không thể tính kết quả do hệ thống | Phân biệt với kết luận về nội dung video |

Không có path không tự đồng nghĩa điều kiện người dùng mâu thuẫn. Nếu chưa đủ cơ sở chứng minh mâu thuẫn, dùng mô tả bảo thủ về việc không dựng được path.

## 12. Hợp đồng tích hợp với frontend và AnswerWorkspace

### 12.1 Input/output ngữ nghĩa của exploration

Input cần xác định query/event version, video nhánh, event_index nếu thao tác nhắm event, khoảng thời gian, loại thao tác và phiên bản trạng thái exploration.

Output gồm trạng thái kết quả, điều kiện đang hiệu lực, canonical path, thay đổi so với path trước, phiên bản exploration và thông tin cần thiết để hoàn tác. Các đầu ra không còn đúng phiên bản hiện tại không được thay kết quả mới trên UI.

Đây là hợp đồng hành vi; không tự đặt endpoint mới hoặc sửa public schema hiện có trong tài liệu research. Người implement chọn cách mở rộng sau khi rà router/contracts hiện hành, không tạo bản sao `Frame`/`AlignedPath`.

### 12.2 Phần thuộc UI/UX của thành viên khác

- Cho biết feedback đang áp vào event và khoảng nào.
- Phân biệt “tìm quanh đây” với “event đã đúng ở đây”.
- Hiển thị điều kiện đang giữ, frame thay đổi và nút hoàn tác/quay lại.
- Không bắt người dùng đọc giải thích dài để thực hiện thao tác cơ bản.

### 12.3 Phần giữ nguyên theo plan DRES

- AnswerWorkspace tiếp tục quản lý FRAME/TEXT, AVS mode, workspace revisions, task binding và submission attempts.
- Exploration không ghi trực tiếp vào answer workspace và không dùng workspace revision để quản lý nhánh.
- Chọn đáp án chỉ mở dialog editable; Enter lưu candidate, không submit.
- Timestamp tùy ý trong inspector được capture tại click; không dùng nearest keyframe để thay timestamp nộp. Provenance source frame chỉ có khi thực sự tương ứng.
- Submit vẫn qua xác nhận read-only, snapshot và trạng thái FORWARDING/UNKNOWN theo plan.
- API session/credentials/DRES client, logging và migration submission cũ do plan của thành viên kia sở hữu.
- Kết quả retrieval từ exploration cần đi qua logging chung khi tích hợp; không tạo DRES client hoặc cơ chế log cạnh tranh.
- Nội dung giải thích agent/feedback không đưa vào body của answer row hoặc payload DRES.

## 13. Agent và model

MVP của cơ chế feedback không phụ thuộc model. Nếu thêm agent sau:

- Agent diễn giải đề xuất event/phạm vi/hành động từ lời người dùng.
- Agent kiểm tra đọc bằng chứng và có thể trả “chưa rõ”.
- Backend thực thi điều kiện; người dùng xác nhận hành động ảnh hưởng miền tìm.
- Hai vai trò không bắt buộc là hai model; không mặc định multiagent là novelty.
- Không tự cập nhật hard constraints, không submit, không tự coi metadata là bằng chứng thị giác đã xác nhận.

CPU embeddings và hạ tầng hiện có được giữ theo quyết định của người dùng. Lựa chọn model/API, quota và deployment cụ thể nằm ngoài đặc tả lát cắt này; không hardcode một quota miễn phí hoặc deadline model thành kết quả thực nghiệm.

## 14. Thiết kế đánh giá

| Cấu hình | Thành phần | Mục đích |
|---|---|---|
| A | Retriever hiện tại, player, temporal context | Nền duyệt thông thường |
| B | A + tìm theo video/khoảng; người dùng nhập lại điều kiện | Đo lợi ích công cụ tìm cục bộ |
| C | Cùng công cụ như B + feedback tích lũy event–khoảng | Đo giá trị cơ chế được đề xuất |

**So sánh chính là B–C.** C thắng A chưa tách được lợi ích feedback khỏi lợi ích có thêm tìm kiếm cục bộ. Giữ cùng retriever, scoring, data và ngân sách thời gian; điều kiện dùng model phải tương đương. Không chỉ cho C dùng model mạnh hơn.

Chỉ số chính:

1. Tỷ lệ hoàn thành đúng trong cùng giới hạn thời gian.
2. Thời gian hoàn thành, kèm tỷ lệ thất bại; không chỉ báo trung bình trên các câu thành công.
3. Công sức làm lại: nhập lại điều kiện và xem lại bằng chứng đã xác nhận, định nghĩa nhất quán khi ghi log.

Số click là chỉ số phụ; ít click nhưng chờ lâu hơn không tự tốt hơn. Với người dùng thử cùng các điều kiện, cần phân bổ task/thứ tự để giảm hiệu ứng nhớ đáp án. Bộ query đã xem đáp án dùng cho development/demo phải được nhận diện riêng.

Không chỉ đánh giá bốn failure cases thuận lợi. Benchmark 30 query hiện tại giúp chọn baseline và phân tích, chưa đo lợi ích tương tác. Mô phỏng feedback nếu dùng chỉ là đánh giá cơ chế, không thay thế bằng chứng hành vi người thật.

Không trì hoãn implementation chỉ để chạy nhiều thử nghiệm lựa chọn hướng. Thiết kế đã được chọn bằng engineering judgment; kết quả đánh giá quyết định mức claim được phép viết.

## 15. Tiêu chí chấp nhận chức năng

- [ ] Không có feedback thì decoder giữ hành vi baseline tương ứng.
- [ ] Xác nhận A chỉ giới hạn A; bác bỏ B không loại cùng khoảng cho A/C.
- [ ] Xác nhận mới thay mốc cũ và hoàn tác khôi phục đúng bộ điều kiện.
- [ ] Có kiểm tra biên khoảng đóng, khoảng một thời điểm và miền không có frame.
- [ ] Mask vẫn đúng khi `event_power != 1`; không làm nhiễu clustering bằng `-inf` đầu vào.
- [ ] Phân biệt mâu thuẫn, thiếu frame và không có path; không kết luận video sai.
- [ ] Không áp event_index cũ sau khi decomposition thay đổi.
- [ ] Giữ canonical IDs/timestamps; không sửa shared score matrix tại chỗ.
- [ ] Phân biệt điều kiện giữ nguyên với frame đại diện thay đổi.
- [ ] Trả về toàn corpus và hoàn tác được; kết quả cũ không ghi đè revision mới.
- [ ] Candidate handoff chỉ mở dialog theo plan hiện có; không tự submit.
- [ ] Hoàn thành chuỗi xác nhận → bác bỏ → giải lại → xem thay đổi → hoàn tác.

Các mục chưa đánh dấu là acceptance criteria cho implementation của người dùng, không phải xác nhận tính năng đã tồn tại.

## 16. Rủi ro và tiêu chuẩn phản biện

| Rủi ro | Cách giữ thiết kế/claim trung thực |
|---|---|
| Chỉ là temporal filter có UI mới | So sánh B–C; chứng minh tác động của tích lũy và tái sử dụng feedback |
| Người dùng mắc kẹt vào mốc sai | Nhánh cục bộ là giả thuyết; giữ toàn corpus, hoàn tác và sửa điều kiện |
| Chi phí thao tác nhiều hơn tự tua | Dùng player + temporal context làm baseline, đo thời gian và completion |
| Planner sai hoặc đổi event | Version binding; thuộc tính đồng thời không ép thành event nối tiếp |
| Điểm retrieval bị diễn giải thành độ chắc chắn | Tách similarity, xác nhận người dùng và phán đoán agent |
| Kết quả benchmark thuận lợi nhưng quá nhỏ | Báo n, counts và giới hạn; không claim significance hoặc tính tổng quát |
| Trùng prior work | Chưa claim first; đối chiếu full text trước khi viết limitation đối thủ |
| Phình scope trước deadline | Một nhánh, full re-decode, một video cache; không RL/k-best/entity-aware |

## 17. Quyết định đã khóa và self-review

Đã khóa: FF-MDP làm nền; feedback event–khoảng; một nhánh cục bộ; giữ toàn corpus; khoảng đóng ms; xác nhận lại thay mốc; negative tích lũy theo event; unknown không tạo negative; full re-decode; decomposition version; không tự nới điều kiện; giữ ranh giới AnswerWorkspace; implementation trên `vbs` do người dùng phụ trách.

Đã tự rà soát đặc tả:

- [x] Nội dung phù hợp phạm vi người dùng đã duyệt, không triển khai code.
- [x] Phân biệt task VBS với TRAKE nội bộ và tránh entity-aware.
- [x] Không nhân bản schema canonical hoặc tự đặt API cạnh tranh với plan frontend.
- [x] Phân biệt semantics, giả thuyết nghiên cứu, bằng chứng benchmark và prior work.
- [x] Không gọi full re-decode là incremental DP hoặc endpoint alternatives là true k-best.
- [x] Ghi rõ lỗi, hành vi hoàn tác, phiên bản và tiêu chí chấp nhận.
- [x] Không claim novelty/hiệu quả đã được chứng minh.

Tài liệu này là đặc tả thiết kế theo Superpowers, không phải kế hoạch coding từng bước. Các vòng duyệt thiết kế trong hội thoại đã hoàn tất; không yêu cầu người dùng xác nhận lại để xuất tài liệu. Việc tạo spec không cho phép assistant tự chuyển sang implementation.
