
# Lộ trình nghiên cứu cho truy hồi đa phương thức theo thời gian trong HCMAI: Từ một hệ thống thi đấu mạnh đến một bài báo SoICT có thể bảo vệ được

## Chẩn đoán tổng quan

Sau khi kiểm tra `src_hcmai_v7.zip` và đối chiếu hệ thống hiện tại với các công trình gần đây về truy hồi video theo thời gian, temporal grounding, suy luận trên video dài, căn chỉnh chuỗi, cũng như các bài báo xuất hiện trực tiếp từ AI Challenge HCMC 2025, kết luận chính của tôi là:

> **Bài báo mạnh nhất của bạn không nên là “một hệ thống truy hồi đa phương thức khác dùng Dynamic Programming”. Nó nên là một bài báo về suy luận thời gian có cấu trúc trên các truy vấn đa sự kiện, trong đó Dynamic Programming chỉ là thuật toán suy luận.**

Sự khác biệt này quan trọng vì không gian novelty quanh “multimodal retrieval + DP” đã khá chật chội. DANTE công khai đề xuất dynamic programming cho TRAKE; Lucifer-TRACE kết hợp temporal search kiểu DP với LVLM verification; MADTempo tổng hợp bằng chứng từ các sự kiện tuần tự; U-CESE đưa ra truy hồi dựa trên clip và caption nhất quán theo thời gian; và một hệ thống AIC 2025 khác cũng đã kết hợp query expansion với truy hồi sự kiện theo thời gian xuyên phương thức. citeturn12academia1turn11academia17turn13academia24turn12academia0turn14search11

Khoảng trống hứa hẹn nhất trong code hiện tại của bạn sâu hơn đáng kể:

> **DP hiện tại bắt buộc thứ tự thời gian, nhưng thực tế không mô hình hóa sự chuyển tiếp ngữ nghĩa giữa các sự kiện liên tiếp.**

Quan trọng hơn, tồn tại một vấn đề toán học trong hàm mục tiêu hiện tại khiến sự khác biệt này trở nên rất cụ thể. Hàm mục tiêu hiện tại của bạn được mô tả là

$$
\mathcal{S}(p_{1:M})
=
\sum_{i=1}^{M} U_i(p_i)
-
\lambda
\sum_{i=2}^{M}
\left(t_{p_i}-t_{p_{i-1}}\right),
$$

dưới ràng buộc thứ tự thời gian nghiêm ngặt

$$
p_1 < p_2 < \cdots < p_M.
$$

Vì timestamp tăng đơn điệu, thành phần thời gian bị **triệt tiêu theo kiểu telescoping**:

$$
\sum_{i=2}^{M}
(t_{p_i}-t_{p_{i-1}})
=
t_{p_M}-t_{p_1}.
$$

Do đó,

$$
\mathcal{S}
=
\sum_i U_i(p_i)
-
\lambda(t_{p_M}-t_{p_1}).
$$

Vì vậy, temporal penalty hiện tại của bạn **không phân biệt nhịp độ bên trong của chuỗi**.

Ví dụ, nếu bỏ qua unary score, hai path sau nhận đúng cùng một temporal penalty:

$$
[0s,1s,100s]
$$

và

$$
[0s,99s,100s].
$$

Cả hai đều trải dài 100 giây.

Điều đó có nghĩa là DP hiện tại chỉ biết:

> sự kiện A phải xảy ra trước B, B phải xảy ra trước C, và toàn bộ chuỗi tốt nhất nên tương đối gọn.

Nó **không** biết:

> A nên chuyển sang B như thế nào; transition đó hợp lý thì mất bao lâu; có phải cùng một người/vật hay không; một vật có thay đổi trạng thái hay không; B có phải hệ quả của A hay không; chuyển động giữa hai anchor có tương thích với query hay không; hoặc có nên bỏ qua một clause của query vì clause đó không thể ground bằng thị giác hay không.

Đây chính là nơi tôi sẽ xây dựng bài báo.

Một điểm chiến lược quan trọng thứ hai là bối cảnh xuất bản. Trang chính thức của AI Challenge HCMC 2026 mô tả bài toán là intelligent multimedia retrieval, nói rõ cả chế độ thi conventional lẫn automated, và cho biết các phương pháp tốt có thể được chọn vào special session về Lifelog và Multimedia Event Retrieval tại SoICT 2026. citeturn11search3 Website submission chung của SoICT hiện liệt kê multimedia intelligence, multimedia information retrieval, event retrieval, multimodal lifelog retrieval và event understanding là các chủ đề phù hợp. citeturn11search2 Hiện có sự không nhất quán giữa các trang chính thức về việc proceedings của special session sẽ theo ACM hay theo general SoICT CCIS, vì vậy tôi sẽ xem route mời qua competition là một luồng hành chính riêng và **không để timeline submission thông thường của SoICT ràng buộc kế hoạch khoa học**, đúng như bạn yêu cầu. citeturn11search3turn11search0

Do đó, khuyến nghị của tôi là định hình dự án quanh:

> **Transition-Aware Multimodal Alignment for Multi-Event Video Retrieval**

với ba ý tưởng liên kết chặt chẽ:

**hiểu query sự kiện có cấu trúc → unary evidence đa phương thức thích ứng → pairwise temporal transition được điều kiện hóa theo sự kiện**, sau đó giải mã bằng một candidate-lattice DP hiệu quả.

Một stage coarse-to-fine để refinement clip nên hỗ trợ phương pháp này. Null/skip state nên giúp hệ thống robust hơn. Có thể thêm VLM verifier, nhưng chỉ nên là module phụ thay vì novelty trung tâm của bài báo.

Cách này tạo ra một bài báo đồng thời:

- hữu ích trực tiếp cho KIS và TRAKE;
- tương thích với kiến trúc hiện tại thay vì đòi hỏi rewrite;
- phân biệt được về mặt khoa học với DANTE;
- dễ bảo vệ hơn nhiều so với “chúng tôi đổi embedding model”;
- có thể phân rã thực nghiệm thành các hypothesis và ablation rõ ràng;
- phù hợp với automated retrieval vì reasoning đắt tiền có thể bị giới hạn trong một candidate lattice nhỏ.

Tôi cũng đã kiểm tra inventory các research plugin khả dụng trong môi trường này. Một connector tìm kiếm khoa học như Consensus có thể được discover nhưng chưa được cài/kết nối trong session, vì vậy tôi không để báo cáo phụ thuộc vào nó. Các claim về literature bên dưới thay vào đó chủ yếu được grounding trên trang chính thức của challenge và các kho xuất bản sơ cấp như arXiv, CVF, ACL Anthology, PMLR, NeurIPS và Springer.

## Hệ thống hiện tại thực sự đang triển khai gì

Codebase được tổ chức tốt hơn đáng kể so với một prototype thi đấu điển hình. Điều này rất có giá trị cho nghiên cứu vì đã có ranh giới khá sạch giữa tạo evidence, temporal decoding, projection theo từng task, offline enrichment và model inference.

Kiến trúc hiện tại có thể tóm tắt là:

$$
\text{query}
\rightarrow
\text{các chuỗi sự kiện}
\rightarrow
\text{điểm frame đa phương thức}
\rightarrow
\text{ma trận điểm theo từng video}
\rightarrow
\text{DP đơn điệu}
\rightarrow
\text{đầu ra KIS/TRAKE}.
$$

### DP hiện tại là một ordered unary model hiệu quả, không phải transition model

`src/hcmai/temporal/dp.py:78–165` thực hiện strict monotonic alignment. Với một ma trận event-by-frame, mỗi sự kiện chọn một frame xuất hiện sau frame của sự kiện trước.

Recurrence được triển khai thông qua phép biến đổi prefix-maximum quanh `dp.py:120–142`. Cách này tránh predecessor search naïve bậc hai và cho phép scan gần tuyến tính theo số frame với mỗi sự kiện:

$$
O(MF)
$$

với \(M\) là số query event và \(F\) là số frame trong một video.

Đây là một engineering baseline rất tốt.

Nhưng về mặt khoa học, model về bản chất chỉ gồm:

$$
\text{độ tương thích ngữ nghĩa unary}
+
\text{tính đơn điệu}
+
\text{độ gọn toàn cục}.
$$

Chính `src/hcmai/temporal/README.md:63–76` của repository đã xác định đúng một số thành phần còn thiếu: entity continuity, state transition, multimodal dense alignment, multi-frame VLM verification và incremental DP.

Kết luận bổ sung quan trọng khi kiểm tra implementation là tính telescoping đã nêu ở trên. Điều này khiến **transition modeling** còn thuyết phục hơn cả những gì README gợi ý.

### Cấu trúc thời gian hiện tại hoàn toàn được quyết định bởi các frame emission độc lập

Mỗi sự kiện \(e_i\) tạo ra một unary score cho mọi frame:

$$
U_i(f)=
\operatorname{sim}(e_i,f).
$$

Không có gì trong score hiện tại biểu diễn:

$$
P(f_j \mid f_i,e_i,e_{i+1}),
$$

hoặc kiểm tra liệu transition

$$
f_i \rightarrow f_j
$$

có hiện thực hóa về mặt ngữ nghĩa

$$
e_i \rightarrow e_{i+1}.
$$

Sự khác biệt này trở nên cực kỳ quan trọng với các query như:

> một người đàn ông đi tới bàn → nhấc một chiếc cốc đỏ lên → uống từ chiếc cốc đó → đặt chính chiếc cốc đó xuống.

Independent unary retrieval có thể tìm ra bốn frame rất tốt chứa:

- một người đàn ông gần chiếc bàn;
- ai đó đang cầm một chiếc cốc đỏ;
- ai đó đang uống;
- một chiếc cốc đỏ trên bàn.

Tuy nhiên các frame này có thể thuộc về những người khác nhau, những chiếc cốc khác nhau, những shot khác nhau, thậm chí những sub-story khác nhau trong một video bản tin dài.

Chỉ strict ordering là không đủ để loại path đó.

Đây là khác biệt khoa học giữa **ordered retrieval** và **temporal event reasoning**.

### Multimodal fusion hiện tại hữu ích nhưng bị cố định toàn cục

Dense temporal scorer trong `src/hcmai/retrieval/evidence/dense.py:42–63` tính ba signal:

$$
S_{\text{visual}},\qquad
S_{\text{context}},\qquad
S_{\text{ASR}},
$$

với SigLIP2 cho visual retrieval và BGE-M3 cho textual evidence theo cấu hình được pin trong `thundercompute/config.yaml`.

Mỗi row score event-by-corpus được min-max normalize độc lập, sau đó ba modality được kết hợp như sau

$$
S =
w_v S_v+
w_c S_c+
w_a S_a.
$$

Cấu hình mặc định trong `src/hcmai/common/config.py:312–328` là

$$
w_v=w_c=w_a=\frac13.
$$

Khi bật BM25, hybrid scorer lại dùng một fixed convex combination, mặc định:

$$
0.5S_{\text{dense}}+0.5S_{\text{BM25}}
$$

trong `common/config.py:342–358`.

Điều này tạo ra một cơ hội thực nghiệm rất tốt.

Xét hai sự kiện sau:

> “Người thuyết trình nói cụm từ ‘artificial intelligence’.”

và

> “Một chiếc xe máy rẽ trái và lách sát qua một chiếc ô tô.”

Rõ ràng mixture evidence tối ưu khó có thể giống nhau. ASR có thể mang tính quyết định cho trường hợp thứ nhất; visual evidence có nhận thức về motion nên chiếm ưu thế ở trường hợp thứ hai.

Fusion hiện tại không thể hiện được khác biệt đó.

Còn một vấn đề calibration khác. Per-event, per-modality min-max normalization phần lớn làm mất **absolute confidence**. Một modality yếu/nhiễu vẫn có thể bị stretch ra toàn dải \([0,1]\), khiến những chênh lệch score nhỏ và vô nghĩa trở nên ngang hàng với dynamic range của một modality thực sự giàu thông tin.

Điều đó gợi ý một research component thứ hai, tương đối ít rủi ro:

> **độ tin cậy modality phụ thuộc query thay vì modality weight cố định toàn cục.**

### FrameContext là multimodal nhưng không temporal

`FrameContext V1` được xây từ

$$
[\text{CAPTION}],
[\text{VISIBLE\_TEXT}],
[\text{OBJECTS}]
$$

với token budget 80/80/40 theo `offline/enrichment/context/config.py` và `serializer.py`.

ASR được chủ ý duy trì tách riêng dưới dạng evidence có timestamp; tài liệu offline ghi rõ rằng ASR bị loại khỏi FrameContext.

Sự phân tách này hợp lý về kiến trúc, nhưng FrameContext vẫn chỉ là một **mô tả cùng-frame**.

Nó không chứa:

- trạng thái frame trước;
- trạng thái frame sau;
- motion;
- quỹ đạo object;
- persistent entity memory;
- context cấp shot;
- thông tin “before/after”.

Literature rộng hơn ngày càng nhấn mạnh đúng các cấu trúc temporal này. Ví dụ LongVALE coi video dài là chuỗi các multimodal event và xây dựng vision-audio-language event với temporal boundary cùng relation-aware description thay vì các frame độc lập. citeturn15search5 VideoStir tương tự biểu diễn video dài thành spatio-temporal clip graph và truy hồi evidence bằng multi-hop structured reasoning thay vì coi video như một tập frame phẳng. citeturn15search6

### Query decomposition hiện mang tính cú pháp hơn là ngữ nghĩa

`src/hcmai/temporal/planner.py` chia text KIS theo cách deterministic dựa trên line hoặc sentence boundary.

Cách này robust và reproducible, nhưng “sentence” không nhất thiết đồng nghĩa với “temporally groundable event”.

Ví dụ:

> “Một phụ nữ đeo kính bước vào phòng, nói chuyện với người đàn ông đang ngồi rồi sau đó đưa cho anh ta một tài liệu trong khi một người khác đứng nhìn.”

Một câu này chứa ít nhất ba temporal anchor hữu ích:

$$
\text{bước vào}
\rightarrow
\text{nói chuyện}
\rightarrow
\text{đưa tài liệu}.
$$

Ngược lại:

> “Cùng người đó, người đã xuất hiện trước đó trong bản tin, vẫn đang nói.”

có thể chỉ encode một event cộng thêm identity constraint thay vì nhiều retrieval event độc lập.

Subsystem Qwen query-preparation của bạn đã có structured output và có thể sinh đúng năm aligned paraphrase bundle (`query_preparation/service.py:64–89`), nhưng workflow KIS/TRAKE chính vẫn chưa biến query gốc thành một **event graph** tường minh.

ED-VTG cho thấy enrich grounding query có thể hữu ích, nhưng nó cũng huấn luyện rõ ràng bằng multiple-instance learning để chọn giữa các query variant và suppress những enrichment hallucinated có hại. citeturn15search0 Đây là một cảnh báo quan trọng: chỉ riêng “LLM query expansion” không nên được xem là tự động có lợi.

### Full-corpus scoring diễn ra trước temporal reasoning

Repository tự ghi nhận rằng hiện chưa có bước shortlist candidate video trước temporal decoder. Mọi frame được chọn từ visual index đều được score cho từng event.

Với 873 video, điều này có thể hoàn toàn chấp nhận được đối với embedding similarity rẻ. Vì vậy tôi **không** khuyến nghị pitch hierarchical retrieval chủ yếu như một đóng góp về scalability.

Mục đích thật sự nên khác:

> global retrieval rẻ nên tạo ra một search lattice nhỏ, nhờ đó ta có thể trả chi phí cho temporal reasoning giàu hơn đáng kể.

Điều này đổi kiến trúc từ

$$
\text{reasoning đắt trên toàn bộ dữ liệu}
$$

sang

$$
\text{cheap recall}
\rightarrow
\text{rich temporal reasoning trên vùng hợp lý}.
$$

Nguyên tắc coarse-to-fine này được hậu thuẫn mạnh bởi các công trình long-video hiện đại. ReVisionLLM trước hết xác định các vùng rộng có liên quan rồi đệ quy thu hẹp tới temporal boundary chính xác; VideoTree cũng xây query-adaptive hierarchical representation rồi progressively refine các vùng video liên quan. citeturn15search1turn15search8

### Code hiện tại đã có nhiều machinery hữu ích cho hướng nghiên cứu đề xuất

Nhiều component sẵn có cho thấy proposal nghiên cứu không đòi hỏi phải vứt bỏ hệ thống hiện tại.

Code của bạn đã có:

- SigLIP2 visual embedding;
- BGE-M3 multimodal textual evidence;
- OCR;
- ASR;
- YOLOE detection;
- Qwen3-VL captioning;
- Qwen query preparation;
- Qwen3-VL reranker;
- hỗ trợ DINO embedding API;
- interface preprocessing shot/event;
- timestamp và frame identity chuẩn hóa;
- một module DP sạch.

Qwen reranker đặc biệt liên quan. `thundercompute/README.md:108–110` ghi đúng rằng nó chỉ có thể reorder các candidate đã retrieve; nó không thể khôi phục candidate bị bỏ sót.

Điều đó tự nhiên dẫn tới kiến trúc mà VLM reasoning được dùng **sau khi xây candidate high-recall**, chứ không phải làm retrieval engine.

KIS readout cũng đáng chú ý. `orchestration/workflows/kis.py:39–43` hiện chọn một event ở vùng upper-middle của aligned path làm representative frame. Cách này deterministic nhưng không phụ thuộc query. Với một chuỗi event bất đối xứng, evidence phân biệt mạnh nhất cho KIS có thể nằm gần đầu hoặc cuối segment suy ra thay vì ở midpoint.

Đây là một vấn đề phụ nhỏ nhưng rất dễ đo.

## Bối cảnh literature và novelty

Literature ngay quanh challenge này phát triển rất nhanh. Điều quan trọng là phải biết những ý tưởng trông hấp dẫn nào đã bị “chiếm chỗ”.

### Chỉ Dynamic Programming thôi không còn đủ

Đối thủ trực tiếp nhất là **DANTE**, được đề xuất trong *Integrated Semantic and Temporal Alignment for Interactive Video Retrieval*. Paper được thúc đẩy trực tiếp bởi TRAKE của AI Challenge HCMC 2025 và đề xuất Dynamic Alignment of Narrative Temporal Events bằng dynamic programming. citeturn12academia1

Vì vậy một paper có claim chính là

> “chúng tôi dùng DP để căn chỉnh nhiều query event theo thứ tự thời gian”

sẽ rất khó được position như novel.

DP của bạn hoàn toàn có thể giữ lại — nó là một inference engine tốt — nhưng **energy/function đang được tối ưu phải mới**.

Còn một cảnh báo mạnh hơn. **Lucifer-TRACE**, hiện đã được liệt kê trong Springer proceedings của SoICT 2025, kết hợp temporal search dựa trên dynamic programming với LVLM semantic verification. citeturn14search11turn14search5

Do đó:

> **DP + Qwen verification cũng không đủ để làm headline contribution.**

Bạn có thể và có lẽ nên dùng VLM verification, nhưng nó nên hỗ trợ thuật toán chính chứ không định nghĩa novelty.

### Query augmentation là một hướng đã đông đúc

QUEST dùng LLM để rewrite query và external image search nhằm xử lý out-of-knowledge query. citeturn12academia1

MADTempo kết hợp multi-event temporal retrieval với external image search làm OOD fallback. citeturn11academia17

RAPID đã xem LLM query correction/enrichment và parallel retrieval là một thành phần lớn của HCMC video retrieval. citeturn12academia3

Một hệ thống challenge 2025 khác cũng kết hợp LLM query expansion với cross-modal temporal event retrieval. citeturn12academia0

Vì vậy:

> hãy làm query parsing và controlled paraphrasing vì chúng giúp hệ thống, nhưng đừng biến “LLM query expansion” thành claim trung tâm của paper.

Một đóng góp query-side thú vị hơn là **structural parsing**:

$$
q
\rightarrow
(E_1,R_{12},E_2,R_{23},\ldots,E_M)
$$

trong đó mỗi event ghi lại entity, action và state, còn mỗi \(R_{i,i+1}\) biểu diễn transition kỳ vọng.

Cấu trúc này nối language understanding trực tiếp vào temporal algorithm của bạn.

### Keyframe extraction và temporal caption memory cũng đã có người làm

U-CESE đề xuất unified clip-based search engine, DAKE keyframe extraction và ReCap — một framework captioning nhất quán theo thời gian. citeturn13academia24

Paper cross-modal temporal retrieval của Vo et al. cũng đề xuất adaptive keyframe selection bằng KDE-GMM thresholding. citeturn12academia0

Vì vậy tôi sẽ không viết một paper có contribution chính là:

> “keyframe tốt hơn”

hoặc:

> “caption có memory.”

Cả hai vẫn là supporting experiment tốt, đặc biệt vì FrameContext hiện tại của bạn độc lập về thời gian, nhưng chúng không còn là vùng novelty sạch trong cộng đồng nghiên cứu cụ thể này.

### Query đa sự kiện có liên hệ mạnh với nghiên cứu sequence alignment

Chuyển dịch khái niệm hữu ích nhất là xem bài toán của bạn như hỗn hợp của:

$$
\text{cross-modal retrieval}
+
\text{sequence alignment}
+
\text{temporal grounding}.
$$

DTW cổ điển đã mô hình hóa monotonic sequence alignment. Soft-DTW làm objective DTW khả vi để alignment có thể trở thành learning objective thay vì chỉ là inference operation. citeturn10search0

Liên quan trực tiếp hơn tới failure case của bạn là **Drop-DTW**, cho phép bỏ một số element trong khi align phần tín hiệu chung giữa các sequence nhiễu. Nó được đánh giá cụ thể trên temporal step localization và cross-modal retrieval/localization. citeturn10search1

Điều này trực tiếp gợi ý việc cho phép

$$
z_i=\varnothing
$$

với các query event:

- trừu tượng;
- dư thừa;
- bị split sai;
- không có trong keyframe khả dụng;
- paraphrase tệ;
- nghe được nhưng không nhìn thấy;
- mơ hồ về thị giác.

DP hiện tại buộc mọi event phải chọn một frame. Điều đó tạo ra bài toán garbage-in-path kinh điển: chỉ một event tệ có thể kéo toàn bộ alignment về sai region.

StepFormer cũng rất đáng tham khảo. Nó dùng order-aware supervision để khám phá và localize các procedural step đồng thời lọc những phrase không liên quan, và cho thấy zero-shot multi-step localization. citeturn10search2

Một lần nữa, bài học là multi-event query không nên tự động suy ra rằng

$$
\text{mỗi text fragment bắt buộc phải ánh xạ đúng một frame}.
$$

### Các công trình grounding hiện đại ủng hộ mạnh clip-level temporal representation

Hệ thống hiện tại chủ yếu reasoning trên keyframe, nhưng action về bản chất là temporal.

Một image thường có thể nói rằng:

> một người đang cầm cốc.

Nhưng một short clip tốt hơn nhiều để nhận ra:

> người đó nhấc chiếc cốc lên.

Tương tự:

> chiếc ô tô ở cạnh xe máy

là static, trong khi

> ô tô vượt xe máy

là temporal.

Các grounding method gần đây ngày càng xử lý rõ điều này. Sparse-Dense Side-Tuner cho temporal-grounding performance mạnh bằng InternVideo2 feature trong khi vẫn parameter-efficient. citeturn15search13 LongVALE lập luận cho fine-grained multimodal event understanding trên vision, speech/audio và temporal boundary. citeturn15search5 TemporalVLM nhắm đến dense captioning, temporal grounding, highlight detection và action segmentation dưới một long-video temporal representation thống nhất. citeturn15search14

Với hệ thống của bạn, điều này gợi ý:

> giữ image/keyframe embedding cho global retrieval; chỉ đưa true clip/motion representation vào quanh candidate temporal region.

Cách này lấy được phần lớn lợi ích mà không biến toàn bộ corpus 873 video thành một Video-LLM workload đắt tiền.

### Temporal direction nên trở thành một stress test tường minh

ArrowGEV đặc biệt thú vị về mặt khái niệm vì nó phân biệt event có semantics thay đổi khi đảo ngược thời gian với event không nhạy với phép đảo thời gian, và huấn luyện explicit temporal direction awareness. citeturn15search15

Điều này gợi ý một trong những evaluation sạch nhất mà paper của bạn có thể đưa ra.

Với query:

$$
A\rightarrow B\rightarrow C,
$$

hãy tạo hard negative chứa:

$$
C\rightarrow B\rightarrow A
$$

hoặc

$$
A\rightarrow C\rightarrow B.
$$

Một hệ thống frame-bag retrieval có thể score các video này cao vì mọi object/action đều xuất hiện.

Một temporal method thật sự thì không nên như vậy.

Đây có thể trở thành experiment rất thuyết phục vì nó kiểm tra trực tiếp liệu method đang học/reason về chronology hay chỉ được lợi từ semantic retrieval tốt hơn.

### Bản đồ novelty cho dự án

Tôi sẽ đánh giá các hướng candidate như sau.

| Hướng                                    | Novelty khoa học trong landscape challenge này |                                  Tác động kỳ vọng | Chi phí engineering | Khuyến nghị                                           |
| ------------------------------------------ | -----------------------------------------------: | -----------------------------------------------------: | -------------------: | ------------------------------------------------------- |
| Event-conditioned pairwise transition DP   |                               **Rất cao** | **Rất cao** trên TRAKE, có khả năng cả KIS |          Trung bình | **Contribution chính**                           |
| Robust alignment với null/skip event      |                                              Cao |                                  Cao với query nhiễu |   Thấp–trung bình | **Tích hợp vào method chính**                 |
| Query-conditioned modality gating          |                                 Trung bình–cao |                                          Cao và rộng |   Thấp–trung bình | **Contribution phụ mạnh**                       |
| Coarse-to-fine keyframe → clip refinement |          Cao nếu gắn với transition reasoning |                                  Cao cho action/motion |          Trung bình | **Component phụ mạnh**                          |
| Entity/state continuity                    |                               **Rất cao** |                                   Cao trên TRAKE khó |     Trung bình–cao | **Extension rất tốt / phiên bản mạnh nhất** |
| VLM path verification                      |                                      Trung bình |                                              Vừa–cao |          Trung bình | Dùng, nhưng không làm headline                      |
| Query expansion                            |                               Thấp–trung bình |                                     Thường hữu ích |                Thấp | Chỉ engineering                                        |
| Keyframe extraction tốt hơn              |                                      Trung bình |                                           Có thể cao |          Trung bình | Supporting experiment                                   |
| Temporally consistent captioning           |                                      Trung bình |                               Cao cho semantic context |     Trung bình–cao | Supporting experiment                                   |
| “Dùng embedding model lớn hơn”        |                                            Thấp |                                              Chưa rõ |   Thấp–trung bình | Chỉ baseline/ablation                                  |
| Thay DP bằng end-to-end Video-LLM         |                                          Rủi ro |                                              Chưa rõ |             Rất cao | Không khuyến nghị cho paper này                     |

Do đó research question có expected return cao nhất là:

> **Retrieval có cải thiện hay không khi score của một multi-event video path không chỉ phụ thuộc vào việc mỗi frame có khớp từng event hay không, mà còn phụ thuộc vào việc các candidate clip liên tiếp có hiện thực hóa đúng entity, state, motion và temporal transition được mô tả giữa các event hay không?**

Đó là một paper.

## Các hướng nghiên cứu giá trị nhất

### Structured temporal alignment có nhận thức transition

Đây nên là centerpiece.

Thay vì biểu diễn query thành các chuỗi độc lập

$$
E_1,E_2,\ldots,E_M,
$$

hãy xây các structured event node:

$$
E_i =
(
\text{entities},
\text{action},
\text{objects},
\text{attributes},
\text{state-before},
\text{state-after},
\text{modality cues}
)
$$

và transition edge:

$$
R_i =
R(E_{i-1},E_i).
$$

Ví dụ:

> “Một phụ nữ bước vào bếp, lấy một chai từ tủ lạnh, rót nước vào cốc, rồi rời đi cùng chiếc cốc.”

có thể trở thành:

$$
E_1: \text{người phụ nữ bước vào bếp}
$$

$$
R_2: \text{cùng người phụ nữ; cùng địa điểm; transition ngắn}
$$

$$
E_2: \text{người phụ nữ lấy chai từ tủ lạnh}
$$

$$
R_3: \text{người phụ nữ và chai được duy trì; trạng thái chai thay đổi}
$$

$$
E_3: \text{người phụ nữ rót nước vào cốc}
$$

$$
R_4: \text{người phụ nữ và cốc được duy trì; bắt đầu di chuyển}
$$

$$
E_4: \text{người phụ nữ rời đi khi đang cầm cốc}.
$$

Bây giờ định nghĩa path objective là

$$
\boxed{
\mathcal{J}(z_{1:M})
=
\sum_{i=1}^{M} U_i(z_i)
+
\sum_{i=2}^{M} T_i(z_{i-1},z_i)
-
\rho\sum_i\mathbf{1}[z_i=\varnothing]
}
$$

trong đó \(z_i\) có thể là candidate frame/clip hoặc null state.

Khác biệt then chốt là:

$$
T_i(a,b)
\neq
-\lambda(t_b-t_a).
$$

Thay vào đó:

$$
T_i(a,b)
=
\alpha D_i(\Delta t)
+
\beta C_i(a,b)
+
\gamma R_i(a,b)
+
\delta M_i(a,b).
$$

Trong đó:

$$
D_i(\Delta t)
$$

là **event-conditioned duration potential**;

$$
C_i
$$

đo entity continuity;

$$
R_i
$$

đo semantic state/transition consistency;

$$
M_i
$$

đo motion compatibility.

Điều này thay đổi hoàn toàn cách diễn giải DP.

DP không còn là contribution.

Nó chỉ là exact/approximate decoder cho structured temporal model của bạn.

Về học thuật, framing này mạnh hơn nhiều.

### Temporal distance phụ thuộc từng event transition

Đây là dạng transition modeling đơn giản nhất và nên được implement trước.

Hệ thống hiện tại giả định mọi adjacent event pair đều nhận cùng một linear gap preference.

Thay vào đó hãy học hoặc dự đoán

$$
P(\Delta t\mid E_{i-1},E_i).
$$

Khi đó:

$$
D_i(\Delta t)
=
\log P(\Delta t\mid E_{i-1},E_i).
$$

Một implementation ban đầu đơn giản có thể dùng category:

| Loại transition      | Ví dụ                                        | Temporal prior kỳ vọng |
| --------------------- | ---------------------------------------------- | ------------------------ |
| atomic continuation   | “nâng cốc → uống”                        | rất ngắn               |
| same-scene sequential | “mở tủ lạnh → lấy chai”                 | ngắn                    |
| extended action       | “bắt đầu nấu → dọn món”               | trung bình              |
| narrative transition  | “phỏng vấn → cảnh ngoài trời sau đó” | rộng                    |
| unknown               | event tổng quát                              | yếu/gần uniform        |

Một implementation mạnh hơn dự đoán parameter từ event-pair embedding:

$$
[\mu_i,\sigma_i]
=
g_\theta(E_{i-1},E_i),
$$

rồi dùng potential dạng log-normal:

$$
D_i(\Delta t)
=
-
\frac{
(\log(1+\Delta t)-\mu_i)^2
}{
2\sigma_i^2
}.
$$

Bạn không nhất thiết cần large-scale training. Một held-out query set nhỏ hoặc pseudo-labeling có thể đủ để kiểm tra hypothesis.

Điểm mấu chốt là cách này lập tức loại vấn đề telescoping vì

$$
D_i
$$

khác nhau cho từng transition và là phi tuyến theo \(\Delta t\).

Ngay cả trước entity tracking hay VLM reasoning, đây đã là một temporal model mới có ý nghĩa.

### Alignment có nhận thức entity continuity

Đây có thể là extension ấn tượng nhất.

Nhiều query TRAKE khó mang implicit constraint:

> **cùng một entity tham gia xuyên suốt chuỗi event.**

Frame score hiện tại chủ yếu chỉ hỏi liệu từng event có được thể hiện đâu đó trong frame hay không.

Giả sử:

$$
E_1=\text{người đàn ông cầm ô}
$$

và

$$
E_2=\text{người đàn ông bước vào ô tô}.
$$

Hệ thống có thể align người A ở \(E_1\) với người B ở \(E_2\).

Một transition-aware model có thể định nghĩa:

$$
C_i(a,b)
=
\max_{x\in \mathcal E(a), y\in \mathcal E(b)}
\operatorname{sim}
(\phi(x),\phi(y)),
$$

trong đó \(\mathcal E(a)\) và \(\mathcal E(b)\) là các entity được detect, còn \(\phi\) là appearance embedding.

Bạn đã có sẵn các mảnh ghép quan trọng:

- YOLOE detection;
- hỗ trợ DINO embedding;
- frame timestamp;
- infrastructure raw-video/keyframe.

Phiên bản thực dụng đầu tiên không cần sophisticated multi-object tracking trên toàn corpus.

Chỉ dùng entity continuity **bên trong các candidate video đã shortlist**:

$$
\text{retrieval}
\rightarrow
\text{top candidate interval}
\rightarrow
\text{detect/crop candidate entity}
\rightarrow
\text{appearance continuity score}.
$$

Cách này dễ chịu hơn nhiều về compute.

Với object tổng quát, nên thử DINO-like crop embedding.

Với người, hãy bắt đầu từ clothing/body appearance thay vì biến face recognition thành dependency.

Claim của paper không nên là “chúng tôi track người.” Nên là:

> **persistent-entity evidence được đưa vào dưới dạng pairwise potential bên trong multi-event temporal alignment.**

Cách này tổng quát hơn.

### Retrieval có nhận thức state transition

Entity continuity thôi vẫn không phân biệt được:

> người cầm chiếc ô đang đóng

với:

> người cầm chiếc ô đang mở.

Hoặc:

> cốc rỗng

với:

> cốc đầy.

Hoặc:

> người đang tiến tới ô tô

với:

> người đang ở trong ô tô.

Đó là các **state transition**, và chính ở đây independent-frame embedding bắt đầu yếu.

Một transition score có thể hoạt động trên một pair hoặc một short sequence candidate observation:

$$
R_i(a,b)
=
\operatorname{score}
(
\text{visual transition }a\rightarrow b,
\text{text relation }E_{i-1}\rightarrow E_i
).
$$

Bạn có nhiều mức implementation.

Rẻ nhất là structured textual comparison bằng caption và object state.

Mạnh hơn là short local clip encoder.

Mạnh nhất nhưng đắt nhất là dùng Qwen3-VL trên một bounded sequence frame như:

$$
\{a-\tau,a,a+\tau,b-\tau,b,b+\tau\}
$$

và yêu cầu constrained score thay vì free-form reasoning.

Prompt không nên hỏi:

> “Video này có khớp toàn bộ query không?”

Thay vào đó hãy hỏi các atomic question như:

> “Các frame A–C có cho thấy cùng một người chuyển từ trạng thái cầm chai sang rót từ chai đó không? Trả về score từ 0 đến 1.”

Cách này tạo ra pairwise feature được kiểm soát tốt hơn nhiều.

Một lần nữa, chỉ làm điều này trên candidate lattice nhỏ.

### Local clip refinement có nhận thức motion

Keyframe representation hiện tại vốn yếu với các động từ như:

- tiến lại gần;
- vượt;
- rẽ;
- ngã;
- mở;
- đóng;
- ném;
- nhận;
- trao đổi;
- đi vào;
- rời đi.

Một static frame có thể chứa object evidence rất tốt nhưng vẫn mơ hồ về action.

Thay vì thay thế pipeline indexing hiện tại, hãy dùng nó làm Stage A.

Stage A:

$$
\text{keyframe multimodal retrieval}
\rightarrow
\text{high recall}.
$$

Stage B:

$$
\text{candidate timestamp}
\rightarrow
[t-\tau,t+\tau]
$$

lấy từ video gốc.

Stage C:

$$
\text{clip encoder}
\rightarrow
\text{score có nhận thức action/motion}.
$$

Các công trình temporal-grounding hiện đại cung cấp lý do mạnh để dùng temporal representation chuyên biệt thay vì phụ thuộc hoàn toàn vào frozen image embedding. Ví dụ, Sparse-Dense Side-Tuner đạt grounding result mạnh với InternVideo2 trong khi dùng parameter-efficient adaptation. citeturn15search13 ReVisionLLM và VideoTree tiếp tục ủng hộ cách xử lý coarse-to-fine cho video dài thay vì đồng đều xử lý mọi frame với chi phí cao. citeturn15search1turn15search8

Với bài toán của bạn, tôi sẽ thử ba representation:

$$
\text{một keyframe},
$$

$$
\text{mean/attention pooling các keyframe lân cận},
$$

và

$$
\text{true short-video embedding}.
$$

Chỉ experiment này thôi đã cho biết bao nhiêu phần error còn lại đến từ representation so với alignment.

### Query-conditioned multimodal gating

Tôi xem đây là **secondary contribution chi phí thấp tốt nhất**.

Thay

$$
S_i
=
\frac13S_i^V+
\frac13S_i^C+
\frac13S_i^A
$$

bằng

$$
S_i
=
\sum_{m}
w_{i,m}S_{i,m}
$$

trong đó

$$
\mathbf w_i
=
\operatorname{softmax}
g(E_i,\mathbf r_i).
$$

\(\mathbf r_i\) có thể encode availability/reliability của evidence:

- event chứa lời nói được quote;
- event chứa chữ/số có thể đọc;
- event phụ thuộc nhiều vào motion;
- event phụ thuộc nhiều vào object/scene;
- phân phối ASR candidate score phẳng;
- không có OCR evidence;
- caption confidence yếu;
- modality retrieval entropy cao.

Ví dụ:

> “Màn hình hiển thị 2026.”

nên upweight OCR.

> “Một phóng viên nói rằng lạm phát đã tăng.”

nên upweight ASR.

> “Cầu thủ bóng đá đá quả bóng.”

nên ưu tiên video.

Phiên bản đầu tiên thậm chí có thể dùng rule-based gate suy ra từ structured query parsing. Learned gate có thể làm sau.

Một confidence statistic đặc biệt hữu ích là retrieval entropy.

Với modality \(m\):

$$
p_{m,j}
=
\frac{\exp(s_{m,j}/\tau)}
{\sum_k \exp(s_{m,k}/\tau)},
$$

và

$$
H_m
=
-\sum_jp_{m,j}\log p_{m,j}.
$$

Một modality có phân phối sharply peaked có khả năng mang evidence hữu ích; modality gần uniform có thể ít phân biệt.

Bạn có thể condition contribution của modality theo reliability này thay vì mù quáng stretch mọi modality bằng row-wise min-max normalization.

### Null-event alignment

Tôi đặc biệt khuyến nghị thêm thành phần này ngay cả khi nó chỉ chiếm một subsection trong paper.

Decoding hiện tại yêu cầu

$$
z_i\neq\varnothing
\quad\forall i.
$$

Hãy thay đổi state space để cho phép

$$
z_i=\varnothing
$$

với penalty \(\rho_i\).

Khi đó một event không liên quan hoặc ground kém có thể được skip thay vì làm nhiễm độc toàn bộ sequence.

Drop-DTW là precedent mạnh cho nguyên lý chung của sequence alignment có outlier dropping. citeturn10search1 Việc StepFormer order-aware filter các text không liên quan trong multi-step localization cũng củng cố cùng motivation trong video. citeturn10search2

Điều này đặc biệt hữu ích vì event segmentation của bạn có thể đến từ:

- sentence splitting;
- LLM parsing;
- TRAKE subquery do user cung cấp.

Không cách nào trong số đó đảm bảo rằng mọi clause đều tương ứng với một visible frame duy nhất.

Một extension gọn là **event visibility estimate**:

$$
\rho_i=
h(E_i),
$$

để một clause như

> “người dẫn chuyện giải thích tại sao điều này xảy ra”

có penalty thấp hơn nếu bị visually skip so với

> “một chiếc xe tải đỏ đi vào giao lộ.”

### Selective VLM verification, không phải universal VLM reranking

VLM verification đáng dùng vì code của bạn đã hỗ trợ, nhưng hãy xem nó là **final stage được trigger bởi confidence**.

Ví dụ, chỉ gọi VLM khi

$$
\mathcal{J}(P_1)-\mathcal{J}(P_2)<\epsilon
$$

hoặc khi best path có entity/transition confidence kém.

Khi đó:

$$
\text{cheap retrieval}
\rightarrow
\text{structured DP}
\rightarrow
\begin{cases}
\text{return}, & \text{confident}\\
\text{VLM verify}, & \text{uncertain}.
\end{cases}
$$

Điều này khiến computational cost dễ báo cáo và tránh framing paper như một hệ “DP + LVLM” khác, một không gian đã được Lucifer-TRACE đại diện. citeturn14search11

Nó cũng cho phép bạn tạo một empirical statement hữu ích:

> selective VLM reasoning mang lại phần lớn verification gain chỉ với \(X\%\) số VLM call so với uniform reranking.

Đây là systems contribution tốt hơn nhiều.

## Thesis và method khuyến nghị cho paper

Tôi sẽ cấu trúc paper thực tế quanh một method trung tâm, tạm đặt tên:

> **Transition-Aware Multimodal Alignment for Multi-Event Video Retrieval**

Tôi sẽ chưa khóa acronym cho tới khi kiểm tra literature cuối cùng để tránh naming collision.

### Problem formulation

Cho corpus

$$
\mathcal V=\{V_1,\ldots,V_N\}.
$$

Một query mô tả chuỗi event có thứ tự thời gian

$$
Q=(E_1,\ldots,E_M).
$$

Target không chỉ là video \(V^*\), mà là một ordered path

$$
P^*=
(z_1^*,\ldots,z_M^*)
$$

sao cho

$$
t(z_1^*)<\cdots<t(z_M^*)
$$

và joint path giải thích được query.

Với KIS, path còn sinh ra inferred segment

$$
[\hat s,\hat t]
$$

và representative submission frame.

Với TRAKE, expose toàn bộ path.

Điều này ngay lập tức thống nhất hai task về mặt khoa học:

$$
\boxed{
\text{KIS và TRAKE cùng chia sẻ latent temporal path inference;}
\text{chúng chủ yếu khác nhau ở readout.}
}
$$

Đây là một framing rất gọn cho paper.

### Structured event parser

Query parser xuất ra:

$$
\mathcal G_Q=
(\mathcal E,\mathcal R),
$$

trong đó event \(E_i\) chứa:

$$
E_i=
(a_i,o_i,x_i,l_i,s_i,c_i)
$$

cho action, object/entity, attribute, location, state và modality cue.

Các edge chứa:

$$
R_i=
(\text{continuity},
\text{state change},
\text{temporal relation},
\text{duration class}).
$$

Parser phải giữ nguyên original text làm fallback.

Điều này quan trọng: **không bao giờ để LLM rewrite trở thành representation duy nhất**.

Hãy lưu:

$$
E_i^{original},
E_i^{literal},
E_i^{structured},
E_i^{paraphrases}.
$$

`QueryCandidateSet` hiện tại của bạn về mặt kiến trúc đã khá gần với việc hỗ trợ cấu trúc này.

### Candidate generation

Dùng high-recall multimodal retriever hiện tại.

Với mỗi event:

$$
\mathcal C_i=
\operatorname{TopK}
\{
U_i(f)
\}.
$$

Sau đó xây video coverage score như:

$$
A(V)
=
\sum_{i=1}^{M}
\operatorname{LSE}_{f\in V\cap\mathcal C_i}
U_i(f),
$$

hoặc một top-event-score aggregation đơn giản hơn.

Tiêu chí chính nên là **event coverage**, không chỉ single best frame.

Một video đúng có candidate khá tốt cho cả bốn event nên được ưu tiên hơn một video chỉ có một event cực tốt nhưng không có evidence cho các event còn lại.

Giữ top \(B\) video.

Trong mỗi shortlisted video, thêm temporal neighbor quanh retrieved anchor.

Cách này tạo một candidate lattice nhỏ hơn nhiều.

### Adaptive unary evidence

Với event \(i\) và candidate \(z\):

$$
U_i(z)
=
w_{i,V}S_{i,V}(z)
+
w_{i,C}S_{i,C}(z)
+
w_{i,A}S_{i,A}(z)
+
w_{i,O}S_{i,O}(z)
+
w_{i,B}S_{i,B}(z)
+
w_{i,M}S_{i,M}(z).
$$

Ở đây \(M\) có thể đại diện cho motion/clip evidence.

Weight phụ thuộc query và tùy chọn phụ thuộc confidence:

$$
\mathbf w_i
=
\operatorname{softmax}
g_\theta
(
\phi(E_i),
r_{i,V},\ldots,r_{i,M}
).
$$

Phiên bản paper đơn giản nhất có thể dùng một MLP nhỏ thay vì model trainable lớn.

### Transition-aware pairwise potential

Contribution chính là:

$$
T_i(a,b)=
D_i(a,b)
+
C_i(a,b)
+
R_i(a,b).
$$

Một decomposition thực dụng là

$$
T_i(a,b)
=
\lambda_dD_i(\Delta t)
+
\lambda_eC_i(a,b)
+
\lambda_rR_i(a,b).
$$

Full score trở thành:

$$
\boxed{
\mathcal{J}(P)=
\sum_i U_i(z_i)
+
\sum_{i=2}^{M}T_i(z_{i-1},z_i)
-
\sum_i\rho_i\mathbf1[z_i=\varnothing].
}
$$

Inference vẫn monotonic:

$$
t(z_i)>t(z_{i-1})
$$

trừ khi một state là null.

Recurrence về mặt khái niệm là:

$$
DP[i,b]
=
U_i(b)+
\max_{a<t_b}
\left[
DP[i-1,a]+T_i(a,b)
\right].
$$

Khác recurrence linear-gap hiện tại, arbitrary pairwise transition không còn dùng được prefix-max trick đơn giản.

Nhưng điều đó không phải vấn đề nếu DP giàu hơn này chạy trên shortlisted lattice.

Với \(K\) candidate cho mỗi event:

$$
O(MK^2)
$$

thường hoàn toàn hợp lý.

Ta có thể giới hạn predecessor thêm bằng temporal window \(W\):

$$
O(MKW).
$$

Đây là architectural trade quan trọng:

> **full-frame DP hiện tại cực rẻ vì temporal model đơn giản; DP đề xuất giàu hơn nên retrieval trước tiên phải giảm state space.**

Đó là một algorithmic story mạch lạc.

### Local clip refinement

Với mỗi candidate anchor \(z\), tạo local interval:

$$
I(z)=[t_z-\tau_1,t_z+\tau_2].
$$

Chỉ encode bằng video backbone sau coarse retrieval.

Sau đó thu được:

$$
S_{motion}(E_i,z)
=
\cos(
\phi_{text}(E_i),
\phi_{video}(I(z))
).
$$

Với pairwise transition, tùy chọn encode:

$$
I(a,b)
$$

hoặc các bounded sample trải qua transition.

Một experiment quan trọng là xác định lượng temporal context cần thiết:

$$
\tau\in
\{0,1s,2s,4s,8s\}.
$$

Điều này tạo ra plot hữu ích:

$$
\text{accuracy vs temporal window vs latency}.
$$

Đây là kiểu result reviewer dễ nhớ.

### KIS readout

Không nên tự động dùng upper-middle aligned event.

Sau khi chọn path, suy ra

$$
[\hat s,\hat t]
=
[t(z_1),t(z_M)]
$$

hoặc dùng start/end margin phụ thuộc event.

Sau đó chọn representative KIS frame theo một criterion tường minh:

$$
f^*
=
\arg\max_{f\in[\hat s,\hat t]}
\left[
\sum_i\alpha_iU_i(f)
+
\eta\,\text{distinctiveness}(f)
\right].
$$

Một baseline đơn giản hơn là frame gắn với event phân biệt mạnh nhất:

$$
i^*
=
\arg\max_i
\left(
U_i(z_i)-U_i^{(2nd)}
\right),
$$

rồi submit \(z_{i^*}\).

Hãy test cách này so với upper-middle policy hiện tại.

Nó có thể là một KIS improvement dễ hơn dự kiến.

### TRAKE readout

TRAKE tự nhiên trả về:

$$
(z_1,\ldots,z_M).
$$

Nhưng hệ thống cũng nên giữ:

- unary score cho mỗi event;
- transition score cho mỗi edge;
- modality weight;
- null state;
- confidence.

Điều này tạo ra qualitative visualization rất tốt cho paper.

Thay vì chỉ show:

> “kết quả của chúng tôi đúng,”

hãy show tại sao:

$$
\begin{array}{c|c|c|c}
\text{Event} & \text{Frame} & U_i & T_i\\
\hline
E_1 & 01{:}22 & .81 & -\\
E_2 & 01{:}27 & .74 & .88\\
E_3 & 01{:}31 & .79 & .91
\end{array}
$$

và so với baseline DP chọn frame semantic mạnh hơn nhưng transition-inconsistent.

### Các hypothesis của paper

Tôi sẽ freeze research quanh một tập hypothesis nhỏ, có thể falsify.

**H1 — Transition hypothesis**

$$
\text{Unary + event-conditioned pairwise transition}
>
\text{Unary + linear-gap DP}
$$

đặc biệt với query chứa action sequence và state change.

**H2 — Modality hypothesis**

$$
\text{query-conditioned fusion}
>
\text{fixed global weight}
$$

đặc biệt khi chỉ một modality mang decisive evidence.

**H3 — Motion hypothesis**

$$
\text{local clip refinement}
>
\text{keyframe-only retrieval}
$$

đặc biệt với các động từ nhạy thời gian.

**H4 — Robust alignment hypothesis**

$$
\text{null-aware DP}
>
\text{mandatory-event DP}
$$

trong điều kiện syntactic over-segmentation, query expansion noise và non-visual clause.

**H5 — Temporal-reasoning hypothesis**

Relative improvement của hệ đề xuất phải lớn hơn trên **wrong-order hard negative** so với ordinary negative.

Hypothesis cuối đặc biệt giá trị vì nó test claimed mechanism chứ không chỉ leaderboard accuracy.

### Những gì nên được xem là core contribution

Tôi sẽ giữ contribution statement thật kỷ luật.

Paper nên nói gần như sau:

> Trước hết, chúng tôi xác định một limitation của multi-event DP với linear gap truyền thống: dưới monotonic alignment, cumulative linear gap penalty co lại thành prior trên tổng segment span và không thể biểu diễn transition đặc thù cho từng event.

> Chúng tôi đưa ra một structured temporal retrieval formulation kết hợp multimodal unary evidence với event-conditioned pairwise transition potential và optional null-event state.

> Chúng tôi triển khai formulation này trong một coarse-to-fine candidate lattice, cho phép motion/entity/state reasoning mà không cần expensive video processing toàn cục.

> Chúng tôi đánh giá không chỉ standard retrieval accuracy mà cả order-reversal, entity-consistency và modality-corruption stress test để xác định improvement có thật sự đến từ temporal reasoning hay không.

Đây là research contribution dễ bảo vệ hơn nhiều so với việc đưa vào năm competition trick rời rạc.

## Chương trình thực nghiệm và ablation

Experimental design sẽ quyết định đây là một paper hay chỉ là một hệ thống được cải thiện.

### Xây frozen research benchmark trước khi tune method mới

Repository đã đúng khi lưu ý rằng không thể chứng minh HCMAI accuracy improvement nếu không có frozen evaluation set.

Đây nên là scientific deliverable đầu tiên.

Với mỗi research query, lưu:

$$
Q,\quad
V^*,\quad
[s^*,t^*],
$$

và lý tưởng là event-level grounding:

$$
(E_i,[s_i^*,t_i^*]).
$$

Với query kiểu TRAKE, event-level annotation cực kỳ giá trị.

Một minimum viable research set nên ở mức vài trăm event instance được kiểm soát cẩn thận thay vì hàng nghìn weak label. Kích thước chính xác nên tùy annotation capacity; chất lượng quan trọng hơn việc giả vờ rằng competition corpus tự cung cấp supervision trong khi thực tế không có.

Với bất kỳ learned calibration/gating component nào, hãy split **theo video**, không theo frame, để tránh các frame liên quan chặt chẽ trong cùng broadcast bị leak giữa train và test.

Không liên tục sửa test annotation trong lúc develop.

Tạo:

$$
D_{train},D_{val},D_{test}
$$

một lần, hash manifest và report hash/run identifier.

### Tách video retrieval khỏi temporal alignment

Một sai lầm evaluation lớn là chỉ report final Recall@K.

Khi đó không thể biết paper cải thiện semantic retrieval hay temporal reasoning.

Hãy report ít nhất ba layer.

#### Video retrieval

Đo:

$$
R@1,\ R@5,\ R@10,\ R@20
$$

và MRR.

Metric này trả lời:

> Method có tìm được đúng video không?

#### Event grounding

Với aligned timestamp \(\hat t_i\), định nghĩa:

$$
Hit_\delta(i)=
\mathbf1[
d(\hat t_i,[s_i,t_i])\leq \delta
].
$$

Report nhiều tolerance như:

$$
\delta\in\{1s,2s,5s\},
$$

điều chỉnh theo keyframe density thực tế.

Cũng report median/mean temporal error.

#### Path-level success

Điều này đặc biệt quan trọng với TRAKE.

Định nghĩa:

$$
AllHit@\delta
=
\frac1{|Q|}
\sum_q
\mathbf1[
\forall i,\ Hit_\delta(q,i)=1
].
$$

Một query bốn event có ba event đúng và một event sai không nên được xem tương đương với bốn single-event success độc lập.

Cũng report average per-event hit rate để metric không quá khắc nghiệt.

### Đánh giá interval đúng như interval

Vì query user mô tả một short temporal segment \([s,t]\), hãy report temporal IoU khi có interval ground truth:

$$
tIoU
=
\frac{
|[\hat s,\hat t]\cap[s,t]|
}{
|[\hat s,\hat t]\cup[s,t]|
}.
$$

Metric này đặc biệt hữu ích cho KIS vì nó tách:

> đúng video nhưng localization kém

khỏi

> đúng video và temporal localization chính xác.

### Tạo temporal hard-negative subset

Đây là nơi paper đề xuất có thể mạnh hơn hẳn một challenge report.

Hãy xây các subset mà ordinary semantics cố ý không đủ.

**Wrong-order case**

Cả hai video đều chứa A, B, C, nhưng:

$$
V^+:A\rightarrow B\rightarrow C
$$

trong khi

$$
V^-:A\rightarrow C\rightarrow B.
$$

Việc ArrowGEV nhấn mạnh time-sensitive so với time-insensitive event ủng hộ tầm quan trọng của việc test temporal direction một cách tường minh thay vì chỉ generic semantic similarity. citeturn15search15

**Entity-switch case**

Đúng:

$$
\text{person A thực hiện }E_1,E_2,E_3.
$$

Hard negative:

$$
A:E_1,\quad B:E_2,\quad A:E_3.
$$

**State-transition case**

Đúng:

$$
\text{đóng}\rightarrow\text{đang mở}\rightarrow\text{mở}.
$$

Hard negative chứa đủ ba state nhưng sai sequence hoặc thuộc các object khác nhau.

**Motion/static-confusion case**

Ví dụ:

$$
\text{đứng gần ô tô}
\quad\text{vs}\quad
\text{đi vào ô tô},
$$

$$
\text{cầm bóng}
\quad\text{vs}\quad
\text{ném bóng}.
$$

**Modality-conflict case**

Visual evidence biểu thị một event trong khi ASR/OCR chứa lexical distractor.

Các subset này cho phép bạn nói:

> gain tập trung cụ thể ở nơi temporal reasoning là cần thiết.

Điều đó mạnh hơn đáng kể so với:

> overall Recall@5 của chúng tôi tăng 3%.

### Test query corruption một cách tường minh

Tạo controlled perturbation:

$$
Q \rightarrow Q_{\text{split-noise}}
$$

bằng cách thêm event không cần thiết;

$$
Q \rightarrow Q_{\text{missing}}
$$

bằng cách bỏ một event;

$$
Q \rightarrow Q_{\text{paraphrase}}
$$

bằng cách dùng cách diễn đạt khác;

$$
Q \rightarrow Q_{\text{abstract}}
$$

với một non-visible clause.

Sau đó so mandatory alignment với null-aware alignment.

Đây là nơi robustness theo cảm hứng Drop-DTW nên thể hiện rõ. citeturn10search1

### Bảng ablation thiết yếu

Paper nên có bảng cấu trúc như sau:

| Model                   | Adaptive modality | Transition duration | Clip motion | Entity continuity | Null event | VLM verify | TRAKE AllHit | KIS R@1 |
| ----------------------- | ----------------- | ------------------- | ----------- | ----------------- | ---------- | ---------- | -----------: | ------: |
| Baseline hiện tại     | ✗                | linear              | ✗          | ✗                | ✗         | ✗         |           … |      … |
| + calibrated unary      | ✓                | linear              | ✗          | ✗                | ✗         | ✗         |           … |      … |
| + event duration        | ✓                | ✓                  | ✗          | ✗                | ✗         | ✗         |           … |      … |
| + motion transition     | ✓                | ✓                  | ✓          | ✗                | ✗         | ✗         |           … |      … |
| + null state            | ✓                | ✓                  | ✓          | ✗                | ✓         | ✗         |           … |      … |
| + entity continuity     | ✓                | ✓                  | ✓          | ✓                | ✓         | ✗         |           … |      … |
| Full + selective verify | ✓                | ✓                  | ✓          | ✓                | ✓         | ✓         |           … |      … |

Đừng bỏ hàng đầu tiên.

**Code hiện tại của bạn là baseline quan trọng nhất** vì reviewer cần thấy rằng chính temporal transition modeling — chứ không đơn giản là một hệ thống hoàn toàn khác — tạo ra gain.

Cũng hãy thêm:

$$
\lambda_{gap}=0
$$

như một baseline.

Điều này sẽ cho thấy linear gap term hiện tại thực sự đóng góp bao nhiêu.

### Test trực tiếp telescoping hypothesis

Đây nên là một experiment riêng vì nó hỗ trợ theoretical motivation.

Hãy dựng synthetic score matrix trong đó unary score được giữ cố định nhưng intermediate pacing khác nhau.

Ví dụ với ba event:

$$
P_A=(0,2,10)
$$

và

$$
P_B=(0,8,10).
$$

Cho hai path cùng unary score.

Objective hiện tại của bạn bắt buộc phải gán cho chúng cùng temporal cost.

Sau đó dựng một query trong đó expected transition là:

$$
E_1\rightarrow E_2
$$

xảy ra nhanh, rồi có một khoảng delay dài trước \(E_3\).

Event-conditioned transition model đề xuất nên ưu tiên \(P_A\).

Một proposition lý thuyết nhỏ cộng với controlled experiment này có thể là một trong những phần sạch nhất của paper.

### So sánh các event segmentation strategy

Đánh giá:

$$
\text{sentence split},
$$

$$
\text{LLM event split},
$$

và

$$
\text{LLM structured event graph}.
$$

Nhưng đừng biến phần này thành một prompt-engineering exercise khổng lồ.

Đo:

- số event trung bình;
- event-level grounding recall;
- downstream path accuracy;
- null-event usage;
- latency.

Structured parser chỉ hữu ích nếu nó cải thiện retrieval.

### So sánh fixed fusion và adaptive fusion một cách đúng đắn

Tối thiểu hãy đánh giá:

$$
V,
C,
A,
V+C,
V+A,
C+A,
V+C+A,
$$

sau đó:

$$
\text{fixed optimized weights}
$$

so với

$$
\text{query-conditioned weights}.
$$

Sự phân biệt này quan trọng. Nếu không, reviewer hoàn toàn có thể lập luận rằng adaptive gate chỉ thắng vì default weight \(1/3,1/3,1/3\) vốn chưa tối ưu.

Trước tiên tune best global fixed weight trên validation.

Sau đó hãy đánh bại **nó**, không chỉ default.

Tương tự, so sánh:

$$
\text{row min-max}
$$

với các calibration approach khác trước khi claim chính gate là nguyên nhân của gain.

### Chỉ so keyframe với clip ở nơi thực sự có ý nghĩa

Đừng chỉ report overall clip-model gain.

Chia query thành:

$$
Q_{static}
$$

và

$$
Q_{motion}.
$$

Nếu hypothesis đúng, ta kỳ vọng:

$$
\Delta_{motion}
\gg
\Delta_{static}.
$$

Result dựa trên mechanism như vậy thuyết phục hơn nhiều.

### Efficiency phải là first-class result

Framing challenge 2026 có automated intelligent retrieval, khiến runtime ngày càng quan trọng. citeturn11search3

Report:

$$
P50,\quad P95
$$

cho:

- query parsing;
- global retrieval;
- candidate construction;
- DP;
- clip refinement;
- VLM verification;
- total request.

Cũng report:

- số candidate video;
- số candidate cho mỗi event;
- số/tỷ lệ query gọi VLM verification;
- GPU memory;
- số expensive model call.

Vẽ:

$$
\text{accuracy}
\quad\text{vs}\quad
K
$$

cho candidate lattice size.

Điều này xác lập liệu pairwise reasoning có thực sự hữu dụng trong thực tế hay chỉ hấp dẫn về mặt lý thuyết.

### Dùng một public benchmark cho một external-validity experiment

Main benchmark vẫn nên bám challenge vì đó là bài toán thật của paper.

Nhưng một external experiment sẽ tăng độ mạnh đáng kể.

Các family gần nhất là multi-sentence/paragraph temporal grounding và multi-step localization, không phải generic text-video retrieval. Video Paragraph Grounding nghiên cứu trực tiếp việc localize nhiều sentence mà semantic relationship và temporal order giữa chúng có ý nghĩa, về cấu trúc gần TRAKE hơn nhiều so với single-moment retrieval. StepFormer cũng cung cấp một setting multi-step localization hữu ích. citeturn10search2

Bạn không cần chạy đua SOTA trên mọi public benchmark.

Claim mạnh hơn là:

> cùng transition-aware decoder đó cải thiện một temporal-grounding backbone chuẩn trên external ordered-event dataset.

Điều này chứng minh algorithm không chỉ là một HCMAI heuristic thủ công.

## Positioning paper, rủi ro và câu hỏi mở

### Paper mà tôi sẽ viết

Một title ngắn gọn có thể là:

> **Transition-Aware Multimodal Alignment for Multi-Event Video Retrieval**

Abstract nên bắt đầu từ failure của retrieval hiện tại:

> Multi-event video query không chỉ mô tả những gì xuất hiện tại từng thời điểm riêng lẻ, mà còn mô tả cách entity, action và state tiến triển theo thời gian.

Sau đó xác định problem:

> Existing frame-retrieval pipeline theo sau bởi monotonic dynamic programming chủ yếu kết hợp independent event-frame compatibility với chronological constraint.

Sau đó là theoretical observation:

> Với linear temporal-gap objective thường dùng dưới strict monotonic alignment, pairwise gap cost co lại thành penalty trên tổng path span và vì vậy không thể biểu diễn temporal transition đặc thù cho từng event.

Sau đó là method:

> Chúng tôi giới thiệu một transition-aware multimodal alignment objective kết hợp adaptive unary evidence, event-conditioned pairwise temporal potential và optional null-event state, được decode hiệu quả trên coarse-to-fine candidate lattice.

Sau đó là evaluation:

> Trên HCMAI multi-event retrieval và các targeted stress set về wrong-order/entity/state, formulation đề xuất cải thiện cả video retrieval lẫn complete-path localization, đặc biệt với các query đòi hỏi temporal reasoning thật sự.

Dĩ nhiên, final performance claim phải chờ experiment.

### Paper không nên được position như một system paper chứa đầy module không liên quan

Tránh contribution list kiểu:

> Chúng tôi dùng SigLIP2, BGE-M3, Qwen, BM25, ASR, OCR, YOLOE, DP, DINO, reranking, query expansion, v.v.

Đó là mô tả software architecture, không phải scientific novelty.

Thay vào đó:

$$
\boxed{\text{một problem}
\rightarrow
\text{một central hypothesis}
\rightarrow
\text{một structured model}
\rightarrow
\text{controlled experiment}}
$$

Các system component phục vụ hypothesis đó.

### Những hướng tôi sẽ chủ động tránh làm main direction

Tôi sẽ **không** mở đầu bằng “improved dynamic programming.” DANTE đã chiếm không gian đó. citeturn12academia1

Tôi sẽ **không** mở đầu bằng “DP + LVLM verification.” Lucifer-TRACE đã có positioning đó. citeturn14search11

Tôi sẽ **không** mở đầu bằng “query augmentation + web image search.” QUEST, MADTempo và RAPID đã có contribution rất gần. citeturn12academia1turn11academia17turn12academia3

Tôi sẽ **không** mở đầu bằng “temporally consistent captioning.” U-CESE đã đưa ra ReCap. citeturn13academia24

Tôi sẽ **không** mở đầu bằng “adaptive keyframe extraction.” Cả U-CESE lẫn một hệ HCMC 2025 khác đã có keyframe-selection contribution. citeturn13academia24turn12academia0

Tôi cũng sẽ **không** ngay lập tức thay toàn bộ architecture bằng end-to-end Video-LLM. Literature gần đây cho thấy Video-LLM grounding model rất mạnh, nhưng cũng cho thấy cần effort kiến trúc đáng kể để long-video temporal reasoning khả thi — hierarchical search, specialized temporal expert, temporal training data, token reduction và dedicated temporal representation liên tục xuất hiện như giải pháp. citeturn15search1turn15search2turn15search3turn15search4 Vì vậy retrieval-first architecture của bạn không lỗi thời; nó là nền tảng mạnh cho structured search.

### Một research sequence thực tế

Thứ tự implementation có giá trị cao nhất là:

$$
\boxed{
\text{freeze benchmark}
\rightarrow
\text{phân tích error hiện tại}
\rightarrow
\text{candidate lattice}
\rightarrow
\text{event-conditioned gap}
\rightarrow
\text{null state}
\rightarrow
\text{clip motion}
\rightarrow
\text{adaptive fusion}
\rightarrow
\text{entity/state transition}
\rightarrow
\text{selective VLM verification}
}
$$

Điểm mấu chốt là bạn **không nên implement mọi thứ trước khi đo bất kỳ thứ gì**.

Experiment quyết định đầu tiên rất nhỏ:

$$
\text{DP hiện tại}
\quad\text{vs}\quad
\text{event-conditioned transition DP}.
$$

Nếu cách mới không cải thiện path-level accuracy trên temporally difficult query, hãy điều tra nguyên nhân trước khi thêm component khác.

Experiment quyết định tiếp theo là:

$$
\text{keyframe unary}
\quad\text{vs}\quad
\text{local clip unary}.
$$

Điều này cho biết motion representation có phải bottleneck chính hay không.

Sau đó:

$$
\text{fixed fusion}
\quad\text{vs}\quad
\text{adaptive fusion}.
$$

Chỉ sau đó mới nên thêm entity/state continuity và VLM verification.

### Qualitative figure giá trị nhất

Một figure nên hiển thị một query và hai candidate path.

Ví dụ:

**Query**

> Một người đàn ông nhấc một chiếc hộp lên → mang nó về phía xe tải → đặt chính chiếc hộp đó vào bên trong xe tải.

**DP hiện tại**

$$
f_{21}:
\text{người đàn ông A đang cầm hộp}
$$

$$
\downarrow
$$

$$
f_{48}:
\text{người đàn ông B đi cạnh xe tải}
$$

$$
\downarrow
$$

$$
f_{61}:
\text{chiếc hộp ở trong xe tải}.
$$

Unary score cao, chronological order đúng, **event chain sai**.

**Đề xuất**

$$
f_{24}
\rightarrow
clip_{31}
\rightarrow
f_{43}
$$

với:

$$
\text{same-person consistency}=0.91,
$$

$$
\text{box continuity}=0.87,
$$

$$
\text{transition score}=0.89.
$$

Chỉ một figure này truyền đạt toàn bộ paper hiệu quả hơn một architecture diagram đầy model.

### Theoretical statement mạnh nhất có thể rút ra từ code hiện tại

Tôi thực sự cân nhắc formalize proposition sau.

**Mệnh đề.** Với timestamp tăng nghiêm ngặt, một constant linear adjacent-gap penalty

$$
G(P)
=
-\lambda
\sum_{i=2}^{M}
(t_i-t_{i-1})
$$

tương đương với

$$
G(P)
=
-\lambda(t_M-t_1).
$$

Do đó, khi cố định first và last selected timestamp, gap term bất biến đối với mọi intermediate event timestamp.

**Hệ quả.**

Objective không thể ưu tiên một internal temporal arrangement này hơn arrangement khác dựa trên event-specific pacing; mọi discrimination giữa các path như vậy đến từ unary event-frame score và các constraint khác.

Quan sát này rất sơ cấp về toán học, và thực ra đó là một điểm mạnh: reviewer dễ verify, nó áp dụng trực tiếp vào baseline hiện tại, và nó motivate pairwise model đề xuất một cách chính xác.

Đừng overclaim rằng đây là một theorem mới.

Research novelty là:

> nhận ra hệ quả thực tiễn của nó đối với multi-event video retrieval và thay degenerate temporal prior bằng semantically conditioned transition potential.

### Scientific success nên trông như thế nào

Kết quả lý tưởng không chỉ là:

$$
R@1:+X.
$$

Mà là một pattern như:

$$
\Delta_{\text{ordinary queries}}
=
+2.1
$$

nhưng

$$
\Delta_{\text{wrong-order}}
=
+11.8,
$$

$$
\Delta_{\text{state-change}}
=
+13.4,
$$

$$
\Delta_{\text{entity-continuity}}
=
+9.7.
$$

Các con số này chỉ mang tính minh họa, không phải dự đoán.

Một pattern như vậy sẽ chứng minh method mới cải thiện đúng class query mà theory dự đoán.

Tương tự, với multimodal gating, một result thuyết phục sẽ là:

$$
\Delta_{\text{speech queries}}
\gg 0
$$

khi adaptive gating tăng ASR weight,

trong khi:

$$
\Delta_{\text{motion queries}}
\gg 0
$$

khi nó tăng video/clip evidence.

Mechanistic evaluation thuyết phục hơn nhiều so với tối ưu aggregate leaderboard.

### Các câu hỏi mở và limitation hiện tại

ZIP chứa research/runtime implementation nhưng không chứa một frozen benchmark hoàn chỉnh với query-level ground truth và các bảng experimental result hiện có. Do đó tôi chưa thể xác định thực nghiệm liệu dominant failure mode hiện tại là candidate recall, temporal alignment, keyframe sparsity, modality fusion hay event parsing. Vì vậy roadmap ở trên dựa trên architectural analysis cộng với literature hiện tại, không dựa trên measured error frequency từ test set của chính bạn.

Corpus size **873 video** do user cung cấp được xem là con số authoritative cho project này; tôi không tìm thấy public source chính thức hiện tại xác nhận chính xác con số corpus đó, và các trang public của challenge mô tả broader multimedia-retrieval task hơn là dataset snapshot cụ thể trong repository của bạn. citeturn11search3

Cơ chế publication chính xác của special session cũng cần được xử lý cẩn thận. Site AI Challenge HCMC 2026 nói rằng các method được chọn có thể được mời vào SoICT 2026 special session và cho biết proceedings của session đó được ACM xuất bản, trong khi website general SoICT 2026 hiện mô tả Springer CCIS proceedings và format main paper 12 trang. citeturn11search3turn11search2 Sự không nhất quán hành chính này không ảnh hưởng khuyến nghị khoa học, nhưng manuscript template cuối cùng nên theo instruction được gửi riêng qua competition route thay vì assumption từ normal SoICT track.

Cuối cùng, literature challenge gần đây khiến novelty boundary khá rõ. DANTE đã xác lập DP cho TRAKE; Lucifer-TRACE đã xác lập DP + LVLM verification; U-CESE đã xác lập clip-based retrieval + temporally consistent caption; MADTempo/QUEST/RAPID đã chiếm vùng query augmentation/OOD retrieval. citeturn12academia1turn14search11turn13academia24turn11academia17turn12academia3

Điều này để lại vùng nghiên cứu dễ bảo vệ nhất của bạn là:

$$
\boxed{
\begin{aligned}
&\textbf{independent event-frame matching}\\
&\qquad\Downarrow\\
&\textbf{structured event-transition retrieval}
\end{aligned}
}
$$

hay cụ thể hơn,

$$
\boxed{
\text{Điều gì xảy ra ở từng thời điểm}
+
\text{một thời điểm biến đổi thành thời điểm kế tiếp như thế nào}
}
$$

thay vì chỉ

$$
\boxed{
\text{điều gì xảy ra ở từng thời điểm}
+
\text{timestamp bắt buộc phải tăng}.
}
$$

Đây là bước chuyển khái niệm mà tôi sẽ dùng làm trọng tâm để xây bài SoICT.
