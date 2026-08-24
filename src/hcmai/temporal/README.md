# Temporal evidence và temporal alignment

hcmai.temporal là lớp dùng chung để xử lý **bằng chứng có thông tin thời gian**
trong hai task KIS và TRAKE. Folder này không tạo caption/OCR/ASR, không build
embedding/index và không thực hiện task-specific answer generation. Nó nhận kết
quả retrieval đã có, bảo toàn canonical frame identity, sau đó localization hoặc
alignment theo semantics của từng task.

Hai bài toán được giữ tách biệt:

~~~text
KIS:   sparse evidence -> bounded temporal scenes
TRAKE: dense event/frame scores -> ordered monotonic paths
~~~

## 1. Kiến trúc tổng quát

TemporalEvidenceCore là composition facade chính:

~~~text
TemporalQueryPlan
       |
       v
  alignment_mode
       |
       +-- progressive_scene
       |      |
       |      +--> ProgressiveEvidenceProvider
       |      +--> ProgressiveSceneAligner
       |      +--> SceneCandidate[]       (KIS)
       |
       +-- ordered_path
              |
              +--> DenseOrderedEvidenceProvider
              +--> MonotonicOrderedPathAligner
              +--> OrderedPathCandidate[] (TRAKE)
~~~

Các module chính:

| Module | Trách nhiệm |
| --- | --- |
| [core.py](./core.py) | Điều phối plan, state, provider và aligner. |
| [state.py](./state.py) | Lưu progressive state, TTL, lock và compare-and-swap commit. |
| [query.py](./query.py) | Normalize cumulative snapshot và lấy delta hint an toàn. |
| [evidence.py](./evidence.py) | Đổi retrieval candidate thành canonical FrameEvidence; quản lý ba trạng thái đánh giá. |
| [providers/sparse.py](./providers/sparse.py) | Global/local retrieval, backfill và candidate-video scoring cho KIS. |
| [providers/dense.py](./providers/dense.py) | Lấy dense event/frame score matrix cho TRAKE. |
| [aligners/scene.py](./aligners/scene.py) | Cluster frame thành scene và score scene. |
| [aligners/monotonic.py](./aligners/monotonic.py) | Adapter gọi monotonic DP ổn định của TRAKE và materialize frame canonical. |
| [relations.py](./relations.py) | Parse temporal relation rõ ràng và chấm relation trên timestamp. |
| [scoring.py](./scoring.py) | Normalize score, tính scene-score component và deterministic ranking. |

TemporalQueryPlan kiểm tra alignment mode theo task:

~~~text
KIS   -> progressive_scene
TRAKE -> ordered_path
~~~

## 2. Các contract cốt lõi

### 2.1 QueryUnit

Một query được biểu diễn thành các semantic unit có ID và thứ tự ổn định:

~~~text
Progressive KIS:
  h0 = "một người đứng cạnh xe"
  h1 = "sau đó người này bước vào xe"

TRAKE:
  e0 = "một người đứng cạnh xe"
  e1 = "người này bước vào xe"
~~~

Reveal order của progressive hint chỉ là thứ tự nhập vào. Nó **không tự động**
ngụ ý rằng các event xảy ra theo cùng thứ tự trong video. Chỉ các cụm từ quan
hệ rõ ràng mới tạo temporal constraint.

### 2.2 FrameEvidence

Mỗi evidence giữ:

~~~text
FrameRecord canonical
unit_scores
source_scores
source_ranks
score
provenance
~~~

Retrieval candidate được resolve qua DataService.get_frame(frame_id). Temporal
kiểm tra lại frame_id, video_id, frame_idx và timestamp_ms. Metadata do retriever
trả về mà mâu thuẫn với canonical store sẽ bị reject.

Score được chọn theo thứ tự:

~~~text
candidate.final_score
  -> candidate.reranker_score
  -> candidate.fusion_score
  -> max(source_scores)
~~~

### 2.3 Output contract

SceneCandidate là một đoạn thời gian có nhiều frame evidence, dùng cho KIS.
OrderedPathCandidate là một path cùng video, có đúng một frame cho mỗi
QueryUnit, dùng cho TRAKE.

~~~text
SceneCandidate
  video_id
  start_ms / end_ms
  evidence[]
  semantic/coverage/temporal/relation/final score

OrderedPathCandidate
  video_id
  frames[]                 # một frame cho mỗi event
  query_unit_ids[]
  score
~~~

## 3. Progressive workflow cho KIS

Luồng đầy đủ:

~~~text
request(snapshot, search_id, filters)
  -> load hoặc tạo ProgressiveSearchState
  -> normalize và diff cumulative snapshot
  -> tạo QueryUnit mới từ delta
  -> parse explicit temporal constraints
  -> global retrieval cho hint mới
  -> local retrieval trong candidate videos cũ
  -> canonicalize thành FrameEvidence
  -> backfill video rescued hoặc còn UNKNOWN
  -> score video và prune candidate pool
  -> cluster evidence thành bounded scenes
  -> score/rank scenes
  -> commit state nếu toàn bộ bước thành công
~~~

### 3.1 Snapshot và state transaction

Frontend có thể gửi cumulative snapshot:

~~~text
Q1 = H1
Q2 = H1 + H2
Q3 = H1 + H2 + H3
~~~

diff_snapshot() normalize Unicode NFC, khoảng trắng và punctuation:

~~~text
FIRST        snapshot đầu tiên; delta là toàn bộ snapshot
APPEND       snapshot mới nối thêm; delta là phần suffix
NO_CHANGE    khác biệt chỉ là format/punctuation; không chạy retrieval
REPLACEMENT  query bị viết lại; reject để bảo vệ state đã commit
~~~

Progressive state được clone trước khi xử lý. Chỉ khi provider và aligner thành
công thì clone mới được commit:

~~~text
current committed state
          |
          +--> clone/proposed state --> retrieval --> alignment --> commit
          |
          +--> nếu lỗi: giữ nguyên current state
~~~

Mỗi session cố định:

- task_type;
- base_filters;
- search_id và version.

State store có per-search lock, compare-and-swap version, TTL và giới hạn số
entry. Request có search_id hết hạn hoặc không tồn tại sẽ báo lỗi thay vì âm
thầm bắt đầu search mới.

### 3.2 Global/local retrieval

Với QueryUnit mới h_k, provider thực hiện hai nhánh:

~~~text
global:
  search(h_k, base_filters, global_quota)
  -> tìm video mới trên toàn corpus

local:
  search(h_k, base_filters ∩ previous_video_ids, local_quota)
  -> kiểm tra candidate videos từ các hint trước
~~~

RetrievalService có thể dùng visual, context, caption, OCR, ASR hoặc fusion
tùy index/config. Temporal chỉ nhận RetrievalResult và giữ nguyên source
provenance.

Kết quả hai nhánh được merge theo canonical frame_id, sau đó mới áp dụng
Top-M. Vì vậy một frame xuất hiện ở cả global và local không chiếm hai vị trí
evidence.

### 3.3 Ba trạng thái đánh giá

Với một cặp (query_unit, video), state phân biệt:

~~~text
UNKNOWN
  Không có trong pooled retrieval; chưa đủ thông tin để kết luận.

EVALUATED_NO_MATCH
  Đã dedicated-search đúng video nhưng không giữ được evidence.

MATCHED
  Đã evaluate và có FrameEvidence.
~~~

Implementation dùng hai container:

~~~python
evaluated_keys: set[(unit_id, video_id)]
evidence[(unit_id, video_id)]: tuple[FrameEvidence, ...]
~~~

Do đó video không xuất hiện trong pooled Top-K **không bị biến thành score 0**.
Chỉ dedicated single-video search mới được phép biến UNKNOWN thành
EVALUATED_NO_MATCH.

### 3.4 Rescued-video backfill

Ví dụ:

~~~text
Hint 1 -> video A, B
Hint 2 -> video C xuất hiện lần đầu
~~~

Video C là rescued video. Trước khi so sánh C với A/B, provider tìm lại các
hint cũ trong riêng video C:

~~~text
temporary candidate union
  -> phát hiện rescued/unknown pairs
  -> dedicated search từng video và từng unit còn thiếu
  -> mark evaluated hoặc giữ evidence
  -> candidate scoring
  -> prune
~~~

Thứ tự backfill -> score -> prune là bắt buộc. Prune trước backfill có thể
loại một video chỉ vì những hint lịch sử của nó chưa được evaluate.

Backfill được giới hạn bởi:

~~~text
backfill_max_videos
backfill_max_units_per_video
~~~

Nếu giới hạn bị chạm, phần còn lại có thể tiếp tục là UNKNOWN; đó là trạng
thái chưa đủ thông tin, không phải negative evidence.

### 3.5 Candidate-video scoring

Raw retrieval score được normalize riêng cho từng query unit trong active
candidate set. Với score s và range [l_u, h_u]:

$$
\hat{s}_{u,v} =
\begin{cases}
1, & h_u \le l_u, \\\
\operatorname{clip}\left(\dfrac{s_{u,v}-l_u}{h_u-l_u},0,1\right), & h_u > l_u.
\end{cases}
$$

Với video v:

$$
semantic_v =
\operatorname{mean}_{u \in matched(v)}
\left(\max_f \hat{s}_{u,v,f}\right)
$$

Hai loại coverage được giữ riêng:

$$
matchCoverage_v =
\dfrac{matched\ evaluated\ units}{evaluated\ units}
$$

$$
evaluationCoverage_v =
\dfrac{evaluated\ units}{total\ query\ units}
$$

Candidate score:

$$
S_v =
\dfrac{
w_s\,semantic_v
+ w_m\,matchCoverage_v\,evaluationCoverage_v
+ w_e\,evaluationCoverage_v
}{w_s+w_m+w_e}
$$

Default weights:

~~~text
w_s = 0.45
w_m = 0.25
w_e = 0.30
~~~

Sau khi rank theo S_v, hệ thống giữ candidate_pool_size video và xoá evidence
nằm ngoài active pool. Việc này vừa giới hạn bộ nhớ vừa đảm bảo scene aligner
chỉ làm việc trên candidate pool đã được xác định rõ.

## 4. Scene assembly và scene scoring

### 4.1 Canonical scene clustering

Evidence của cùng một video được:

~~~text
deduplicate theo frame_id
  -> merge unit_scores/source provenance nếu cần
  -> sort theo timestamp_ms, frame_id
  -> cluster theo gap và span
~~~

Frame mới ở timestamp t_i nhập cluster hiện tại nếu:

$$
t_i - t_{i-1} \le scene\_max\_gap\_ms
$$

và:

$$
t_i - t_{clusterStart} \le scene\_max\_span\_ms
$$

scene_max_gap_ms ngăn nối hai evidence quá xa. scene_max_span_ms ngăn
chaining tạo scene quá dài dù từng cặp frame liên tiếp vẫn gần nhau.

Default config:

~~~text
scene_max_gap_ms  = 5_000   # 5 giây
scene_max_span_ms = 30_000  # 30 giây
~~~

SceneCandidate luôn phải có evidence, cùng video_id, và timestamp của mọi
evidence phải nằm trong [start_ms, end_ms].

### 4.2 Scene score components

Với mỗi query unit, scene lấy evidence score cao nhất sau normalize:

$$
semantic(scene) =
\operatorname{mean}_{u}
\left(\max_{f \in scene}\hat{s}_{u,f}\right)
$$

Match coverage và evaluation coverage:

$$
effectiveCoverage(scene) =
coverage(scene) \times evaluationCoverage(scene)
$$

Temporal coherence của scene có span d:

$$
temporal(scene) =
\dfrac{1}{1+d/scene\_coherence\_ms}
$$

Relation score chỉ được thêm khi có constraint và constraint có đủ evidence để
đánh giá. Nếu relation là UNKNOWN, component này bị bỏ khỏi weighted sum;
trọng số các component đang active được normalize lại.

Final scene score:

$$
SceneScore =
\dfrac{\sum_{c \in active} w_c S_c}
     {\sum_{c \in active} w_c}
$$

Default weights:

~~~text
semantic = 0.45
coverage = 0.30
temporal = 0.15
relation = 0.10
~~~

Scene được rank theo final_score, sau đó dùng các component còn lại và
canonical identity để tie-break. Mỗi video chỉ giữ scene_top_b_per_video, rồi
mới lấy scene_top_p_global trên toàn bộ candidate pool.

### 4.3 Temporal relations

Parser hiện chỉ tạo relation cho các pattern rõ ràng:

~~~text
"sau đó", "rồi", "then"
  -> previous unit BEFORE current unit

"đồng thời", "cùng lúc", "trong lúc",
"simultaneously", "at the same time"
  -> OVERLAP

"cuối cùng", "ở cuối cảnh", "finally", "at the end"
  -> AT_END
~~~

Các pattern mơ hồ như “ngay trước”, “ngay sau”, “sau khi” chưa tự động tạo
constraint. Constraint sai có thể làm hỏng ranking nên parser ưu tiên bỏ qua
hơn là đoán.

Với BEFORE, relation được xem là thỏa nếu tồn tại một cặp timestamp hợp lệ:

$$
\exists\ t_a \in T_A,\ t_b \in T_B:
t_a \le t_b
$$

Với OVERLAP hoặc NEAR:

$$
\left|t_a-t_b\right| \le near\_ms
$$

Relation là soft scoring component, không phải hard filter. Một scene vi phạm
relation vẫn có thể được trả về nhưng bị giảm điểm tương ứng.

## 5. KIS workflow

KIS gọi TemporalEvidenceCore.localize() với query snapshot:

~~~text
SearchRequest
  -> progressive localization
  -> ranked SceneCandidate[]
  -> representative-frame selection
  -> optional bounded reranking
  -> canonical materialization
  -> SearchResponse
~~~

Mỗi scene sinh tối đa một representative frame. Selector hiện dùng:

~~~text
evidence score cao nhất
  -> gần midpoint của scene nhất
  -> frame_idx nhỏ hơn nếu hòa
~~~

Reranker chỉ được reorder bounded candidate pool. Nó không được tạo frame mới,
đổi frame_id hoặc viết lại canonical frame metadata.

## 6. TRAKE workflow

TRAKE không dùng progressive snapshot và không dùng search_id.

~~~text
TRAKERequest(events = E1..En)
  -> TemporalEvidenceCore.ordered_plan()
  -> DenseOrderedEvidenceProvider
  -> event/frame dense score matrices
  -> video shortlisting
  -> monotonic dynamic programming
  -> OrderedPathCandidate[]
  -> level-wise cross-video ranking
  -> TRAKEResponse
~~~

### 7.1 Ordered plan

ordered_plan(events) tạo e0, e1, ..., en và các adjacent BEFORE constraints:

~~~text
e0 BEFORE e1
e1 BEFORE e2
...
~~~

Plan được đánh dấu:

~~~text
task_type      = TRAKE
alignment_mode = ORDERED_PATH
~~~

### 7.2 Dense event/frame evidence

Dense provider gọi visual retrieval để:

1. Encode toàn bộ event text.
2. Retrieve Top-K frame cho mỗi event.
3. Cho mỗi event, mỗi video chỉ vote một lần.
4. Rank video theo event coverage, RRF vote và video_id.
5. Giữ max_videos video.
6. Rescore toàn bộ frame của các video được shortlist.

RRF vote có dạng:

$$
vote(v) =
\sum_e \dfrac{1}{rrf\_k + rank_{e,v}}
$$

Kết quả cho mỗi video là ma trận:

$$
S_v \in \mathbb{R}^{N_{events} \times N_{frames}}
$$

Trong đó S_v[e, j] là similarity của event e với frame position j.
Các mảng frame_ids, frame_idx, timestamps_ms dùng cùng một column order.

### 7.3 Monotonic dynamic programming

Mục tiêu chọn một frame position cho từng event:

$$
p_1 < p_2 < \dots < p_n
$$

và tối đa hóa:

$$
\max_{p_1 < \dots < p_n}
\left[
\sum_{e=1}^{n} S_v[e,p_e]
- \lambda_{gap}\sum_{e=2}^{n}
\left(t_{p_e}-t_{p_{e-1}}\right)
\right]
$$

Nếu dùng event_power < 1, similarity không âm được biến đổi trước alignment:

$$
S'_v =
\operatorname{clip}(S_v,0,\infty)^{event\_power}
$$

DP recurrence được viết lại để dùng prefix maximum:

$$
DP_e(j) =
S_v[e,j] - \lambda t_j
+ \max_{i<j}\left(DP_{e-1}(i)+\lambda t_i\right)
$$

Nhờ prefix maximum, một DP layer chạy tuyến tính theo số frame:

~~~text
naive:  thử mọi i < j cho từng j
       gần O(events × frames²)

current: prefix maximum + backpointer
         O(events × frames)
~~~

Backpointer khôi phục frame path. Adapter sau đó resolve từng frame_id qua
DataService và reject nếu có bất kỳ conflict nào về video, frame index hoặc
timestamp.

Nếu cluster_delta > 0, các frame có score vector gần nhau được gom cluster và
event liên tiếp phải chọn cluster khác nhau. Điều này giảm việc nhiều event cùng
trỏ tới các frame gần như giống hệt nhau.

### 7.4 Path diversification

Một video có thể tạo nhiều path. rank_paths() lấy path theo level:

~~~text
level 1: path tốt nhất của mỗi video
level 2: path thứ hai của mỗi video
level 3: path thứ ba của mỗi video
~~~

Sau mỗi level, các path được sort theo score và ghép vào output cho đến khi đạt
max_rows. Cách này ưu tiên video diversity trước khi lặp lại cùng video.

## 8. Canonical identity và invariant

Temporal phải bảo toàn các trường:

~~~text
video_id
frame_id
frame_idx
timestamp_ms
~~~

Các nguyên tắc bắt buộc:

- frame_id là identity nội bộ canonical, không được thay bằng array position;
- frame_idx là tọa độ competition-facing, không phải keyframe order;
- reranker chỉ được reorder, không được rewrite identity;
- scene evidence phải cùng video_id và nằm trong time range;
- ordered path phải có đúng một frame cho mỗi query unit;
- mọi frame trong ordered path phải cùng video và tăng dần theo timestamp;
- canonical dedup chỉ dedup theo frame_id;
- evidence source/provenance vẫn được giữ sau khi merge.

## 9. Default budgets

Các giá trị dưới đây là default config, không phải scientific truth; có thể thay
đổi cho từng experiment:

| Budget | Default |
| --- | ---: |
| progressive state TTL | 1800 giây |
| progressive state max entries | 256 |
| maximum hints | 10 |
| candidate pool | 50 videos |
| global retrieval quota | 100 |
| local retrieval quota | 50 |
| Top-M evidence | 5 frame / unit / video |
| backfill videos | 10 |
| backfill units / video | 5 |
| top scenes / video | 3 |
| top scenes globally | 100 |
| scene gap | 5000 ms |
| scene span | 30000 ms |
| scene coherence | 15000 ms |

Các budget này giới hạn chi phí online và kích thước state. Nếu thay đổi budget,
nên ghi lại config cùng benchmark vì nó ảnh hưởng trực tiếp tới recall và
latency.

## 10. Failure behavior và giới hạn hiện tại

Temporal ưu tiên fail rõ ràng thay vì silently fallback:

- snapshot rewrite bị reject;
- state hết hạn hoặc sai search_id bị reject;
- task/filter context đổi giữa session bị reject;
- canonical metadata conflict bị reject;
- dense provider failure của TRAKE được báo như dependency failure;
- TRAKE không tự fallback sang unordered scene alignment;
- ordered dense retrieval hiện không nhận SearchFilters;
- backfill và Top-M đều có quota nên evidence còn thiếu có thể vẫn là UNKNOWN;
- relation parser hiện là rule-based và chỉ nhận một tập pattern bảo thủ.

## 11. Tóm tắt thuật toán

~~~text
KIS
  cumulative snapshot
    -> delta QueryUnit
    -> global + local retrieval
    -> canonical FrameEvidence
    -> rescued-video backfill
    -> normalized multi-hint video score
    -> candidate prune
    -> gap/span bounded clustering
    -> semantic + coverage + temporal + relation score
    -> SceneCandidate[]

TRAKE
  ordered events
    -> visual Top-K per event
    -> event coverage + RRF video shortlist
    -> dense event × frame matrix
    -> monotonic DP with timestamp gap penalty
    -> canonical OrderedPathCandidate[]
~~~

Mục tiêu của folder là đảm bảo hệ thống không chỉ tìm được frame có similarity
cao, mà còn tìm được **đúng đoạn thời gian**, giữ được provenance và xuất ra
canonical identity đúng cho competition.
