# HỘI THI THỬ THÁCH TRÍ TUỆ NHÂN TẠO  
## THÀNH PHỐ HỒ CHÍ MINH NĂM 2026

## 1. NỘI DUNG CÁC TRUY VẤN VÒNG SƠ TUYỂN

### 1.1. Truy vấn dạng 1: Tìm kiếm chính xác theo văn bản (Textual Known Item Search – Textual KIS)

Đây là nhiệm vụ tìm kiếm sự kiện dựa trên mô tả bằng văn bản.

- **Nội dung truy vấn:** Ban giám khảo cung cấp một mô tả bằng ngôn ngữ tự nhiên về một sự kiện. Các đội dự thi cần định vị chính xác đoạn video chứa sự kiện bằng cách chỉ ra một khung hình bất kỳ thuộc đoạn video đó. Ở vòng sơ tuyển, nội dung đoạn mô tả được cung cấp sẵn và trọn vẹn.
- **Ví dụ:** Truy vấn: “Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh.”  
  **Kết quả nộp:** `video_id = video_abc(.mp4), frame_id = 1500`.

### 1.2. Truy vấn dạng 2: Truy vấn dạng Hỏi–Đáp (Q&A)

Đây là nhiệm vụ tìm kiếm sự kiện và trích xuất thông tin cụ thể từ video.

- **Nội dung truy vấn:** Ban giám khảo cung cấp một mô tả bằng ngôn ngữ tự nhiên của một sự kiện và một câu hỏi về thông tin trong sự kiện này. Các đội dự thi cần tìm ra chính xác khoảnh khắc liên quan và trả lời câu hỏi. Câu trả lời có thể bằng tiếng Việt hoặc tiếng Anh.
- **Ví dụ:** Truy vấn: “Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?”  
  **Kết quả nộp:** `video_id = video_xyz(.mp4), frame_id = 3450, answer = "5"` hoặc `"Năm"`.

### 1.3. Truy vấn dạng 3: Truy xuất và căn chỉnh sự kiện video theo thời gian (Temporal Retrieval and Alignment of Key Events – TRAKE)

Đây là một nhiệm vụ phức hợp đòi hỏi độ chính xác cao trong cả việc truy xuất video và căn chỉnh thời gian của các khoảnh khắc quan trọng.

TRAKE nhằm đánh giá khả năng của một hệ thống trong việc hiểu sâu sắc nội dung video một cách toàn diện, từ bối cảnh chung cho đến từng khoảnh khắc chi tiết. Nhiệm vụ yêu cầu hệ thống không chỉ tìm kiếm một video phù hợp từ một kho dữ liệu lớn mà còn phải xác định chính xác các khoảnh khắc ngữ nghĩa (**semantic keyframe**) của một chuỗi sự kiện có cấu trúc bên trong video đó.

Nhiệm vụ được chia thành hai giai đoạn:

- **Giai đoạn 1 – Truy xuất (Retrieval):** Từ một thư viện video lớn, tìm ra một video duy nhất chứa chuỗi sự kiện khớp nhất với truy vấn.
- **Giai đoạn 2 – Căn chỉnh (Alignment):** Đối với video đã truy xuất, xác định chính xác một khung hình (**semantic keyframe**) duy nhất cho mỗi giai đoạn của chuỗi sự kiện.

> **Lưu ý:** “Khung hình ngữ nghĩa” (**semantic keyframe**) trong truy vấn này là khoảnh khắc mang ý nghĩa về nội dung, khác với “I-Frame” là khung hình kỹ thuật trong các thuật toán nén video đã được cung cấp cho các đội thi.

#### Ví dụ – hành động “Nhảy cao”

Chuỗi sự kiện gồm 4 khoảnh khắc:

1. **Event 1 – Chạy đà (Approach):** Khoảnh khắc bàn chân đầu tiên chạm đất và bước qua khỏi vạch xuất phát.
2. **Event 2 – Giậm nhảy (Take-off):** Khoảnh khắc đầu tiên bàn chân của chân giậm nhảy rời hoàn toàn khỏi mặt đất.
3. **Event 3 – Bay qua xà (Clearance):** Khoảnh khắc phần hông của vận động viên ở vị trí cao nhất so với xà ngang.
4. **Event 4 – Tiếp đất (Landing):** Khoảnh khắc đầu tiên bất kỳ bộ phận nào của lưng (từ vai đến hông) bắt đầu chạm vào đệm.

---

## 2. PHƯƠNG PHÁP ĐÁNH GIÁ VÒNG SƠ TUYỂN

Đối với mỗi truy vấn, đội thi được gửi tối đa **100 câu trả lời**. Mỗi câu trả lời sẽ được chấm một điểm gọi là **Điểm Tương Quan (R-Score)** — thang đo độ chính xác nhận giá trị từ 0 đến 1:

- `1`: hoàn toàn chính xác;
- `0`: không chính xác;
- giá trị trung gian, ví dụ `0.7`: chính xác một phần.

Điểm cuối cùng cho mỗi truy vấn (**Final Score**, mục 2.2) không chỉ dựa trên một câu trả lời duy nhất, mà là trung bình của những câu trả lời tốt nhất ở nhiều vị trí xếp hạng khác nhau.

### 2.1. Điểm Tương Quan (R-Score)

Cách tính R-Score khác nhau tùy theo từng loại truy vấn.

### 2.1.1. Truy vấn Textual KIS

- **Định dạng trả lời** \((r_i)\):  
  `<video_id>, <frame_id>`

- **Điều kiện:** Câu trả lời được xem là chính xác nếu khớp video \((v_i = GT_v)\) và `frame_id` nằm trong khoảng đáp án đúng \((id_i \in [s,e])\).

$$
R\text{-}Score(r_i)
=
I\left(v_i = GT_v \land id_i \in [s,e]\right)
$$

Trong đó \(I(\cdot)\) là hàm chỉ thị, trả về `1` nếu điều kiện đúng và `0` nếu sai.

#### Ví dụ

Câu hỏi: “Tìm cảnh một người đang mở laptop trong kho video.”

Đáp án đúng của BTC:

- Video: `L01_V001`
- Khung hình: từ `500` đến `510`

Kết quả:

- `L01_V001, 505` → **Đúng**, R-Score = `1`.
- `L01_V001, 600` → **Sai**, khung hình không nằm trong khoảng cho phép. R-Score = `0`.
- `L02_V003, 505` → **Sai**, sai video. R-Score = `0`.

### 2.1.2. Truy vấn Q&A (Visual Question Answering)

- **Định dạng trả lời** \((r_i)\):  
  `<video_id>, <frame_id>, <answer>`

- **Điều kiện:** Câu trả lời được xem là chính xác nếu:
  - khớp video \((v_i = GT_v)\);
  - `frame_id` nằm trong khoảng đáp án đúng \((id_i \in [s,e])\);
  - `answer` khớp với đáp án về mặt ngữ nghĩa \((a_i = GT_a)\).

$$
R\text{-}Score(r_i)
=
I\left(
v_i = GT_v
\land id_i \in [s,e]
\land a_i = GT_a
\right)
$$

#### Ví dụ

Câu hỏi: “Trong video quay cảnh bữa tiệc, người phụ nữ mặc váy đỏ đang cầm ly màu gì?”

Đáp án đúng của BTC:

- Video: `L05_V005`
- Khung hình: từ `800` đến `900`
- Answer: `"màu xanh"`

Kết quả:

- `L05_V005, 888, màu xanh` → **Hoàn hảo**, R-Score = `1`.
- `L05_V005, 888, màu trắng` → **Sai answer**, R-Score = `0`.
- `L06_V007, 888, màu xanh` → **Sai video**, R-Score = `0`.

### 2.1.3. Truy vấn TRAKE (Temporal-alignment)

- **Định dạng trả lời** \((r_i)\):  
  `<video_id>, <frame_id_1>, ..., <frame_id_n>`

- **Điều kiện tiên quyết:** Nếu `video_id` nộp không khớp với đáp án \((v_i \ne GT_v)\), truy vấn nhận `0` điểm ngay lập tức.

Nếu đúng video, điểm được tính bằng tỷ lệ khung hình khớp với đáp án. \(N\) là tổng số khoảnh khắc trong truy vấn.

$$
R\text{-}Score(r_i)
=
\frac{1}{N}
\sum_{j=1}^{N}
I\left(id_{i,j}\in[s_j,e_j]\right),
\qquad \text{nếu } v_i = GT_v
$$

$$
R\text{-}Score(r_i)=0,
\qquad \text{nếu } v_i \ne GT_v
$$

Với mỗi khoảnh khắc thứ \(j\) trong chuỗi sự kiện, đáp án quy định một đoạn khung hình \([s_j,e_j]\) tương ứng với khoảnh khắc ngữ nghĩa đó — cùng nguyên tắc xác định đoạn \([s,e]\) như ở truy vấn Textual KIS và Q&A.

Lưu ý là đoạn ứng với khoảnh khắc ngữ nghĩa này thường rất ngắn, thông thường là **dưới 10 frame**. Một khung hình nộp \((id_{i,j})\) được coi là khớp nếu nằm trong đoạn \([s_j,e_j]\) này.

#### Ví dụ

Câu hỏi:

> “Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy:  
> (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy.”

Đáp án đúng của BTC: video `L10_V010`, với mỗi khoảnh khắc:

1. Khoảnh khắc 1 (**giậm nhảy**): `[95, 105]`
2. Khoảnh khắc 2 (**bay qua xà**): `[145, 155]`
3. Khoảnh khắc 3 (**tiếp đất**): `[195, 205]`
4. Khoảnh khắc 4 (**đứng dậy**): `[245, 255]`

**Câu trả lời của đội thi:**

`L10_V010, 101, 156, 203, 251`

Kiểm tra:

- Video: đúng `L10_V010`.
- Khoảnh khắc 1: `101 ∈ [95, 105]` → **Đúng**.
- Khoảnh khắc 2: `156 ∉ [145, 155]` → **Sai**.
- Khoảnh khắc 3: `203 ∈ [195, 205]` → **Đúng**.
- Khoảnh khắc 4: `251 ∈ [245, 255]` → **Đúng**.

Kết quả: khớp 3 trên 4 khoảnh khắc:

$$
R\text{-}Score = \frac{3}{4} = 0.75
$$

### 2.2. Điểm Cuối Cùng (Final Score)

Điểm Cuối Cùng được tính dựa trên những câu trả lời tốt nhất của đội thi ở các mốc xếp hạng (**top**) khác nhau.

Với mỗi ngưỡng:

$$
k \in \{1,5,20,50,100\}
$$

hệ thống xác định **Top-k R-Score (R@k)**: điểm R-Score cao nhất trong \(k\) câu trả lời đầu tiên.

$$
R@k
=
\max_{1\le i\le k}
\left\{
R\text{-}Score(r_i)
\right\}
$$

Điểm Cuối Cùng là trung bình cộng của 5 giá trị \(R@k\):

$$
Final\ Score
=
\frac{1}{5}
\sum_{k\in\{1,5,20,50,100\}}
R@k
$$

#### Ví dụ

Đội thi nộp 100 câu trả lời cho một truy vấn:

- Câu trả lời đầu tiên có R-Score = `0.5`.
- Câu trả lời ở vị trí số 3 có R-Score = `0.8` (cao nhất trong 100 câu).
- Câu trả lời ở vị trí số 15 có R-Score = `0.6`.
- Các câu trả lời còn lại thấp hơn.

Khi đó:

- Top 1 = `0.5` (câu trả lời đầu tiên).
- Top 5 = Top 20 = Top 50 = Top 100 = `0.8` (câu số 3 vẫn là cao nhất trong mọi ngưỡng từ 5 trở lên).

$$
Final\ Score
=
\frac{0.5 + 0.8 + 0.8 + 0.8 + 0.8}{5}
=
0.74
$$

Cách tính điểm này khuyến khích đội thi không chỉ tìm ra một câu trả lời đúng, mà còn phải xếp nó ở những vị trí đầu tiên trong danh sách trả lời của mình.

---

## 3. THÔNG TIN DỮ LIỆU VÒNG SƠ TUYỂN – ĐỢT 1

Dữ liệu cung cấp cho các đội thi để làm quen với bài toán là một phần dữ liệu từ cuộc thi AIC 2026, gồm các thành phần sau:

### Videos

Chứa video được cung cấp.

### Keyframes

Chứa tất cả keyframe được trích xuất từ video được cung cấp ở trên.

- Keyframe được lưu trong thư mục tương ứng với tên file video.
- Ví dụ: các keyframe của video `L01_V001.mp4` được lưu trong thư mục `L01_V001`.
- Tên các file keyframe được đặt theo thứ tự tăng dần.
- Vị trí (**frame index**) tương ứng của mỗi keyframe được ghi trong file metadata.

### Objects

Chứa file JSON liệt kê tất cả vật thể (**object**) phát hiện được từ mô hình **Faster R-CNN pretrained trên OpenImages V4**.

Tên file JSON tương ứng với tên file keyframe.

Ví dụ:

- Keyframe: `L01_V001/0000.jpg`
- Object JSON: `L01_V001/0000.json`

### CLIP features

Chứa CLIP features được trích xuất từ mô hình **clip-ViT-B-32** của tất cả các khung hình trong thư mục `Keyframes`.

Toàn bộ CLIP features của các keyframe được lưu trong một file `.npy` duy nhất, với thứ tự các vector feature tăng dần tương ứng với chỉ số của keyframe.

### Metadata

Thông tin metadata của video được lấy từ YouTube của kênh cung cấp dữ liệu.

- Metadata của mỗi video là một file JSON có tên tương ứng với tên file video.
- Ví dụ:
  - Video: `L01_V001.mp4`
  - Metadata: `L01_V001.json`
- Một số video trong dữ liệu cung cấp có thể không có file metadata tương ứng.

### Download dữ liệu

https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?usp=sharing

### Lưu ý

- **Dữ liệu thi chính thức là Video**; các thành phần còn lại (`Keyframes`, `Objects`, `CLIP features`, `Metadata`) chỉ nhằm mục đích cung cấp thêm thông tin hoặc hỗ trợ xây dựng giải pháp mẫu cho thí sinh.
- Đây cũng là dữ liệu **batch 1 của AIC 2025**. Dữ liệu đầy đủ của vòng sơ tuyển AIC 2026 sẽ bao gồm thêm dữ liệu **batch 2**, dự kiến được thông báo cho các đội thi trong thời gian tới.
::: ​​