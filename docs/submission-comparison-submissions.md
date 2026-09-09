# So sánh `submissions/` với `submission/`

**Ngày kiểm tra:** 2026-09-04  
**Reference:** `submission/`  
**Bài kiểm tra:** `submissions/`

## Phạm vi và giới hạn

Hai thư mục hiện có cùng 30 query và trùng tên file. Reference là một ranked
submission gồm 100 candidate/query, không phải ground truth chính thức của BTC
(video đích và các khoảng frame chấp nhận). Bởi vậy đây là **bảng đối chiếu hai
submission**, không phải điểm R-Score/Final Score chính thức.

`Δ frame` là khoảng cách nhỏ nhất giữa một frame trong bài kiểm tra và bất kỳ
candidate reference cùng video. Với TRAKE, `Δ path L1` là tổng độ lệch tuyệt
đối của các event frame, lấy nhỏ nhất với một path reference cùng video. Hai
chỉ số chỉ là tín hiệu đối chiếu, không chứng minh candidate nằm trong interval
được BTC chấm đúng.

## Kiểm tra gói nộp

- Mỗi thư mục có đúng 30 CSV; không thiếu hay thừa tên file.
- Tất cả file trong `submissions/` đúng số cột của task và có giá trị frame số.
- 29 query có 99 candidate; `p2-30` có 14 candidate. Đây là hợp lệ vì BTC cho
  phép tối đa 100 candidate, nhưng `p2-30` có ít phương án dự phòng hơn.
- Ba file có UTF-8 BOM: `p2-10-kis`, `p2-15-kis`, và `p2-19-qa`. Nên lưu lại
  UTF-8 không BOM để tránh `video_id` của dòng đầu bị scorer đọc sai.
- Video ở candidate hạng 1 khớp candidate reference hạng 1 ở 28/30 query.
  Hai ngoại lệ là `p2-6` và `p2-27`.

## Bảng đối chiếu

Chỉ hiển thị năm candidate đầu tiên của bài kiểm tra khi file có 99 dòng.

| Query | Task | Reference candidate hạng 1 | Năm candidate đầu của bài bạn | Video reference hạng 1 | Đối chiếu gần nhất | Ghi chú |
|---|---|---|---|---|---|---|
| p2-1 | KIS | `L24_V035,642` | `L24_V035,832`; `800`; `810`; `842`; `852` | Hạng 1 | Δ frame 0 | Có 2 dòng trùng reference; dòng trùng đầu tiên hạng 44. |
| p2-2 | KIS | `L21_V013,22350` | `L21_V013,22474`; `22450`; `22440`; `22430`; `22460` | Hạng 1 | Δ frame 1 | — |
| p2-3 | KIS | `L24_V044,217` | `L24_V044,203`; `210`; `200`; `209`; `195` | Hạng 1 | Δ frame 0 | Có 4 dòng trùng reference; dòng trùng đầu tiên hạng 19. |
| p2-4 | KIS | `L30_V072,676` | `L30_V072,710`; `720`; `715`; `705`; `712` | Hạng 1 | Δ frame 5 | — |
| p2-5 | KIS | `L21_V022,26081` | `L21_V022,24477`; `24500`; `24450`; `24490`; `24470` | Hạng 1 | Δ frame 562 | Cần kiểm tra lại vùng frame. |
| p2-6 | KIS | `L22_V024,20709` | `L21_V018,16013`; `16000`; `15900`; `15920`; `15930` | Không có | Δ frame 2 tới lower-ranked reference | Video reference hạng 1 không xuất hiện trong bài bạn. |
| p2-7 | QA | `L21_V009,21120,1204` | `L21_V009,21183, 1204`; `21200, 1204`; `21100, 1204`; `21123, 1204`; `21085, 1204` | Hạng 1 | Δ frame 0 | Answer có leading space: dùng `1204`, không dùng ` 1204`. |
| p2-8 | TRAKE | `L27_V011,3858,3870,4012,4102` | `L27_V011,3818,3881,4024,4103`; `3803,3866,4009,4088`; `3833,3896,4039,4118`; `3788,3851,3994,4073`; `3848,3911,4054,4133` | Hạng 1 | Δ path L1 64 | — |
| p2-9 | QA | `L26_V161,2582,Cá Sòng` | `L26_V161,2491,sòng`; `2500,cá sòng`; `2480,sòng`; `2510,cá sòng`; `2470,sòng` | Hạng 1 | Δ frame 0 | Chuẩn hóa thành `Cá Sòng` giảm rủi ro semantic matching. |
| p2-10 | KIS | `L26_V120,5362` | `L26_V120,5341`; `5336`; `5346`; `5331`; `5351` | Hạng 1 | Δ frame 1 | File có UTF-8 BOM. |
| p2-11 | KIS | `L26_V392,3693` | `L26_V392,3629`; `3614`; `3644`; `3599`; `3659` | Hạng 1 | Δ frame 0 | Có 1 dòng trùng reference, ở hạng 19. |
| p2-12 | QA | `L26_V192,1248,7` | `L26_V192,5389,7`; `5374,7`; `5404,7`; `5359,7`; `5419,7` | Hạng 1 | Δ frame 6 | — |
| p2-13 | KIS | `L26_V422,4283` | `L26_V422,4255`; `4240`; `4270`; `4225`; `4285` | Hạng 1 | Δ frame 0 | Có 6 dòng trùng reference; dòng trùng đầu tiên hạng 11. |
| p2-14 | KIS | `L23_V010,631` | `L23_V010,635`; `620`; `650`; `605`; `665` | Hạng 1 | Δ frame 0 | Có 3 dòng trùng reference; dòng trùng đầu tiên hạng 6. |
| p2-15 | KIS | `L26_V074,607` | `L26_V074,495`; `490`; `500`; `485`; `505` | Hạng 1 | Δ frame 1 | File có UTF-8 BOM. |
| p2-16 | KIS | `L29_V014,10721` | `L29_V014,17500`; `17600`; `17550`; `17520`; `17530` | Hạng 1 | Δ frame 0 | Có 2 dòng trùng reference; dòng trùng đầu tiên hạng 46. |
| p2-17 | KIS | `L30_V026,1958` | `L30_V026,2500`; `2520`; `2510`; `2505`; `2490` | Hạng 1 | Δ frame 0 | Có 4 dòng trùng reference; dòng trùng đầu tiên hạng 32. |
| p2-18 | KIS | `L23_V017,2561` | `L23_V017,2428`; `2420`; `2410`; `2415`; `2430` | Hạng 1 | Δ frame 0 | Có 1 dòng trùng reference, ở hạng 19. |
| p2-19 | QA | `L30_V043,3594,Lý Thường Kiệt` | `L30_V043,3593,LÝ THƯỜNG KIỆT`; `3600,LÝ THƯỜNG KIỆT`; `3580,LÝ THƯỜNG KIỆT`; `3620,LÝ THƯỜNG KIỆT`; `3593,LÝ THƯỜNG KIỆT` | Hạng 1 | Δ frame 0 | File có BOM; dùng `Lý Thường Kiệt` thay vì case/whitespace biến thể. |
| p2-20 | KIS | `L25_V060,33304` | `L25_V060,33547`; `33587`; `33610`; `33500`; `33485` | Hạng 1 | Δ frame 1 | — |
| p2-21 | TRAKE | `L30_V031,2110,2148,2221,2260` | `L30_V031,2091,2135,2173,2236`; `2076,2120,2158,2221`; `2106,2150,2188,2251`; `2061,2105,2143,2206`; `2121,2165,2203,2266` | Hạng 1 | Δ path L1 48 | — |
| p2-22 | KIS | `L26_V470,2072` | `L26_V470,1794`; `1834`; `1854`; `1744`; `1729` | Hạng 1 | Δ frame 1 | — |
| p2-23 | QA | `L25_V012,14413,20` | `L25_V012,18514,20`; `18499,20`; `18529,20`; `18484,20`; `18544,20` | Hạng 1 | Δ frame 3 | — |
| p2-24 | KIS | `L23_V013,6839` | `L23_V013,6872`; `6870`; `6860`; `6880`; `6850` | Hạng 1 | Δ frame 1 | — |
| p2-25 | KIS | `L25_V045,16636` | `L25_V045,16271`; `16071`; `16471`; `16371`; `16171` | Hạng 1 | Δ frame 0 | Có 2 dòng trùng reference; dòng trùng đầu tiên hạng 27. |
| p2-26 | KIS | `L25_V062,13881` | `L25_V062,13961`; `13950`; `13920`; `13930`; `13940` | Hạng 1 | Δ frame 1 | — |
| p2-27 | QA | `L24_V029,506,2` | `L24_V023,6840,2`; `6840,8`; `6840,"2,8"`; `6825,2`; `6855,8` | Không có | Δ frame 5703 tới lower-ranked reference | **Sai video reference hạng 1**; answer cũng không nhất quán. |
| p2-28 | QA | `L26_V450,1013,Cá hồi` | `L26_V450,6749,CÁ HỒI`; `6750,CÁ HỒI`; `6700,CÁ HỒI`; `6685,CÁ HỒI`; `6765,CÁ HỒI` | Hạng 1 | Δ frame 1 | Chuẩn hóa thành `Cá hồi` giảm rủi ro exact matching. |
| p2-29 | QA | `L26_V439,1478,300g` | `L26_V439,1499,300g`; `1484,300g`; `1514,300g`; `1469,300g`; `1529,300g` | Hạng 1 | Δ frame 1 | — |
| p2-30 | QA | `L26_V254,550,Nghêu` | `L26_V254,450,NGHÊU SỮA`; `500,NGHÊU SỮA`; `510,NGHÊU SỮA`; `475,NGHÊU SỮA`; `475,NGHÊU SỮA` | Hạng 1 | Δ frame 10 | Chỉ 14 candidate; có answer `NGHÊU` ở các dòng sau, nên nên xếp nó lên trước. |

## Tổng hợp tín hiệu đối chiếu

- Có 25 dòng trùng tuyệt đối với reference, thuộc 9 query: p2-1, p2-3,
  p2-11, p2-13, p2-14, p2-16, p2-17, p2-18 và p2-25.
- Nếu dùng giả định quá chặt “một dòng trùng reference mới là đúng”, mean proxy
  score là `0.1533`. Đây **không phải** điểm BTC vì thiếu ground-truth interval.
- Các điểm cần ưu tiên: sửa BOM ở p2-10/p2-15/p2-19; chuẩn hóa answer ở
  p2-7/p2-9/p2-19/p2-28/p2-30; rà soát retrieval video p2-6 và đặc biệt p2-27;
  kiểm tra localization p2-5.
