# 10 KIS queries: timestamp answer sheet

> Dành cho người chấm đánh giá. Không đưa cột đáp án cho người tham gia.

Bộ câu hỏi: HCMAI 2026, split `002`. Các khoảng thời gian dưới đây là nhãn đã kiểm chứng để chấm nội bộ. Mỗi câu trả lời theo kiểu VBS chọn **một timestamp**; cột cuối là một điểm đại diện nằm trong khoảng hợp lệ.

Quy đổi: mốc thời gian trong nhãn `time_windows_ms` được trình bày ở dạng `HH:MM:SS.mmm`. Timestamp đại diện được tính từ `representative_frame_idx / fps` của video tương ứng. Theo quy tắc HCMAI `frame_idx = floor(fps × timestamp_seconds)`, một frame có thể ứng với nhiều timestamp, nên không suy ngược các cận khoảng từ frame một cách duy nhất.

| STT | Query ID | Câu query | Video đúng | Khoảng timestamp hợp lệ | Timestamp đại diện |
| ---: | --- | --- | --- | --- | --- |
| 1 | `query-p2-1-kis` | Tìm đoạn video quay cảnh chú lân màu vàng thức dậy sau giấc ngủ say. Chú lân ngơ ngác khi quả bí ngô đỏ của mình đã bị ai đó lấy trộm mất. Chú lân đi tìm và phát hiện quả bí ngô của mình đang nằm trong chiếc giỏ mây của một người lạ mặt. | `L24_V035` | 00:00:20.000–00:00:45.000<br>00:06:05.000–00:06:15.000 | 00:00:24.000 |
| 2 | `query-p2-4-kis` | Hình ảnh hai bạn trẻ cùng nhau treo một tấm bạt màu xanh dương có dòng chữ "CÙNG EM ĐẾN TRƯỜNG". Trên tấm bạt có hình ảnh ngọn núi, mây trời, con đường dẫn đến trường học và hai em nhỏ trong trang phục màu vàng đang tung tăng bước đi. | `L30_V072` | 00:00:27.000–00:00:31.000 | 00:00:28.000 |
| 3 | `query-p2-10-kis` | Tìm đoạn video quay cảnh đầu bếp xào dồi trường với bông hẹ trên chảo. | `L26_V120` | 00:03:14.000–00:03:38.000 | 00:03:22.000 |
| 4 | `query-p2-11-kis` | Một dĩa cá kèo đã được xiên que, lăn qua một dĩa ớt xanh đỏ đã băm nhỏ, sau đó được xoay tròn trên một dĩa bột màu trắng. | `L26_V392` | 00:02:25.000–00:02:52.000 | 00:02:32.000 |
| 5 | `query-p2-14-kis` | Cảnh quay cận cảnh 3 tay đua xe đạp, 2 người mặc áo xanh dương đội nón bảo hiểm đỏ trắng, 1 người mặc áo vàng. Bên dưới quai nón bảo hiểm của tay đua mặc áo vàng có một sợi dây màu trắng thòng xuống. | `L23_V010` | 00:00:20.000–00:00:26.000 | 00:00:23.000 |
| 6 | `query-p2-16-kis` | Trong một ngôi nhà nông thôn có cửa sổ lớn, hai người phụ nữ đang làm thủ công trên một bộ ván ngựa, phía sau là một dãy khoảng 10 thớt gỗ được treo thành một hàng ngang. | `L29_V014` | 00:09:21.000–00:09:48.000 | 00:09:28.000 |
| 7 | `query-p2-17-kis` | Sân khấu với dòng chữ nổi 3D to, ánh kim phủ kim tuyến có nội dung: “SẮC CỔ ...” đặt ở mép trước sân khấu. | `L30_V026` | 00:01:18.000–00:01:59.000 | 00:01:40.000 |
| 8 | `query-p2-18-kis` | Đoạn phim quay từ phía sau nhóm dẫn đầu, gồm 1 tay đua dẫn trước và 3 tay đua bám phía sau, khi cả nhóm rẽ phải vào đường Hồ Tùng Mậu tại giao lộ có đèn xanh đang đếm ngược đến 13 giây. | `L23_V017` | 00:01:37.000–00:01:44.000 | 00:01:43.000 |
| 9 | `query-p2-25-kis` | Cảnh giáo viên nam đeo kính, mặc áo sơ mi kẻ sọc ngắn tay, xuất hiện ở góc dưới bên trái và dùng hai tay làm cử chỉ minh họa khi đang giảng bài.<br><br>Khung hình chứa một bức ảnh minh họa cô gái trẻ đeo kính, mặc áo sơ mi trắng, ngồi khoanh chân trên ghế sofa màu xám vừa cầm cốc nước vừa nhìn vào laptop mở trên đùi. | `L25_V045` | 00:10:50.000–00:11:14.000 | 00:11:06.000 |
| 10 | `query-p2-26-kis` | Đây là một đoạn trong bài giảng. Trên slide bao gồm:<br>- Một nhóm nhân vật người 3D màu trắng vây quanh một nhân vật màu đỏ ở chính giữa.<br><br>- Hai nhân vật hoạt hình nam đang trong tư thế thi đấu kéo co, đối đầu nhau với sợi dây thừng. | `L25_V062` | 00:09:13.000–00:09:32.500 | 00:09:18.000 |
