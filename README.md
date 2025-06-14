Hệ Thống Thông Minh Nhận Diện Biển Số Xe
Dự án này triển khai một hệ thống thông minh sử dụng YOLOv5 để nhận diện biển số xe và thực hiện nhận dạng ký tự quang học (OCR). Hệ thống xử lý video từ camera IP (ví dụ: ứng dụng IP Webcam) để phát hiện phương tiện, nhận diện biển số và ghi lại thông tin như trạng thái vào/ra.
Hướng dẫn này cung cấp các bước chi tiết để thiết lập môi trường, cài đặt thư viện, huấn luyện mô hình YOLOv5 và chạy ứng dụng.

Yêu Cầu
Trước khi bắt đầu, hãy đảm bảo bạn có:

Máy tính có ít nhất 8GB RAM và GPU (khuyến nghị để huấn luyện).
Hai điện thoại thông minh cài ứng dụng IP Webcam (cho camera vào và ra).
Kết nối internet ổn định để tải thư viện và dữ liệu.


Thiết Lập Môi Trường
1. Cài Đặt Python
Ứng dụng yêu cầu Python 3.9.23. Tải và cài đặt từ trang chính thức:
https://www.python.org/downloads/release/python-3923/

Kiểm tra cài đặt bằng lệnh:python --version

Kết quả nên hiển thị Python 3.9.23.

2. Cài Đặt IP Webcam Trên Điện Thoại
Hệ thống sử dụng camera điện thoại làm nguồn video thông qua ứng dụng IP Webcam.

Tải ứng dụng IP Webcam từ Google Play Store trên cả hai điện thoại.
Mở ứng dụng, kéo xuống dưới cùng và nhấn Start Server để kích hoạt camera.
Ghi lại địa chỉ IP hiển thị (ví dụ: http://192.168.x.x:8080). Bạn sẽ cần địa chỉ này để chạy ứng dụng.

Lưu ý: Hệ thống cần ít nhất hai camera—một cho lối vào và một cho lối ra.
3. Tải Mã Nguồn Dự Án
Tải mã nguồn dự án từ GitHub:

Kho lưu trữ: thanhleo123/he-thong-thong-minh
Sao chép kho lưu trữ:git clone https://github.com/thanhleo123/he-thong-th-ng-minh


Di chuyển vào thư mục dự án:cd he-thong-thong-minh




Cài Đặt Thư Viện
1. Cài Đặt Thư Viện Python
Dự án yêu cầu các thư viện Python được liệt kê trong requirements.txt.

Mở terminal hoặc command prompt trong thư mục dự án.
Cài đặt thư viện:pip install -r requirements.txt



2. Cài Đặt YOLOv5
YOLOv5 được sử dụng cho cả nhận diện biển số và OCR.

Tải gói YOLOv5 từ Google Drive:
Gói YOLOv5
Cài đặt yolov5
 [yolov5 - google drive](https://drive.google.com/file/d/1g1u7M4NmWDsMGOppHocgBKjbwtDA-uIu/view?usp=sharing)

Giải nén thư mục yolov5 và sao chép vào thư mục dự án (ví dụ: he-thong-thong-minh/yolov5).


Huấn Luyện Mô Hình
Hệ thống sử dụng hai mô hình YOLOv5:

Nhận Diện Biển Số: Phát hiện biển số trong ảnh.
OCR: Nhận diện ký tự trên biển số đã phát hiện.

1. Tải Dữ Liệu Huấn Luyện
Tải các tập dữ liệu để huấn luyện cả hai mô hình:

Tập Dữ Liệu Nhận Diện Biển Số:
Link Tải
- [License Plate Detection Dataset](https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing)

Tập Dữ Liệu Nhận Diện Ký Tự (OCR):
Link Tải

- [Character Detection Dataset](https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing)

2. Chuẩn Bị Dữ Liệu

Giải nén cả hai tập dữ liệu.
Sao chép các thư mục đã giải nén vào thư mục dự án. Cấu trúc dữ liệu nên như sau:he-thong-thong-minh/
├── dataset/
│   ├── train/  # Dữ liệu huấn luyện cho nhận diện biển số
│   ├── val/    # Dữ liệu đánh giá cho nhận diện biển số
├── dataset-ocr/
│   ├── train/  # Dữ liệu huấn luyện cho OCR
│   ├── val/    # Dữ liệu đánh giá cho OCR



3. Huấn Luyện Mô Hình Nhận Diện Biển Số
Chạy lệnh sau để huấn luyện mô hình YOLOv5 cho nhận diện biển số:
python yolov5/train.py \
  --img 640 \
  --batch 8 \
  --epochs 40 \
  --data LP_detection.yaml \
  --weights model/LP_detector.pt \
  --name LP_detector


Tham số:
--img 640: Kích thước ảnh (640x640 pixel).
--batch 8: Kích thước lô (điều chỉnh theo bộ nhớ GPU).
--epochs 40: Số vòng huấn luyện.
--data LP_detection.yaml: File cấu hình dữ liệu.
--weights model/LP_detector.pt: Trọng số đã được huấn luyện trước.
--name LP_detector: Tên thư mục kết quả.


4. Huấn Luyện Mô Hình OCR
Chạy lệnh sau để huấn luyện mô hình YOLOv5 cho OCR:
python yolov5/train.py \
  --img 640 \
  --batch 8 \
  --epochs 40 \
  --data LP_ocr.yaml \
  --weights model/LP_ocr.pt \
  --name LP_ocr


Tham số: Tương tự mô hình nhận diện, nhưng sử dụng LP_ocr.yaml và LP_ocr.pt.

5. Tìm Kết Quả Huấn Luyện
Sau khi huấn luyện, trọng số mô hình và kết quả được lưu tại:
yolov5/runs/train/
├── LP_detector/  # Kết quả cho nhận diện biển số
├── LP_ocr/       # Kết quả cho OCR

Trọng số đã huấn luyện (ví dụ: best.pt) sẽ được sử dụng để suy luận.

Chạy Ứng Dụng
1. Chuẩn Bị Camera

Đảm bảo cả hai điện thoại đang chạy ứng dụng IP Webcam với server được kích hoạt.
Ghi lại địa chỉ IP của cả hai camera (vào và ra).

2. Khởi Chạy Ứng Dụng
Chạy script chính của ứng dụng:
python3 main.py

3. Nhập Địa Chỉ IP

Ứng dụng sẽ yêu cầu nhập địa chỉ IP của hai camera (ví dụ: http://192.168.x.x:8080).
Nhập địa chỉ theo hướng dẫn.

4. Vận Hành Hệ Thống

Hệ thống sẽ xử lý video từ cả hai camera.
Nó phát hiện biển số, thực hiện OCR và ghi lại thông tin phương tiện (biển số, loại xe, tọa độ và trạng thái vào/ra).

5. Dừng Ứng Dụng

Nhấn phím q để thoát chương trình.


Kết Quả Đầu Ra
Ứng dụng tạo ra kết quả trong thư mục output/ với cấu trúc sau:
output/
├── crops/        # Ảnh vùng biển số được cắt
├── data/         # Thông tin về phương tiện (biển số, loại xe, tọa độ, trạng thái vào/ra)
├── images/       # Ảnh chụp từ camera
├── process.log   # Nhật ký lưu toàn bộ tiến trình xử lý


data/: Chứa thông tin như:
Số biển số
Loại phương tiện
Tọa độ nhận diện
Trạng thái (in cho vào, out cho ra)


process.log: Ghi lại tất cả hoạt động và lỗi của hệ thống để gỡ lỗi.


Xử Lý Sự Cố

Không Nhận Camera: Đảm bảo ứng dụng IP Webcam đang chạy và địa chỉ IP đúng. Kiểm tra kết nối mạng.
Lỗi Huấn Luyện: Xác minh đường dẫn dữ liệu trong LP_detection.yaml và LP_ocr.yaml khớp với cấu trúc dự án.
Lỗi Thư Viện: Chạy lại pip install -r requirements.txt hoặc kiểm tra tương thích phiên bản Python.
Hiệu Suất Kém: Giảm kích thước lô (ví dụ: --batch 4) nếu huấn luyện thiếu bộ nhớ GPU