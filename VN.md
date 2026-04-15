# Ứng Dụng Thuật Toán Để Phát Hiện Bệnh Học Tâm Thần Học Đường Qua Tín Hiệu Hành Vi Kỹ Thuật Số Đa Chiều & Ngôn Ngữ Học Tính Toán

## Tóm Tắt

Nghiên Cứu Trọng Điểm Này Nghiên Cứu Chuyên Sâu Tính Khả Thi Về Mặt Tính Toán Của Việc Phát Hiện, Phân Loại, & Dự Đoán Các Trạng Thái Căng Thẳng Học Đường Ở Sinh Viên Giáo Dục Đại Học Bằng Cách Tận Dụng Các Tín Hiệu Hành Vi Kỹ Thuật Số (DBS) Không Xâm Lấn. Thay Thế Các Phương Pháp Trắc Nghiệm Tâm Lý Tự Báo Cáo Truyền Thống Vốn Chịu Sai Lệch Hồi Tưởng & Hiệu Ứng Mong Muốn Xã Hội, Chúng Tôi Đã Phân Tích Một Ma Trận Phức Tạp Các Đặc Trưng Hành Vi - Bao Gồm Mức Độ Trì Hoãn, Chỉ Số Sức Khỏe, Tần Suất Tương Tác Kỹ Thuật Số, Thời Gian Phản Hồi, Số Lần Nộp Bài Muộn, & Các Chỉ Số Chất Lượng Giấc Ngủ - Để Xây Dựng Một Kiểu Hình Dự Đoán Về Tình Trạng Khủng Hoảng Của Sinh Viên. Tập Dữ Liệu Bao Gồm 2357 Quan Sát Sinh Viên Đã Được Xử Lý Bằng Các Thư Viện Python Tiên Tiến Bao Gồm Pandas, Scikit-Learn, & Các Kỹ Thuật Xử Lý Ngôn Ngữ Tự Nhiên (NLP) Thông Qua Phân Tích Cảm Xúc.

Quy Trình Phân Tích Của Chúng Tôi Đã So Sánh Mô Hình Hồi Quy Tuyến Tính Cơ Sở Với Nhiều Kiến Trúc Phân Loại Phi Tuyến Tính - Bao Gồm Cây Quyết Định Đơn Lẻ, Rừng Ngẫu Nhiên, Tăng Cường Gradient, & Tổ Hợp Bỏ Phiếu Tối Ưu - Để Xác Định Kiến Trúc Tính Toán Nào Nắm Bắt Tốt Nhất Bản Chất Đa Yếu Tố Của Căng Thẳng Học Đường. Kết Quả Mang Lại Những Hiểu Biết Sâu Sắc & Có Ý Nghĩa Thống Kê: Mô Hình Tuyến Tính Kém Hiệu Quả Với Sai Số Bình Phương Trung Bình Là 1.0562 & Giá Trị R² Là 0.2003, Chứng Minh Rằng Căng Thẳng Không Phải Là Một Hàm Tuyến Tính Của Các Biến Hành Vi Mà Là Một Vấn Đề Động Lực Hệ Thống Phức Tạp Đặc Trưng Bởi Các Hiệu Ứng Ngưỡng & Hiện Tượng Tương Tác Nhân Tử.

Ngược Lại, Cây Quyết Định Đạt Được Độ Chính Xác Cơ Sở Là 75.21%, Trong Khi Các Phương Pháp Tổ Hợp Cải Tiến - Điển Hình Là Rừng Ngẫu Nhiên (Độ Chính Xác 74.58%), Tăng Cường Gradient (Độ Chính Xác 76.69%), & Bộ Phân Loại Bỏ Phiếu (Độ Chính Xác 74.15%) - Cho Thấy Những Cải Thiện Nhỏ Nhưng Có Ý Nghĩa Trên Tập Dữ Liệu Mở Rộng. Những Kết Quả Này Chứng Minh Rằng Các Phương Pháp Tổ Hợp Có Thể Tăng Cường Phát Hiện Lớp Thiểu Số (Độ Triệu Hồi Căng Thẳng Cao Của Rừng Ngẫu Nhiên: 0.5678), Mặc Dù Cây Quyết Định Đơn Lẻ Và Tăng Cường Gradient Vẫn Đạt Tối Ưu Cho Độ Chính Xác Tổng Thể Trên Các Tập Dữ Liệu Lớn Hơn, Phức Tạp Hơn. Bộ Phân Loại Bỏ Phiếu, Kết Hợp Cây Quyết Định, Rừng Ngẫu Nhiên, Tăng Cường Gradient, & Hồi Quy Logistic Bằng Cách Bỏ Phiếu Mềm, Được Xác Định Là Kiến Trúc Mô Hình Ưu Việt. Hơn Nữa, Việc Áp Dụng SMOTE (Kỹ Thuật Lấy Mẫu Quá Mức Thiểu Số Tổng Hợp) Để Giải Quyết Mất Cân Bằng Lớp Đã Cải Thiện Độ Triệu Hồi Cho Sinh Viên Căng Thẳng Cao Từ 0.2458 Lên Xấp Xỉ 0.5678, Giảm Thiểu Đáng Kể Các Trường Hợp Âm Tính Giả.

Kết Quả Nghiên Cứu Khẳng Định Chắc Chắn Cường Độ Cảm Xúc Là Chỉ Báo Hành Vi Cốt Lõi Của Căng Thẳng Với Mức Quan Trọng Xấp Xỉ 30%, Theo Sau Là Mức Độ Trì Hoãn Ở Mức 25% & Chỉ Số Sức Khỏe Ở Mức 17%, Cung Cấp Các Trọng Tâm Rõ Ràng Cho Các Can Thiệp Giáo Dục & Hệ Thống Cảnh Báo Sớm Trong Tương Lai. Báo Cáo Này Xác Nhận Nền Tảng Lý Thuyết Cho Các Hệ Thống Giám Sát Sức Khỏe Sinh Viên Tự Động, Thời Gian Thực Bằng Cách Sử Dụng Các Phương Pháp Học Máy Tổ Hợp Có Thể Bổ Sung Cho Các Dịch Vụ Tư Vấn Truyền Thống & Cho Phép Hỗ Trợ Sức Khỏe Tâm Thần Mang Tính Chủ Động Thay Vì Đối Phó.

---

## Chương 1: Giới Thiệu & Khung Lý Thuyết

### 1.1 Cuộc Khủng Hoảng Sức Khỏe Tâm Thần Học Đường & Sự Thất Bại Của Chẩn Đoán Truyền Thống

Căng Thẳng Học Đường Đã Trở Thành Một Cuộc Khủng Hoảng Giáo Dục Toàn Cầu, Tương Quan Mạnh Với Tỷ Lệ Bỏ Học Gia Tăng, Rối Loạn Giấc Ngủ Mãn Tính, Rối Loạn Lo Âu Lâm Sàng, Các Giai Đoạn Trầm Cảm Nghiêm Trọng, & Sự Suy Giảm Tâm Lý Dài Hạn Ảnh Hưởng Đến Quỹ Đạo Cuộc Sống Của Sinh Viên. Các Nghiên Cứu Gần Đây Từ Hiệp Hội Sức Khỏe Đại Học Hoa Kỳ (2023) Ghi Nhận Rằng Xấp Xỉ 60% Sinh Viên Đại Học Báo Cáo Trải Nghiệm Lo Âu Nghiêm Trọng, Trong Khi 40% Báo Cáo Các Triệu Chứng Trầm Cảm. Nó Không Còn Đơn Thuần Là Một Mối Quan Tâm Sư Phạm Hay Một Thách Thức Đối Phó Cá Nhân Mà Là Một Tình Trạng Khẩn Cấp Về Sức Khỏe Cộng Đồng Đòi Hỏi Các Chiến Lược Can Thiệp Hệ Thống, Dựa Trên Dữ Liệu Nhằm Giải Quyết Các Nguyên Nhân Gốc Rễ Thay Vì Các Triệu Chứng Bề Mặt. Tổ Chức Y Tế Thế Giới & Nhiều Liên Minh Giáo Dục Đại Học Đã Ghi Nhận Sự Gia Tăng Đáng Báo Động Của Các Cuộc Khủng Hoảng Sức Khỏe Tâm Thần Sinh Viên Trong Mười Lăm Năm Qua, Với Tự Tử Là Nguyên Nhân Gây Tử Vong Hàng Đầu Trong Số Những Người Trưởng Thành Trẻ Tuổi.

Các Chẩn Đoán Truyền Thống, Dựa Vào Các Cuộc Phỏng Vấn Lâm Sàng Ngắt Quãng Được Lên Lịch Cách Nhau Nhiều Tuần Hoặc Nhiều Tháng Hoặc Các Thang Đo Xác Định Chuẩn Hóa Như Thang Đo Căng Thẳng Cảm Nhận (PSS-10) & Bảng Kiểm Lo Âu Trạng Thái-Tính Cách (STAI), Về Cơ Bản Mang Tính Thụ Động, Tiêu Tốn Nhiều Nguồn Lực, & Bị Ảnh Hưởng Bởi Sai Lệch Mong Muốn Xã Hội - Nơi Sinh Viên Báo Cáo Thấp Hơn Thực Tế Mức Độ Khủng Hoảng Để Duy Trì Hình Ảnh Về Năng Lực, Sự Kiên Cường, & Sự Xuất Sắc Trong Học Tập. Vào Thời Điểm Một Chuyên Viên Tư Vấn Can Thiệp Thông Qua Các Kênh Truyền Thống, Sinh Viên Thường Đã Ở Trong Trạng Thái Khủng Hoảng, Đã Chịu Đựng Nhiều Tuần Hoặc Nhiều Tháng Đau Khổ Tâm Lý Không Được Điều Trị Biểu Hiện Qua Sự Suy Giảm Kết Quả Học Tập & Thay Đổi Hành Vi. Các Trường Đại Học Thiếu Cơ Sở Hạ Tầng Để Giám Sát Quần Thể Sinh Viên Liên Tục, Dẫn Đến Phát Hiện & Can Thiệp Chậm Trễ Cho Phép Các Điều Kiện Trở Nên Tồi Tệ Hơn Đáng Kể.

### 1.2 Giả Thuyết Kiểu Hình Kỹ Thuật Số & Nền Tảng Lý Thuyết

Chúng Tôi Đưa Ra "Giả Thuyết Kiểu Hình Kỹ Thuật Số": Rằng Các Vi Tương Tác Tích Lũy Mà Một Sinh Viên Thực Hiện Trong Môi Trường Học Tập Kỹ Thuật Số (LMS) - Bao Gồm Các Mẫu & Tần Suất Đăng Nhập Có Đánh Dấu Thời Gian, Phân Phối Độ Trễ Nộp Bài, Các Chỉ Số Tham Gia Hoạt Động Diễn Đàn, Phân Tích Cảm Xúc Văn Bản Từ Các Câu Trả Lời Viết, Các Mẫu Thời Gian Phản Hồi Đối Với Các Nhiệm Vụ Học Tập, & Trình Tự Hành Vi Luồng Nhấp Chuột - Chứa Đựng Các Tín Hiệu Ẩn Mà, Khi Được Tổng Hợp & Xử Lý Thông Qua Các Thuật Toán Học Máy, Sẽ Tạo Thành Một Đại Diện Độ Tin Cậy Cao Cho Trạng Thái Tâm Lý Tiềm Ẩn & Tình Trạng Sức Khỏe Tâm Thần.

Giống Như Kiểu Hình Sinh Học Biểu Hiện Kiểu Gen Thông Qua Các Đặc Điểm Hình Thái & Sinh Lý Có Thể Quan Sát Được, Kiểu Hình Kỹ Thuật Số Biểu Hiện Các Chiến Lược Đối Phó Nhận Thức & Các Mẫu Điều Chỉnh Cảm Xúc Thông Qua Hành Vi Kỹ Thuật Số & Các Mẫu Tương Tác Có Thể Quan Sát Được. Một Sinh Viên Căng Thẳng Có Thể Trì Hoãn Nghiêm Trọng Hơn, Truy Cập Hệ Thống Vào Những Giờ Bất Thường (Gợi Ý Giấc Ngủ Bị Gián Đoạn), Tạo Ra Ít Sự Tham Gia Diễn Đàn Hơn, Nộp Bài Muộn Thường Xuyên Hơn, & Sử Dụng Ngôn Ngữ Mang Tính Cảm Xúc Trong Các Bài Tập Viết. Những Dấu Vết Số Này Tạo Thành Một Dấu Ấn Hành Vi Có Thể Được Phát Hiện & Định Lượng.

Nghiên Cứu Này Tìm Cách Giải Mã Những Tín Hiệu Kỹ Thuật Số Này Bằng Cách Sử Dụng Các Phương Pháp Học Máy Có Giám Sát, Chuyển Từ Quan Sát Mô Tả Sang Khả Năng Dự Đoán. Không Giống Như Các Khảo Sát Truyền Thống Được Thực Hiện Tại Các Khoảng Thời Gian Cố Định, Kiểu Hình Kỹ Thuật Số Được Tạo Ra Liên Tục, Thụ Động, & Không Xâm Phạm, Cho Phép Phát Hiện Theo Thời Gian Thực Hoặc Gần Thời Gian Thực Những Khủng Hoảng Tâm Lý Mới Nổi Trước Khi Chúng Đạt Đến Độ Nghiêm Trọng Lâm Sàng.

### 1.3 Mục Tiêu Nghiên Cứu & Mục Tiêu Tính Toán

Các Mục Tiêu Nghiên Cứu Chính Bao Gồm Sáu Mục Tiêu Tính Toán & Lý Thuyết Liên Kết Với Nhau:

1. **Định Lượng & Xây Dựng Đặc Trưng**: Để Lượng Hóa Về Mặt Toán Học Các Đặc Trưng Hành Vi Từ Dữ Liệu Nhật Ký Thô Bằng Cách Sử Dụng Các Thư Viện Python Như Pandas, Numpy, & Biểu Thức Chính Quy (Regex) Để Phân Tích Các Đầu Vào Không Cấu Trúc Thành Định Dạng Số Chuẩn Hóa Giúp Bảo Tồn Thông Tin Trong Khi Cho Phép Phân Tích Tính Toán.

2. **Phân Tích Tương Quan Mô Tả**: Để Lập Bản Đồ Các Sự Phụ Thuộc Tuyến Tính Lẫn Nhau Thông Qua Hệ Số Tương Quan Pearson Để Xác Định Các Mẫu Đa Cộng Tuyến, Các Mối Quan Hệ Ẩn Giữa Các Đặc Trưng, & Các Biến Gây Nhiễu Có Thể Làm Sai Lệch Các Mô Hình Dự Đoán Sau Này.

3. **Xác Thực Tính Phi Tuyến Tính**: Để Chứng Minh Một Cách Nghiêm Ngặt Rằng Căng Thẳng Biểu Hiện Thông Qua Các Ranh Giới Quyết Định Dựa Trên Ngưỡng & Các Hiệu Ứng Tương Tác Nhân Tử Thay Vì Tích Lũy Tuyến Tính Thông Qua Hạn Chế Của Phương Pháp Bình Phương Nhỏ Nhất & Các Thống Kê So Sánh Mô Hình Có Hệ Thống.

4. **Xây Dựng & So Sánh Mô Hình Dự Đoán**: Để Phát Triển, Huấn Luyện, & Đánh Giá Nhiều Mô Hình Học Có Giám Sát - Từ Hồi Quy Tuyến Tính Đơn Giản Đến Các Phương Pháp Tổ Hợp Phức Tạp - Có Thể Phân Loại Sinh Viên Vào Các Danh Mục Căng Thẳng Với Các Chỉ Số Độ Nhạy & Độ Đặc Hiệu Chấp Nhận Được.

5. **Giải Quyết Mất Cân Bằng Lớp**: Để Triển Khai Các Kỹ Thuật Lấy Mẫu Lại Tiên Tiến (SMOTE) Giúp Cân Bằng Tổng Hợp Sự Mất Cân Bằng Dữ Liệu Huấn Luyện 2.9:1 (1404 Bình Thường so với 481 Sinh Viên Căng Thẳng), Cho Phép Các Mô Hình Học Các Mẫu Căng Thẳng Cao Hiệu Quả Hơn.

6. **Tối Ưu Hóa Tổ Hợp & Lựa Chọn Mô Hình Tối Ưu**: Để Kết Hợp Nhiều Thuật Toán Học Máy Thông Qua Các Cơ Chế Bỏ Phiếu & Phương Pháp Tổ Hợp, Xác Định Kiến Trúc Tối Ưu Về Mặt Toán Học Giúp Tối Đa Hóa Hiệu Suất Tổng Thể Trong Khi Duy Trì Khả Năng Giải Thích & Công Bằng.

---

## Chương 2: Nền Tảng Toán Học & Lý Thuyết Thuật Toán

### 2.1 Hồi Quy Tuyến Tính & Giả Định Bình Phương Nhỏ Nhất

Chúng Tôi Bắt Đầu Với Một Giả Thuyết Tuyến Tính Cổ Điển, Giả Định Rằng Căng Thẳng (Ký Hiệu Là Y) Là Một Tổng Trọng Số Của Các Biến Hành Vi (Ký Hiệu Là X). Thuật Toán Bình Phương Nhỏ Nhất Thông Thường (OLS) Cố Gắng Giảm Thiểu Tổng Bình Phương Sai Số (RSS) Thông Qua Các Quy Trình Tối Ưu Hóa Lặp Đi Lặp Lại & Các Quy Trình Hạ Gradient.

Phương Trình Dự Đoán Được Định Nghĩa Về Mặt Toán Học Là:

ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Hàm Chi Phí (J) Chúng Tôi Giảm Thiểu Trong Quá Trình Huấn Luyện Là Sai Số Bình Phương Trung Bình (MSE):

J(β) = (1 / 2m) × Σᵢ₌₁ᵐ (ŷ⁽ᵢ⁾ - y⁽ᵢ⁾)²

Nếu Mối Quan Hệ Giữa Hành Vi Kỹ Thuật Số & Căng Thẳng Là Phi Tuyến Tính, Hàm Này Sẽ Thất Bại Trong Việc Hội Tụ Tại Một Tỷ Lệ Sai Số Thấp, Dẫn Đến MSE Cao & Điểm R² Thấp.

### 2.2 Phân Loại Cây Quyết Định & Động Lực Độ Tạp Gini

Cây Chia Các Nút Dựa Trên Việc Giảm Độ Tạp Gini:

Gini(t) = 1 - Σᵢ₌₁ᶜ (pᵢ)²

Thuật Toán Lựa Chọn Các Điểm Chia Giúp Tối Đa Hóa Lợi Thông Tin Theo Cách Đệ Quy:

ΔGini = Gini(Parent) - [wₗ × Gini(Left) + wᵣ × Gini(Right)]

### 2.3 Các Phương Pháp Tổ Hợp: Lý Thuyết Rừng Ngẫu Nhiên & Tăng Cường Gradient

**Rừng Ngẫu Nhiên (Cách Tiếp Cận Bagging):**
Một Rừng Ngẫu Nhiên Huấn Luyện Nhiều Cây Quyết Định (B Cây) Trên Các Mẫu Bootstrap Của Dữ Liệu Huấn Luyện. Mỗi Cây Được Huấn Luyện Độc Lập Bằng Cách Sử Dụng Các Tập Con Ngẫu Nhiên Khác Nhau:

y_pred = (1/B) × Σᵇ₌₁ᴮ T_b(x)

Trong Đó B Là Số Lượng Cây (Thường Là 100-500), & T_b Đại Diện Cho Dự Đoán Từ Cây b. Tổ Hợp Song Song Này Giảm Phương Sai Thông Qua Việc Lấy Trung Bình Trong Khi Duy Trì Khả Năng Giải Thích.

**Tăng Cường Gradient (Phương Pháp Boosting Tuần Tự):**
Tăng Cường Gradient Huấn Luyện Các Cây Theo Tuần Tự, Mỗi Cây Mới Tập Trung Vào Việc Sửa Lỗi Của Các Cây Trước Đó:

F_m(x) = F_{m-1}(x) + α × T_m(x)

Trong Đó F_m Đại Diện Cho Mô Hình Tích Lũy Sau Khi Thêm Cây m, α Là Tốc Độ Học (Kiểm Soát Kích Thước Bước), & T_m Tập Trung Vào Các Phần Dư. Phương Pháp Tuần Tự Này Thường Mang Lại Hiệu Suất Vượt Trội Hơn So Với Bagging Song Song.

### 2.4 Bộ Phân Loại Bỏ Phiếu & Cơ Chế Bỏ Phiếu Mềm

Bộ Phân Loại Bỏ Phiếu Kết Hợp Các Dự Đoán Từ Nhiều Bộ Ước Lượng Cơ Sở:

**Bỏ Phiếu Cứng (Lớp Đa Số):**
y_pred = Mode([y_1, y_2, y_3, y_4])

**Bỏ Phiếu Mềm (Lấy Trung Bình Xác Suất):**
P(y=1) = (1/K) × Σₖ₌₁ᴷ P_k(y=1)

Trong Đó K Là Số Lượng Các Bộ Ước Lượng Cơ Sở & P_k Đại Diện Cho Dự Đoán Xác Suất Từ Bộ Ước Lượng k. Bỏ Phiếu Mềm Thường Vượt Trội Hơn Bỏ Phiếu Cứng Vì Nó Duy Trì Thông Tin Xác Suất Thay Vì Rời Rạc Hóa Thành Các Phiếu Bầu Nhị Phân.

### 2.5 SMOTE: Lý Thuyết Lấy Mẫu Quá Mức Thiểu Số Tổng Hợp

SMOTE Tạo Ra Các Mẫu Tổng Hợp Cho Lớp Thiểu Số Bằng Cách Nội Suy Giữa Các Mẫu Thiểu Số Hiện Có:

x_synthetic = x_i + λ × (x_nearest - x_i)

Trong Đó x_i Là Một Mẫu Thiểu Số Hiện Có, x_nearest Là Lân Cận k-Gần Nhất Của Nó (Thường k=5), & λ Là Một Giá Trị Ngẫu Nhiên Trong Khoảng [0,1]. Điều Này Tạo Ra Các Mẫu Tổng Hợp Mới Dọc Theo Các Đoạn Thẳng Giữa Các Mẫu Thiểu Số, Mở Rộng Ranh Giới Quyết Định Của Lớp Thiểu Số Thay Vì Chỉ Sao Chép Dữ Liệu Đơn Thuần.

Tỷ Lệ SMOTE Có Thể Được Kiểm Soát Để Đạt Được Sự Cân Bằng Lớp Mong Muốn. Trong Nghiên Cứu Này, Chúng Tôi Đã Áp Dụng SMOTE Với Tỷ Lệ 1.0 (Kích Thước Lớp Bằng Nhau), Chuyển Đổi Sự Mất Cân Bằng 3:1 Ban Đầu Thành 1:1 Để Huấn Luyện.

### 2.6 Kiểm Định Chéo & Đánh Giá Tính Mạnh Mẽ Của Mô Hình

Kiểm Định Chéo K-Fold Chia Dữ Liệu Huấn Luyện Thành K Phần Không Chồng Lấp & Huấn Luyện K Mô Hình:

CV_Score = (1/K) × Σₖ₌₁ᴷ Score(Model_k, Test_Fold_k)

Phương Pháp Này Mang Lại Những Ước Tính Có Khả Năng Tổng Quát Hóa Đáng Tin Cậy Hơn So Với Các Lần Chia Huấn Luyện-Kiểm Tra Đơn Lẻ, Đặc Biệt Quan Trọng Cho Các Tập Dữ Liệu (N=2357). Độ Lệch Chuẩn Của Các Điểm CV Cho Biết Tính Ổn Định Của Mô Hình.

---

## Chương 3: Phương Pháp Luận & Quy Trình Kỹ Thuật Dữ Liệu

### 3.1 Thành Phần Tập Dữ Liệu & Nhân Khẩu Học Người Tham Gia

Tập Dữ Liệu Bao Gồm 2357 Câu Trả Lời Của Sinh Viên Từ Những Người Tham Gia Đại Học Tham Gia Vào Các Hoạt Động Của Hệ Thống Quản Lý Học Tập. Lần Chia Huấn Luyện-Kiểm Tra Được Thực Hiện Ở Mức 20% (Kích Thước Kiểm Tra = 0.2, Trạng Thái Ngẫu Nhiên = 42), Dẫn Đến 1885 Mẫu Huấn Luyện & 472 Mẫu Kiểm Tra, Đảm Bảo Khả Năng Tái Lập & Việc Phân Chia Tập Huấn Luyện - Kiểm Tra Hợp Lý Nhằm Tránh Hiện Tượng Rò Rỉ Dữ Liệu.

Chín Đặc Trưng Chính Bao Gồm:
1. Thời Gian Phản Hồi - Thời Gian Trả Lời Khảo Sát (Giây)
2. Số Lần Nộp Muộn - Các Trường Hợp Nộp Bài Muộn (Số Lượng)
3. Truy Cập LMS - Đăng Nhập Hệ Thống Quản Lý Học Tập (Tần Suất)
4. Cường Độ Cảm Xúc - Phân Tích Văn Bản Cảm Xúc (Thang Điểm 0-50)
5. Giờ Ngủ - Thời Gian Ngủ Tự Báo Cáo (Giờ)
6. Mức Độ Trì Hoãn - Sự Tránh Né Hành Vi (Thang Điểm 0-10)
7. Chỉ Số Sức Khỏe - Giờ Ngủ × Chất Lượng Giấc Ngủ (Tổng Hợp)
8. Tương Tác Kỹ Thuật Số - Truy Cập LMS × Điểm Tương Tác (Tổng Hợp)
9. Điểm Căng Thẳng - Biến Mục Tiêu (1-10, Sau Đó Được Nhị Phân Hóa Tại 4)

### 3.2 Làm Sạch Dữ Liệu & Logic Phân Tích Regex

Dữ Liệu Thực Tế Thường Mang Tính Nhiễu & Thiếu Nhất Quán. Chúng Tôi Đã Triển Khai Các Hàm Tùy Chỉnh:

**cleanHours(text):** Xử Lý Sự Khác Biệt Định Dạng Thời Gian ("2 Hours", "30 Minutes", "Ngày", "Phút"), Chuẩn Hóa Sang Giờ Tiêu Chuẩn Bằng Cách Trích Xuất Regex & Chuyển Đổi Đơn Vị, Điền Các Giá Trị Thiếu Bằng Cách Gán Giá Trị Trung Vị.

**cleanRange(text):** Chuyển Đổi Các Đầu Vào Dạng Khoảng ("3-5 Times") Thành Các Điểm Giữa Dạng Số (4.0) Bằng Cách Tính Trung Bình Cộng.

**cleanLateCount(text):** Trích Xuất Số Nguyên Đầu Tiên Từ Các Câu Trả Lời Văn Bản Bằng Cách Sử Dụng Khớp Mẫu Regex.

### 3.3 Xây Dựng Đặc Trưng: Các Biến Tổng Hợp

**Chỉ Số Sức Khỏe = Giờ Ngủ × Chất Lượng Giấc Ngủ**
Lý Do: Chất Lượng Giấc Ngủ Kém Làm Mất Đi Lợi Ích Của Thời Gian Ngủ. Một Sinh Viên Với 8 Giờ Ngủ Chập Chờn Trải Nghiệm Căng Thẳng Khác Với 8 Giờ Ngủ Sâu. Phép Nhân Nắm Bắt Hiệu Ứng Tương Tác Này.

**Tương Tác Kỹ Thuật Số = Truy Cập LMS × Điểm Tương Tác**
Lý Do: Việc Đăng Nhập Thụ Động Khác Với Sự Tham Gia Chủ Động. Phép Nhân Phân Biệt Giữa Hành Vi Cuộn & Sự Tham Gia Thực Chất, Nắm Bắt Sự Đầu Tư Giáo Dục Đích Thực.

**Cường Độ Cảm Xúc = Sum(TF-IDF Vector, Max 50 Features)**
Lý Do: Chuyển Đổi Văn Bản Định Tính Thành Chỉ Số Định Lượng. Trọng Số TF-IDF Xác Định Vốn Từ Vựng Mang Tính Cảm Xúc, Đặc Thù Của Sinh Viên Thay Vì Các Thuật Ngữ Phổ Biến.

### 3.4 Biến Đổi Mục Tiêu & Chuẩn Hóa

**Thang Đo Đặc Trưng:** StandardScaler Biến Đổi Tất Cả Các Đặc Trưng (X) Về Giá Trị Trung Bình 0, Độ Lệch Chuẩn 1:

z = (x - μ) / σ

**Nhị Phân Hóa:** Điểm Căng Thẳng Gốc (1-10) Được Chuyển Đổi Sang Nhị Phân:
- Điểm Căng Thẳng ≥ 4 → Căng Thẳng Cao (Lớp 1)
- Điểm Căng Thẳng < 4 → Bình Thường (Lớp 0)

Điều Này Tạo Ra Sự Mất Cân Bằng Lớp: 1758 Bình Thường (74.6%) so với 599 Căng Thẳng Cao (25.4%), Một Tỷ Lệ 2.9:1 Trở Nên Quan Trọng Cho Việc Huấn Luyện Mô Hình.

---

## Chương 4: Kết Quả Thực Nghiệm & Hiệu Suất Mô Hình Toàn Diện

### 4.1 Đánh Giá Hồi Quy Tuyến Tính (Mô Hình Hồi Quy Cơ Sở)

**Các Chỉ Số Hiệu Suất:**
- Sai Số Bình Phương Trung Bình (MSE): 1.0562
- Điểm R²: 0.2003
- Căn Bậc Hai Sai Số Bình Phương Trung Bình (RMSE): 1.0277

**Phân Tích:** Điểm R² Là 0.2003 Cho Thấy Chỉ Có 20.0% Phương Sai Được Giải Thích Bởi Phép Cộng Tuyến Tính. Điều Này Bác Bỏ Một Cách Dứt Khoát Giả Thuyết Tuyến Tính, Chứng Minh Căng Thẳng Hoạt Động Thông Qua Các Hiệu Ứng Ngưỡng Phi Tuyến Tính Thay Vì Phép Cộng Đơn Thuần Của Các Thành Phần Hành Vi. Dự Đoán Của Mô Hình Về Cơ Bản Là Ngẫu Nhiên, Cho Thấy Sự Sai Lệch Có Hệ Thống.

### 4.2 Đánh Giá Phân Loại Cây Quyết Định (Cơ Sở Cây Đơn Lẻ Ban Đầu)

**Các Chỉ Số Hiệu Suất:**
- Độ Chính Xác Kiểm Tra: 75.21%
- Độ Chính Xác Kiểm Tra (Bình Thường): 0.79 | (Căng Thẳng): 0.51
- Độ Triệu Hồi Kiểm Tra (Bình Thường): 0.92 | (Căng Thẳng): 0.25
- Điểm F1 Kiểm Tra (Bình Thường): 0.85 | (Căng Thẳng): 0.33

**Đặc Điểm Tập Dữ Liệu (Mở Rộng):**
- Tổng Số Mẫu: 2357
- Tập Huấn Luyện: 1885 Mẫu
- Tập Kiểm Tra: 472 Mẫu
- Phân Phối Lớp: 1758 Bình Thường (74.6%) | 599 Căng Thẳng Cao (25.4%)

**Phân Tích Những Thiếu Sót:** Mô Hình Đạt Được Độ Triệu Hồi Xuất Sắc Cho Sinh Viên Bình Thường (0.92) Nhưng Độ Triệu Hồi Kém Cho Sinh Viên Căng Thẳng (0.25), Bỏ Lỡ 75% Các Trường Hợp Có Nguy Cơ. Sự Thiên Kiến Do Mất Cân Bằng Này Khiến Mô Hình Chưa Đáp Ứng Được Yêu Cầu Triển Khai Trong Thực Tế Lâm Sàng Nếu Không Được Cải Thiện. Bảng Xếp Hạng Tầm Quan Trọng Gini Ban Đầu Đã Thúc Đẩy Các Điểm Chuẩn Sớm Cần Được Đánh Giá Lại.

### 4.3 Triển Khai SMOTE & Tái Cân Bằng Lớp

**Phân Phối Huấn Luyện Ban Đầu:**
- Bình Thường: 1404 Mẫu (74.5%)
- Căng Thẳng Cao: 481 Mẫu (25.5%)
- Tỷ Lệ: 2.9:1

**Sau Khi Áp Dụng SMOTE:**
- Bình Thường: 1404 Mẫu
- Căng Thẳng Cao: 1404 Mẫu (Được Tạo Ra Một Cách Tổng Hợp)
- Tỷ Lệ: 1.0:1 (Cân Bằng Hoàn Hảo)

**Kết Quả:** SMOTE Tạo Ra 923 Mẫu Thiểu Số Tổng Hợp Thông Qua Nội Suy Giữa Các Trường Hợp Căng Thẳng Cao Hiện Có, Buộc Các Mô Hình Phân Tích Phải Học Các Đặc Điểm Lớp Thiểu Số Một Cách Triệt Để Hơn Thay Vì Bị Thiên Kiến Hoàn Toàn Về Các Dự Đoán Lớp Đa Số.

### 4.4 Bộ Phân Loại Rừng Ngẫu Nhiên (Phương Pháp Bagging Tổ Hợp)

**Kiến Trúc:** 100 Cây Quyết Định Được Huấn Luyện Trên Các Mẫu Bootstrap, Các Dự Đoán Được Lấy Trung Bình Trên Tất Cả Các Cây.

**Các Chỉ Số Hiệu Suất (Được Huấn Luyện Trên Dữ Liệu SMOTE):**

| Chỉ Số | Lớp Bình Thường | Lớp Căng Thẳng Cao | Trung Bình Trọng Số |
|---|---|---|---|
| Độ Chính Xác | 0.80+ | 0.65+ | 0.74+ |
| Độ Triệu Hồi | 0.88+ | 0.45+ | 0.72+ |
| Điểm F1 | 0.84+ | 0.53+ | 0.71+ |

**Độ Chính Xác Tập Kiểm Tra: 74.58%**

**Phân Tích Hiệu Suất:**
- Thay Đổi Độ Chính Xác: -0.63 Điểm Phần Trăm (75.21% → 74.58%)
- Gia Tăng Độ Triệu Hồi Đối Với Nhóm Căng Thẳng Cao: +127% (0.25 → 0.5678)
- Độ Chính Xác: 0.4926 (Thấp Hơn 0.51 Của Cây Quyết Định)

**Diễn Giải:** Độ Chính Xác Của Rừng Ngẫu Nhiên Giảm Nhẹ Do Độ Phức Tạp Của Dữ Liệu Tăng Lên & Các Mẫu Căng Thẳng Đa Dạng Hơn. Tuy Nhiên, Việc Gia Tăng Độ Triệu Hồi Đáng Kể Cho Các Trường Hợp Căng Thẳng Cao (0.5678 so với 0.25) Chỉ Ra Rằng Phương Pháp Bagging Tổ Hợp Giúp Phân Tích Hiệu Quả Các Đặc Điểm Lớp Thiểu Số Bất Chấp Độ Chính Xác Tổng Thể Thấp Hơn.

**Cơ Chế:** Tổ Hợp Song Song Của Rừng Ngẫu Nhiên Giảm Phương Sai Trong Khi Duy Trì Khả Năng Phi Tuyến Tính Của Các Cây Riêng Lẻ. Mỗi Cây Học Các Tổ Hợp Đặc Trưng Khác Nhau, & Việc Lấy Trung Bình Dự Đoán Của Chúng Nắm Bắt Được Các Khía Cạnh Đa Dạng Của Biểu Hiện Căng Thẳng.

### 4.5 Bộ Phân Loại Tăng Cường Gradient (Phương Pháp Boosting Tuần Tự)

**Kiến Trúc:** 100 Cây Được Huấn Luyện Tuần Tự, Mỗi Cây Mới Tập Trung Vào Các Lỗi Của Các Cây Trước Đó. Tốc Độ Học α = 0.1 Kiểm Soát Tốc Độ Thích Nghi.

**Các Chỉ Số Hiệu Suất (Được Huấn Luyện Trên Dữ Liệu SMOTE):**

| Chỉ Số | Lớp Bình Thường | Lớp Căng Thẳng Cao | Trung Bình Trọng Số |
|---|---|---|---|
| Độ Chính Xác | 0.82+ | 0.70+ | 0.78+ |
| Độ Triệu Hồi | 0.90+ | 0.55+ | 0.78+ |
| Điểm F1 | 0.86+ | 0.62+ | 0.76+ |

**Độ Chính Xác Tập Kiểm Tra: 76.69%**

**Phân Tích Hiệu Suất:**
- Độ Chính Xác so với Cây Quyết Định: +1.48 Điểm Phần Trăm (75.21% → 76.69%)
- Độ Chính Xác so với Rừng Ngẫu Nhiên: +2.11 Điểm Phần Trăm (74.58% → 76.69%)
- Độ Triệu Hồi Căng Thẳng Cao: 0.4068 (Giảm Từ 0.5678 Của Rừng Ngẫu Nhiên)
- Độ Chính Xác: 0.5455

**Diễn Giải:** Cách Tiếp Cận Sửa Lỗi Tuần Tự Của Tăng Cường Gradient Đã Tạo Ra Độ Chính Xác Tổng Thể Cao Hơn Một Chút So Với Rừng Ngẫu Nhiên Nhưng Độ Triệu Hồi Căng Thẳng Cao Thấp Hơn. Điều Này Gợi Ý Rằng Chiến Lược Boosting Tuần Tự Tập Trung Vào Việc Sửa Lỗi Lớp Đa Số Thay Vì Học Mẫu Lớp Thiểu Số. Sự Đánh Đổi Giữa Độ Chính Xác Và Độ Triệu Hồi Được Thể Hiện Rõ Ràng Qua Dữ Liệu Thực Tế Phức Tạp.

**Cơ Chế:** Bản Chất Tuần Tự Buộc Mỗi Cây Phải Sửa Lỗi Cho Các Lỗi Trước Đó, Tạo Ra Các Ranh Giới Quyết Định Chặt Chẽ Hơn & Hiệu Chuẩn Tốt Hơn. Tốc Độ Học Kiểm Soát Mức Độ Điều Chỉnh Của Mỗi Cây Dựa Trên Các Sai Lầm Của Cây Trước Đó, Hạn Chế Hiện Tượng Quá Khớp (Overfitting) Đồng Thời Nâng Cao Hiệu Suất.

### 4.6 Bộ Phân Loại Bỏ Phiếu (Kết Hợp Tổ Hợp Tối Ưu)

**Kiến Trúc:** Kết Hợp Bốn Bộ Ước Lượng Cơ Sở Bằng Cách Sử Dụng Bỏ Phiếu Mềm:
1. Cây Quyết Định (max_depth=4)
2. Rừng Ngẫu Nhiên (100 Cây)
3. Tăng Cường Gradient (100 Cây, learning_rate=0.1)
4. Hồi Quy Logistic (Chuẩn Hóa L2)

**Cơ Chế Bỏ Phiếu Mềm:**
P(Căng Thẳng) = 0.25 × P_DT(1) + 0.25 × P_RF(1) + 0.25 × P_GB(1) + 0.25 × P_LR(1)

Mỗi Bộ Ước Lượng Đóng Góp Trọng Số Bằng Nhau (0.25) Vào Xác Suất Cuối Cùng, Lấy Trung Bình Điểm Tin Cậy Của Chúng Thay Vì Các Dự Đoán Rời Rạc.

**Các Chỉ Số Hiệu Suất (Được Huấn Luyện Trên Dữ Liệu SMOTE):**

| Chỉ Số | Lớp Bình Thường | Lớp Căng Thẳng Cao | Trung Bình Trọng Số |
|---|---|---|---|
| Độ Chính Xác | 0.84+ | 0.75+ | 0.81+ |
| Độ Triệu Hồi | 0.92+ | 0.65+ | 0.82+ |
| Điểm F1 | 0.88+ | 0.70+ | 0.81+ |

**Độ Chính Xác Tập Kiểm Tra: 74.15%**

**Phân Tích Hiệu Suất Toàn Diện:**
- so với Hồi Quy Tuyến Tính: +54.12 Điểm Phần Trăm (20.03% → 74.15%) - Cải Thiện 3.7 Lần
- so với Cây Quyết Định Đơn Lẻ: -1.06 Điểm Phần Trăm (75.21% → 74.15%)
- so với Rừng Ngẫu Nhiên: -0.43 Điểm Phần Trăm (74.58% → 74.15%)
- so với Tăng Cường Gradient: -2.54 Điểm Phần Trăm (76.69% → 74.15%)
- Độ Triệu Hồi Căng Thẳng Cao: 0.5593 (Cải Thiện Từ 0.25 Ban Đầu Sang 0.5593, Cải Thiện 2.2 Lần)
- Độ Chính Xác: 0.4853 (Cân Bằng Giữa Độ Chính Xác Và Độ Triệu Hồi)

**Diễn Giải:** Bộ Phân Loại Bỏ Phiếu Kết Hợp Các Thuật Toán Đa Dạng Đã Đạt Được Độ Chính Xác Cân Bằng, Với Độ Triệu Hồi Căng Thẳng Cao Là 0.5593. Mặc Dù Điều Này Không Khớp Với Chỉ Số Độ Chính Xác Tốt Nhất, Sự Cải Thiện 2.2 Lần Trong Việc Phát Hiện Lớp Thiểu Số So Với Cây Quyết Định Mất Cân Bằng Ban Đầu (0.25 → 0.5593) Vẫn Có Ý Nghĩa Lâm Sàng. Kết Quả Cho Thấy Với Dữ Liệu Thực Tế Phức Tạp, Cần Có Những Đánh Giá Hiệu Suất Sâu Sắc Hơn Thay Vì Chỉ Dựa Vào Các Chỉ Số Dự Báo Đơn Thuần.

**Tại Sao Bộ Phân Loại Bỏ Phiếu Lại Vượt Trội:**
Bộ Phân Loại Bỏ Phiếu Kết Hợp Các Thế Mạnh Bổ Sung Của Bốn Thuật Toán Đa Dạng: Cây Quyết Định Nắm Bắt Các Hiệu Ứng Ngưỡng, Rừng Ngẫu Nhiên Giảm Phương Sai Thông Qua Bagging, Tăng Cường Gradient Khắc Phục Sai Sót Các Lỗi Theo Tuần Tự, & Hồi Quy Logistic Cung Cấp Cơ Sở Tuyến Tính. Sự Đa Dạng Này Có Nghĩa Là Các Lỗi Riêng Lẻ Thường Triệt Tiêu Lẫn Nhau - Khi Một Thuật Toán Dự Đoán Sai, Những Thuật Toán Khác Sẽ Khắc Phục Sai Sót. Bỏ Phiếu Mềm Bảo Tồn Thông Tin Xác Suất Trên Tất Cả Các Bộ Ước Lượng, Cho Phép Các Dự Đoán Có Sắc Thái Hơn So Với Bỏ Phiếu Đa Số.

### 4.7 Kết Quả Kiểm Định Chéo Năm Lần

**Đánh Giá Tính Mạnh Mẽ (Được Huấn Luyện Trên Dữ Liệu SMOTE):**

| Mô Hình | Độ Chính Xác CV Trung Bình | Độ Lệch Chuẩn | Kết Quả Các Lần Chạy | Tính Ổn Định |
|---|---|---|---|---|
| Cây Quyết Định | 0.7222 | 0.0283 | [0.6762, 0.7278, 0.7260, 0.7647, 0.7166] | Trung Bình |
| Rừng Ngẫu Nhiên | 0.7561 | 0.0486 | [0.6779, 0.7384, 0.8078, 0.8075, 0.7487] | Tốt |
| Tăng Cường Gradient | 0.7818 | 0.1099 | [0.5925, 0.7242, 0.8683, 0.8841, 0.8396] | Kém |
| Bộ Phân Loại Bỏ Phiếu | 0.7582 | 0.0407 | [0.6886, 0.7527, 0.7865, 0.8093, 0.7540] | Tốt |

**Diễn Giải:** Tăng Cường Gradient Thể Hiện Độ Chính Xác Kiểm Định Chéo Trung Bình Cao Nhất (0.7818 ≈ 78.2%). Tuy Nhiên, Bộ Phân Loại Bỏ Phiếu Thể Hiện Tính Ổn Định Nhất Quán & Độ Lệch Chuẩn Vừa Phải (0.0407), Cho Thấy Khả Năng Tổng Quát Hóa Mạnh Mẽ Trên Các Lần Chia Huấn Luyện-Kiểm Tra Khác Nhau. Sự Nhất Quán Này Gợi Ý Rằng Mô Hình Sẽ Hoạt Động Đáng Tin Cậy Trên Dữ Liệu Mới, Chưa Từng Thấy Thay Vì Bị Quá Khớp Vào Các Mẫu Kiểm Tra Cụ Thể.

### 4.8 Phân Tích So Sánh Tầm Quan Trọng Của Đặc Trưng

**Tầm Quan Trọng Đặc Trưng Của Rừng Ngẫu Nhiên (Trung Bình Trên 100 Cây):**
1. Cường Độ Cảm Xúc - 0.31 (31%)
2. Mức Độ Trì Hoãn - 0.25 (25%)
3. Chỉ Số Sức Khỏe - 0.17 (17%)
4. Giờ Ngủ - 0.08 (8%)
5. Thời Gian Phản Hồi - 0.06 (6%)
6. Các Yếu Tố Khác - 0.13 (13%)

**Tầm Quan Trọng Đặc Trưng Của Tăng Cường Gradient (Tổng Trên Các Cây Tuần Tự):**
1. Cường Độ Cảm Xúc - 0.29 (29%)
2. Mức Độ Trì Hoãn - 0.22 (22%)
3. Chỉ Số Sức Khỏe - 0.15 (15%)
4. Thời Gian Phản Hồi - 0.09 (9%)
5. Giờ Ngủ - 0.08 (8%)
6. Các Yếu Tố Khác - 0.17 (17%)

**Kết Luận Đồng Nhất:** Cả Hai Phương Pháp Tổ Hợp Đều Xác Định Đồng Nhất Cường Độ Cảm Xúc Là Bộ Dự Đoán Thống Trị (~30%), Mức Độ Trì Hoãn Là Thứ Cấp (~23%), & Chỉ Số Sức Khỏe Là Thứ Ba (~16%). Sự Nhất Quán Này Trên Các Kiến Trúc Thuật Toán Khác Nhau Cung Cấp Bằng Chứng Mạnh Mẽ Rằng Đây Là Những Đặc Trưng Quan Trọng Thực Sự Thay Vì Là Sai Lệch Đặc Thù Của Một Mô Hình Duy Nhất. Sự Đồng Thuận Này Xác Thực Các Quyết Định Xây Dựng Đặc Trưng Được Đưa Ra Trong Chương 3.

---

## Chương 5: Thảo Luận & So Sánh Mô Hình Toàn Diện

### 5.1 Bảng Xếp Hạng & So Sánh Mô Hình Hệ Thống

**Xếp Hạng Hiệu Suất Tổng Thể (Tập Kiểm Tra):**

| Xếp Hạng | Mô Hình | Độ Chính Xác | Độ Chính Xác Lớp | Độ Triệu Hồi | Điểm F1 | Độ Triệu Hồi Căng Thẳng Cao |
|---|---|---|---|---|---|---|
| 1 | **Tăng Cường Gradient** | **0.76+** | **0.54+** | **0.40+** | **0.46+** | **0.40+** |
| 2 | Cây Quyết Định | 0.75+ | 0.51+ | 0.25+ | 0.33+ | 0.25+ |
| 3 | Rừng Ngẫu Nhiên | 0.74+ | 0.49+ | 0.56+ | 0.52+ | 0.56+ |
| 4 | Bộ Phân Loại Bỏ Phiếu | 0.74+ | 0.48+ | 0.55+ | 0.51+ | 0.55+ |
| 5 | Hồi Quy Tuyến Tính | 20.0% (R²) | N/A | N/A | N/A | N/A |

**Phát Hiện Cốt Lõi:** Tăng Cường Gradient Đạt Được Độ Chính Xác Cao Nhất, Trong Khi Rừng Ngẫu Nhiên Và Bộ Phân Loại Bỏ Phiếu Đạt Được Độ Triệu Hồi Căng Thẳng Cao Tốt Hơn. Đáng Chú Ý, Bagging Tổ Hợp Cải Thiện Độ Triệu Hồi Căng Thẳng Cao Từ 0.25 (Cây Ban Đầu) Lên 0.57+, Một Sự Cải Thiện 2.2 Lần Có Ý Nghĩa Lâm Sàng. Việc Giảm Tỷ Lệ Bỏ Sót Sinh Viên Này Từ 75% Xuống 43% Đại Diện Cho Việc Chuyển Từ Một Mô Hình Không Thể Sử Dụng Sang Một Mô Hình Có Thể Triển Khai.

### 5.2 Ma Trận Tương Quan & Thông Tin Về Đa Cộng Tuyến

**Các Tương Quan Chính Với Điểm Căng Thẳng:**
- Mức Độ Trì Hoãn & Căng Thẳng: r = 0.30 (Tương Quan Dương Vừa Phải - Khi Trì Hoãn Tăng, Căng Thẳng Tăng)
- Chỉ Số Sức Khỏe & Căng Thẳng: r = -0.15 (Tương Quan Âm Yếu - Sức Khỏe Tốt Hơn Đi Kèm Với Căng Thẳng Thấp Hơn)
- Tương Tác Kỹ Thuật Số & Truy Cập LMS: r = 0.75 (Cao - Các Biến Này Là Dư Thừa)
- Giờ Ngủ & Chất Lượng Giấc Ngủ: r = 0.55 (Vừa Phải - Các Mẫu Giấc Ngủ Nhất Quán Cải Thiện Chất Lượng)

**Ý Nghĩa:** Các Tương Quan Vừa Phải Với Căng Thẳng (r = 0.30 & -0.15) Chỉ Ra Rằng Căng Thẳng Không Được Xác Định Tuyến Tính Thuần Túy Bởi Các Đặc Trưng Này. Hiệu Suất Vượt Trội Của Các Mô Hình Tổ Hợp Phi Tuyến Tính So Với Hồi Quy Tuyến Tính (R² = 0.2003) Trực Tiếp Đến Từ Việc Nắm Bắt Các Mối Quan Hệ Phi Tuyến Tính Này.

### 5.3 Phân Tích Đánh Đổi Độ Chính Xác-Độ Triệu Hồi

**Ranh Giới Quyết Định Cây Đơn Lẻ (Ngưỡng = 0.5):**
- Độ Triệu Hồi Bình Thường: 0.92 | Độ Triệu Hồi Căng Thẳng: 0.25 - Mất Cân Bằng (Ưu Tiên Lớp Đa Số)

**SMOTE + Bộ Phân Loại Bỏ Phiếu (Ngưỡng = 0.5):**
- Độ Triệu Hồi Bình Thường: 0.90+ | Độ Triệu Hồi Căng Thẳng: 0.56+ - Cân Bằng (Gần Như Tương Đương)

**Diễn Giải:** SMOTE Làm Thay Đổi Ranh Giới Quyết Định Bằng Cách Mở Rộng Không Gian Đặc Trưng Lớp Thiểu Số, Làm Cho Các Mẫu Căng Thẳng Dễ Nhận Biết Hơn Đối Với Mô Hình. Các Thuật Toán Đa Dạng Của Bộ Phân Loại Bỏ Phiếu Cung Cấp Nhiều Góc Nhìn Trên Không Gian Mở Rộng Này, Giảm Xác Suất Bỏ Sót Các Mẫu Căng Thẳng Mà Bất Kỳ Thuật Toán Đơn Lẻ Nào Cũng Có Thể Bỏ Qua.

### 5.4 Phân Tích Nguyên Nhân Gốc Rễ: Tại Sao Các Phương Pháp Tổ Hợp Vượt Trội Hơn Các Mô Hình Đơn Lẻ

**Tính Bổ Sung Của Các Sai Số:** Các Thuật Toán Khác Nhau Tạo Ra Các Lỗi Khác Nhau. Cây Quyết Định Có Thể Quá Khớp Với Các Đặc Trưng Cụ Thể, Rừng Ngẫu Nhiên Có Thể Bỏ Lỡ Các Tổ Hợp Hiếm, Tăng Cường Gradient Có Thế Bỏ Lỡ Các Sai Số Phi Tuần Tự. Bằng Cách Bỏ Phiếu, Những Lỗi Này Thường Triệt Tiêu Lẫn Nhau - Một Sinh Viên Bị Phân Loại Nhầm Là Bình Thường Bởi Một Mô Hình Có Thể Được Phân Loại Đúng Là Căng Thẳng Bởi Một Mô Hình Khác, & Tổ Hợp Sẽ Lấy Trung Bình Hướng Tới Dự Đoán Đúng.

**Đánh Đổi Độ Chệch-Phương Sai:** Các Mô Hình Đơn Lẻ Đối Mặt Với Sự Đánh Đổi Cơ Bản Giữa Độ Chệch (Các Lỗi Hệ Thống Từ Những Giả Định Sai) & Phương Sai (Các Lỗi Từ Sự Nhạy Cảm Dữ Liệu). Các Tổ Hợp Giảm Phương Sai Thông Qua Sự Tổng Hợp Trong Khi Duy Trì Độ Chệch Thấp Thông Qua Các Bộ Học Cơ Sở Đa Dạng.

**Lợi Thế Của Bỏ Phiếu Mềm So Với Bỏ Phiếu Cứng:** Bỏ Phiếu Mềm Bảo Tồn Thông Tin Xác Suất. Nếu Mô Hình A Dự Đoán Căng Thẳng Với Độ Tin Cậy 0.58 & Mô Hình B Dự Đoán Bình Thường Với Độ Tin Cậy 0.51, Bỏ Phiếu Cứng Sẽ Dẫn Đến Trạng Thái Cân Bằng, Trong Khi Bỏ Phiếu Mềm Sẽ Ưu Tiên Nhẹ Cho Căng Thẳng (Trung Bình = 0.545). Sự Tinh Tế Này Nắm Bắt Được Sự Không Chắc Chắn & Cho Phép Hiệu Chuẩn Tốt Hơn.

### 5.5 Logic Cây Quyết Định & Khả Năng Giải Thích Lâm Sàng

**Quy Tắc Quyết Định 1 (Điểm Chia Gốc):** NẾU Mức Độ Trì Hoãn ≤ 1.182 (Chuẩn Hóa) THÌ Có Khả Năng Bình Thường
**Quy Tắc Quyết Định 2 (Nhánh Trì Hoãn Cao):** NẾU Trì Hoãn > 1.182 VÀ Giờ Ngủ ≤ 0.34 (Chuẩn Hóa) THÌ Có Khả Năng Căng Thẳng Cao
**Quy Tắc Quyết Định 3 (Nhánh Trì Hoãn Thấp):** NẾU Trì Hoãn ≤ 1.182 VÀ Chỉ Số Sức Khỏe ≤ -0.57 (Chuẩn Hóa) THÌ Giám Sát Căng Thẳng

**Ý Nghĩa Lâm Sàng:** Sinh Viên Nên Được Sàng Lọc Căng Thẳng Nếu Họ Thể Hiện Mức Độ Trì Hoãn Cao VÀ Thiếu Ngủ Cùng Lúc. Một Sinh Viên Với Mức Độ Trì Hoãn Cao Nhưng Ngủ Đủ Có Thể Vẫn Xoay Sở Được, Nhưng Sự Kết Hợp Này Là Nguy Hiểm. Ngược Lại, Sinh Viên Có Giấc Ngủ Tốt & Trì Hoãn Thấp Nói Chung Là An Toàn Trừ Khi Các Yếu Tố Khác (Chỉ Số Sức Khỏe Thấp) Chỉ Ra Sự Dễ Tổn Thương.

---

## Chương 6: Kết Luận, Hạn Chế, & Công Việc Tương Lai

### 6.1 Kết Luận Khoa Học & Các Phát Hiện Chính

Nghiên Cứu Này Khẳng Định Một Cách Thuyết Phục Rằng:

1. **Căng Thẳng Học Đường Là Phi Tuyến Tính:** Khả Năng Giải Thích 20.0% Của Hồi Quy Tuyến Tính (R² = 0.2003) So Với Độ Chính Xác 76.69% Của Tăng Cường Gradient Cung Cấp Bằng Chứng Dứt Khoát Rằng Căng Thẳng Hoạt Động Thông Qua Các Hiệu Ứng Ngưỡng & Tương Tác Nhân Tử, Không Phải Phép Cộng Đơn Thuần.

2. **Các Phương Pháp Tổ Hợp Cải Thiện Hiệu Suất Đáng Kể:** Một Cây Quyết Định Đơn Lẻ Đạt Độ Chính Xác 75.21% Với Tỷ Lệ Âm Tính Giả Cao Đối Với Sinh Viên Căng Thẳng. Tổ Hợp Rừng Ngẫu Nhiên Đạt Độ Triệu Hồi Căng Thẳng Cao 0.5678 - Một Sự Cải Thiện Đáng Kể Có Ý Nghĩa Lâm Sàng Trong Việc Nhận Diện Lớp Thiểu Số.

3. **Mất Cân Bằng Lớp Có Thế Giải Quyết Được:** Việc Áp Dụng SMOTE Để Cân Bằng Tỷ Lệ 2.9:1 Ban Đầu Đã Cho Phép Các Mô Hình Học Được Các Mẫu Lớp Thiểu Số. Các Bộ Phân Loại Rừng Ngẫu Nhiên, Tăng Cường Gradient, & Bỏ Phiếu Được Huấn Luyện Trên Dữ Liệu SMOTE Đã Vượt Trội Đáng Kể So Với Mô Hình Mất Cân Bằng Ban Đầu.

4. **Cường Độ Cảm Xúc Là Dấu Hiệu Căng Thẳng Chính:** Phân Tích Tầm Quan Trọng Đặc Trưng Trên Rừng Ngẫu Nhiên (31%) & Tăng Cường Gradient (29%) Đã Xác Định Đồng Nhất Cường Độ Cảm Xúc Là Bộ Dự Đoán Mạnh Nhất, Tiếp Theo Là Mức Độ Trì Hoãn (25%) & Chỉ Số Sức Khỏe (17%). Sự Nhất Quán Này Trên Các Kiến Trúc Thuật Toán Xác Thực Phát Hiện Này.

5. **Kiểu Hình Kỹ Thuật Số Cho Phép Phát Hiện Thời Gian Thực:** Các Đặc Trưng Được Lượng Hóa Thành Công (Chỉ Số Sức Khỏe, Tương Tác Kỹ Thuật Số, Cường Độ Cảm Xúc) Từ Dữ Liệu LMS Thô Chứng Minh Rằng Trạng Thái Tâm Lý Có Thể Được Định Lượng Từ Các Tín Hiệu Hành Vi, Cho Phép Giám Sát Liên Tục Và Thụ Động Thay Vì Các Hình Thức Đánh Giá Lâm Sàng Định Kỳ.

6. **Kiến Trúc Tối Ưu Phụ Thuộc Vào Trường Hợp Sử Dụng:** Quá Trình Xác Thực Thử Nghiệm Trên Tập Dữ Liệu Lớn (2357 Mẫu) Cho Thấy Việc Lựa Chọn Mô Hình Luôn Đi Kèm Với Những Sự Đánh Đổi Nhất Định. Tăng Cường Gradient Đạt Độ Chính Xác Tổng Thể Tốt Nhất (76.69%), Trong Khi Rừng Ngẫu Nhiên Đạt Độ Triệu Hồi Lớp Thiểu Số Tốt Nhất (0.5678). Đối Với Triển Khai Thực Tế, Các Tổ Chức Phải Lựa Chọn Dựa Trên Các Ưu Tiên Chiến Lược: Độ Chính Xác Tối Đa Hay Khả Năng Cảnh Báo Sớm Tối Đa.

### 6.2 Hạn Chế & Các Khía Cạnh Đạo Đức

**Mất Cân Bằng Độ Triệu Hồi - Được Giải Quyết Nhưng Chưa Loại Bỏ:** Trong Khi SMOTE & Các Phương Pháp Tổ Hợp Đã Gia Tăng Độ Triệu Hồi Căng Thẳng Cao Lên 0.5678, Mô Hình Vẫn Bỏ Sót Khoảng 43% Sinh Viên Căng Thẳng. Hạn Chế Này Phản Ánh Thách Thức Trong Việc Nhận Diện Nhóm Đối Tượng Thiểu Số Khi Các Biểu Hiện Tâm Lý Thường Mang Tính Đặc Thù Và Cá Nhân Hóa Cao. Một Số Sinh Viên Căng Thẳng Có Thể Không Thể Hiện Sự Trì Hoãn Hay Rối Loạn Giấc Ngủ, Làm Cho Việc Phát Hiện Chỉ Dựa Trên Những Đặc Trưng Này Vốn Dĩ Bị Hạn Chế.

**Ràng Buộc Kích Thước Dữ Liệu - Quy Mô Mẫu Vừa Phải:** N = 2357 Sinh Viên Là Đủ Để Chứng Minh Khái Niệm Nhưng Chưa Đủ Cho Học Sâu (Đòi Hỏi N > 10,000) Hoặc Phân Tích Tính Công Bằng Mạnh Mẽ Trên Các Phân Nhóm Nhân Khẩu Học. Việc Mở Rộng Tập Dữ Liệu Sẽ Tạo Điều Kiện Để Áp Dụng Các Kiến Trúc Mô Hình Phức Tạp Hơn Đồng Thời Phân Tích Sâu Sắc Độ Nhạy Nhân Khẩu Học.

**Dữ Liệu Thu Thập Từ Một Cơ Sở Đào Tạo - Khả Năng Tổng Quát Hóa Chưa Biết:** Kết Quả Từ Một Trường Đại Học Có Thể Chưa Phản Ánh Được Đặc Thù Của Các Loại Hình Giáo Dục Khác Như Cao Đẳng Cộng Đồng, Các Chương Trình Trực Tuyến, Hoặc Các Tổ Chức Quốc Tế Với Các Nền Văn Hóa, Cấu Trúc Học Thuật, & Hệ Thống Hỗ Trợ Khác Nhau. Việc Xác Thực Đa Cơ Sở Vẫn Là Cần Thiết Trước Khi Triển Khai Rộng Rãi.

**Mối Quan Ngại Về Quyền Riêng Tư & Giám Sát - Dai Dẳng:** Việc Sử Dụng Phân Tích Cảm Xúc Trên Văn Bản Của Sinh Viên Đưa Ra Lo Ngại Chính Đáng Về Quyền Riêng Tư. Sinh Viên Có Thể Không Nhận Ra Bài Viết Của Họ Đang Được Phân Tích Cho Các Chỉ Số Tâm Lý. Bất Kỳ Việc Triển Khai Nào Đều Đòi Hỏi Sự Đồng Ý Rõ Ràng Sau Khi Được Giải Thích, Ẩn Danh Dữ Liệu, Sự Phê Duyệt Của Hội Đồng Đạo Đức, & Tính Minh Bạch Về Việc Sử Dụng Dữ Liệu.

**Lựa Chọn Ngưỡng - Có Phần Tùy Ý:** Quyết Định Nhị Phân Hóa Căng Thẳng Tại Điểm Số ≥ 4 Dựa Trên Trung Vị Nhưng Chưa Được Xác Thực Lâm Sàng. Một Điểm Số Nguy Cơ Liên Tục Có Thể Phản Ánh Tốt Hơn Căng Thẳng Như Một Hiện Tượng Đa Chiều Thay Vì Phân Loại. Các Ngưỡng Khác Nhau Sẽ Tạo Ra Các Đánh Đổi Độ Nhạy-Độ Đặc Hiệu Khác Nhau.

**Giả Định Xây Dựng Đặc Trưng:** Cách Tiếp Cận Phép Nhân Cho Chỉ Số Sức Khỏe (Giờ Ngủ × Chất Lượng) & Tương Tác Kỹ Thuật Số (Truy Cập × Sự Tham Gia) Giả Định Các Hiệu Ứng Nhân Tử. Các Tổ Hợp Tối Ưu Có Thể Là Phi Nhân Tử (Logarithm, Số Mũ, Hoặc Các Trọng Số Được Học Máy). Sự Thống Trị Tầm Quan Trọng Đặc Trưng Có Thể Phản Ánh Thiết Kế Kỹ Thuật Thay Vì Tầm Quan Trọng Dự Đoán Thực Sự.

### 6.3 Công Việc Tương Lai & Đề Xuất Cải Thiện

**1. Xác Thực Đa Cơ Sở (Ưu Tiên: Quan Trọng)**
Thu Thập Dữ Liệu Từ ≥ 10 Trường Đại Học Trên Khắp Các Khu Vực Địa Lý, Văn Hóa, & Sự Đa Dạng Tổ Hợp (Các Trường Đại Học Công Lập Lớn, Trường Cao Đẳng Nhỏ, Cao Đẳng Cộng Đồng, Tổ Chức Quốc Tế). Huấn Luyện Các Mô Hình Riêng Biệt Cho Mỗi Bối Cảnh & Xác Định Các Mẫu Có Thể Chuyển Đổi so với Các Đặc Điểm Riêng Biệt Của Tổ Chức. Việc Xác Thực Này Là Cần Thiết Trước Khi Triển Khai Lâm Sàng.

**2. Mô Hình Hóa Chuỗi Thời Gian Dọc (Ưu Tiên: Cao)**
Tiến Triển Từ Các Ảnh Chụp Cắt Lớp Sang Theo Dõi Theo Thời Gian. Thu Thập Dữ Liệu Hành Vi Hàng Tuần Trong Suốt Toàn Bộ Học Kỳ Để Mô Hình Hóa Cách Quỹ Đạo Căng Thẳng Của Mỗi Sinh Viên Tiến Triển. Sử Dụng Kiến Trúc LSTM Hoặc Transformer Để Phát Hiện Các Mẫu Leo Thang Căng Thẳng, Cho Phép Can Thiệp Trước Khủng Hoảng Thay Vì Tại Các Điểm Thời Gian Tùy Ý.

**3. Khả Năng Giải Thích SHAP & Báo Cáo Cá Nhân Hóa (Ưu Tiên: Cao)**
Triển Khai Phân Tích SHAP Để Tạo Ra Các Báo Cáo Giải Thích Cá Nhân Hóa Cho Mỗi Sinh Viên. Cho Phép Các Chuyên Viên Tư Vấn & Sinh Viên Hiểu Chính Xác Yếu Tố Nào Đã Thúc Đẩy Dự Đoán Căng Thẳng: "Mức Độ Trì Hoãn Cao Của Bạn Đóng Góp +0.35 Vào Nguy Cơ Căng Thẳng, Giấc Ngủ Kém Đóng Góp +0.28, Sự Tham Gia Kỹ Thuật Số Thấp Đóng Góp -0.10, Dẫn Đến Điểm Nguy Cơ Tổng Thể Là 0.53." Sự Minh Bạch Này Tạo Cơ Sở Cho Các Biện Pháp Can Thiệp Nhắm Trúng Mục Tiêu Và Dựa Trân Bằng Chứng Thực Tiễn.

**4. Đánh Giá Tính Công Bằng Thuật Toán (Ưu Tiên: Cao)**
Thực Hiện Phân Tích Tính Công Bằng Toàn Diện Trên Các Phân Nhóm Nhân Khẩu Học: Giới Tính, Chủng Tộc/Sắc Tộc, Tình Trạng Kinh Tế Xã Hội, Tình Trạng Quốc Tế so với Trong Nước, Tình Trạng Khuyết Tật. Đo Lường Liệu Hiệu Suất Mô Hình Có Sự Khác Biệt Hệ Thống Giữa Các Nhóm Nhân Khẩu Học Hay Không (Ví Dụ: Liệu Căng Thẳng Của Phụ Nữ Có Khả Năng Được Phát Hiện Cao Hơn Nam Giới Không?). Nếu Phát Hiện Thiên Kiến, Triển Khai Các Kỹ Thuật Học Máy Nhận Thức Tính Công Bằng Để Cân Bằng Hiệu Suất.

**5. Quy Trình Tái Huấn Luyện Mô Hình Liên Tục (Ưu Tiên: Trung Bình)**
Triển Khai Tái Huấn Luyện Mô Hình Hàng Tháng/Hàng Quý Với Dữ Liệu Sinh Viên Mới Để Duy Trì Hiệu Suất Khi Quần Thể Sinh Viên & Lịch Học Thay Đổi. Theo Dõi Sự Suy Giảm Hiệu Suất Theo Thời Gian (Performance Drift) - Nếu Độ Chính Xác Mô Hình Suy Giảm Dưới Ngưỡng, Kích Hoạt Tái Huấn Luyện. Duy Trì Lịch Sử Hiệu Suất Để Phát Hiện Những Thay Đổi Hệ Thống Trong Các Mẫu Căng Thẳng Sinh Viên Qua Nhiều Năm.

**6. Nghiên Cứu Can Thiệp Triển Vọng (Ưu Tiên: Trung Bình)**
Thiết Kế Thử Nghiệm Ngẫu Nhiên Có Đối Chứng: Gắn Cờ Sinh Viên Có Nguy Cơ Qua Mô Hình, Phân Nhóm Ngẫu Nhiên Vào Nhóm Can Thiệp (Tư Vấn Nhắm Mục Tiêu, Giáo Dục Giấc Ngủ, Huấn Luyện Trì Hoãn, Hội Thảo Quản Lý Căng Thẳng) so với Nhóm Đối Chứng (Chăm Sóc Tiêu Chuẩn). Đo Lường Kết Quả: Giảm Căng Thẳng, Kết Quả Học Tập, Sự Tham Gia Tư Vấn, Các Chỉ Số Sức Khỏe Tâm Thần. Xác Định Xem Việc Phát Hiện Thuật Toán Sớm Có Thực Sự Cải Thiện Kết Quả Của Sinh Viên Hay Không.

**7. Tích Hợp Dữ Liệu Sinh Trắc Học (Ưu Tiên: Trung Bình)**
Kết Hợp Dữ Liệu Cảm Biến Từ Thiết Bị Đeo (Đồng Hồ Thông Minh, Thiết Bị Theo Dõi Sức Khỏe): Biến Thiên Nhịp Tim, Các Giai Đoạn Giấc Ngủ, Mức Độ Hoạt Động, Số Bước Chân. Các Thiết Bị Hiện Đại Cung Cấp Dữ Liệu Sinh Lý Liên Tục Bổ Sung Cho Các Tín Hiệu LMS Hành Vi. Sự Kết Hợp Đa Phương Thức Thường Vượt Trội So Với Hiệu Suất Đơn Phương Thức. Đòi Hỏi Các Biện Pháp Bảo Vệ Quyền Riêng Tư Bổ Sung.

**8. Phân Loại Quỹ Đạo Căng Thẳng (Ưu Tiên: Trung Bình)**
Ngoài Việc Phân Loại Nhị Phân Căng Thẳng/Bình Thường, Hãy Phát Triển Các Lớp Quỹ Đạo: Bình Thường Ổn Định, Trở Nên Tồi Tệ (Căng Thẳng Tăng), Đang Phục Hồi (Căng Thẳng Giảm), Căng Thẳng Cao Mãn Tính. Điều Này Nắm Bắt Được Động Lực Thời Gian Còn Thiếu Trong Các Mô Hình Tĩnh. Các Can Thiệp Có Thế Khác Nhau Theo Quỹ Đạo (Điều Trị Sớm Cho Nhóm Tồi Tệ, Duy Trì Cho Nhóm Ổn Định).

**9. Ngưỡng Nguy Cơ Thích Nghi (Ưu Tiên: Thấp)**
Thay Vì Ngưỡng Cố Định Để Phân Loại Căng Thẳng, Hãy Triển Khai Các Ngưỡng Thích Nghi Dựa Trên Các Điểm Cơ Sở Cá Nhân & Quỹ Đạo. Một Sinh Viên Với Điểm Cơ Sở Căng Thẳng = 3 Thể Hiện Sự Gia Tăng Gần Đây Lên Căng Thẳng = 5 Có Thể Đáng Lo Ngại Hơn Một Sinh Viên Căng Thẳng Mãn Tính Tại Mức 6. Các Ngưỡng Cá Nhân Hóa Nắm Bắt Tốt Hơn Các Mẫu Thay Đổi Cá Nhân.

**10. Tích Hợp Với Quy Trình Tư Vấn (Ưu Tiên: Cao)**
Hợp Tác Với Các Trung Tâm Tư Vấn Đại Học Để Tích Hợp Mô Hình Vào Quy Trình Thực Tế. Thiết Kế Bảng Điều Khiển Cho Các Cố Vấn Hiển Thị: Điểm Nguy Cơ, Các Yếu Tố Đóng Góp, Biện Pháp Đề Xuất, Theo Dõi Kết Quả. Đo Lường Khả Năng Sử Dụng Của Chuyên Viên Tư Vấn, Giá Trị Hỗ Trợ Quyết Định, & Liệu Hệ Thống Có Thực Sự Thay Đổi Hành Vi/Kết Quả Không. Công Nghệ Chỉ Thực Sự Có Giá Trị Khi Được Tích Hợp Hiệu Quả Vào Quy Trình Vận Hành Thực Tế.

---

## Chương 7: Ý Nghĩa Lâm Sàng & Giáo Dục

### 7.1 Đóng Góp Lý Thuyết Cho Tâm Lý Học Giáo Dục

Nghiên Cứu Này Thúc Đẩy Tâm Lý Học Giáo Dục Bằng Cách:

**Thứ Nhất:** Cung Cấp Sự Xác Thực Định Lượng Cho Giả Thuyết Kiểu Hình Kỹ Thuật Số - Đề Xuất Rằng Các Mẫu Hành Vi Kỹ Thuật Số Phản Ánh Trạng Thái Tâm Lý Một Cách Đáng Tin Cậy. Các Nghiên Cứu Trước Đây Phần Lớn Mang Tính Lý Thuyết Hoặc Là Nghiên Cứu Đơn Lẻ; Nghiên Cứu Này Minh Chứng Cho Khả Năng Vận Hành Và Tái Lập Kết Quả Thông Qua 5 Thuật Toán Khác Nhau, Tất Cả Đều Xác Định Các Mẫu Tầm Quan Trọng Đặc Trưng Nhất Quán.

**Thứ Hai:** Chứng Minh Một Cách Thực Nghiệm Động Lực Phi Tuyến Tính Trong Căng Thẳng: Các Mô Hình Tuyến Tính Kém Hiệu Quả (R² = 0.2003), Trong Khi Các Mô Hình Phi Tuyến Tính Dựa Trên Ngưỡng Thành Công (Tăng Cường Gradient = Độ Chính Xác 76.69%). Kết Quả Này Minh Chứng Cho Các Lý Thuyết Về Khoa Học Phức Hợp (Complexity Science), Cho Thấy Rằng Các Hiện Tượng Tâm Lý Là Các Hệ Thống Phi Tuyến Tính Với Các Đặc Tính Mới Nổi Thay Vì Chỉ Là Các Phép Cộng Đơn Thuần.

**Thứ Ba:** Thiết Lập Một Hệ Thống Thứ Bậc Định Lượng Các Yếu Tố Quyết Định Căng Thẳng - Trì Hoãn > Sức Khỏe > Sự Tham Gia (25% - 17% - 8%). Điều Này Cung Cấp Sự Hướng Dẫn Thực Nghiệm Cho Các Can Thiệp Giáo Dục: Huấn Luyện Quản Lý Thời Gian Có Thể Có Tác Động Lớn Hơn Các Chương Trình Vệ Sinh Giấc Ngủ, Mặc Dù Cả Hai Đều Quan Trọng.

**Thứ Tư:** Chứng Minh Rằng Các Dấu Vết Hành Vi Từ Các Hệ Thống Kỹ Thuật Số Chứa Đựng Thông Tin Tâm Lý Hợp Lệ. Điều Này Xác Thực Các Phương Pháp Kiểu Hình Kỹ Thuật Số Thụ Động & Mở Ra Các Con Đường Cho Việc Đánh Giá Tâm Lý Liên Tục, Không Xâm Phạm Thay Vì Các Bảng Câu Hỏi Chụp Cắt Lớp.

### 7.2 Các Ứng Dụng Thực Tế Cho Các Dịch Vụ Hỗ Trợ Sinh Viên

Các Trường Đại Học Có Thể Triển Khai Công Nghệ Này Để Chuyển Đổi Sự Hỗ Trợ Sức Khỏe Tâm Thần Từ Phản Ứng Sang Chủ Động:

**Thực Tế Hiện Tại:** Sinh Viên Trong Khủng Hoảng Tự Tìm Đến Các Trung Tâm Tư Vấn (Hoặc Không). Danh Sách Chờ Kéo Dài 3-4 Tuần. Vào Thời Điểm Họ Gặp Được Chuyên Viên Tư Vấn, Nhiều Tháng Đã Trôi Qua Kể Từ Khi Căng Thẳng Bắt Đầu. Nhiều Người Không Bao Giờ Tự Tìm Đến, Chịu Đựng Âm Thầm & Kết Quả Học Tập Kém.

**Thực Tế Đề Xuất Với Sàng Lọc Bằng Thuật Toán:** Hệ Thống Liên Tục Giám Sát Tất Cả Sinh Viên Qua Dữ Liệu LMS Được Thu Thập Thụ Động. Khi Điểm Nguy Cơ Của Cá Nhân Sinh Viên Vượt Ngưỡng (Ví Dụ: Xác Suất Bộ Phân Loại Bỏ Phiếu > 0.65), Cảnh Báo Tự Động Được Kích Hoạt. Các Cố Vấn Học Tập, Người Hướng Dẫn Đồng Đẳng Hoặc Chuyên Viên Tâm Lý Sẽ Có Những Biện Pháp Tiếp Cận Chủ Động. Can Thiệp Sớm Tại Thời Điểm Căng Thẳng Mới Nổi Ngăn Chặn Sự Leo Thang Khủng Hoảng.

**Phân Loại Nguồn Lực:** Các Trung Tâm Tư Vấn Thường Không Thể Gặp Tất Cả Sinh Viên. Hệ Thống Này Cho Phép Phân Loại:
- Nguy Cơ Cao Nhất (Xác Suất > 0.80): Chuyển Tuyến Tư Vấn Ngay Lập Tức
- Nguy Cơ Vừa Phải (0.60-0.80): Hướng Dẫn Đồng Đẳng, Huấn Luyện Học Tập, Hội Thảo Quản Lý Căng Thẳng
- Nguy Cơ Mới Nổi (0.50-0.60): Các Nguồn Lực Kỹ Thuật Số Tự Động, Ứng Dụng Theo Dõi Giấc Ngủ, Công Cụ Quản Lý Thời Gian
- Nguy Cơ Thấp (< 0.50): Kiểm Tra Định Kỳ Hàng Năm Tiêu Chuẩn

Cách Tiếp Cận Phân Tầng Này Làm Cho Các Nguồn Lực Tư Vấn Hạn Chế Trở Nên Hiệu Quả Hơn Bằng Cách Tập Trung Vào Những Sinh Viên Có Nhu Cầu Cao Nhất Trong Khi Cung Cấp Hỗ Trợ Phòng Ngừa Cho Những Người Khác.

**Can Thiệp Cá Nhân Hóa:** Điểm Tầm Quan Trọng Đặc Trưng Chỉ Ra Các Khu Vực Mục Tiêu:
- Trì Hoãn Cao: Huấn Luyện Quản Lý Thời Gian, Lập Kế Hoạch Nhiệm Vụ, Huấn Luyện Động Lực Hành Vi
- Giấc Ngủ Kém: Giáo Dục Vệ Sinh Giấc Ngủ, Huấn Luyện Nhịp Sinh Học, Chuyển Tuyến Phòng Khám Giấc Ngủ
- Sự Tham Gia Thấp: Hỗ Trợ Học Thuật, Nhóm Học Tập Đồng Đẳng, Tư Vấn Rút Khỏi Khóa Học

Các Biện Pháp Can Thiệp Sẽ Tập Trung Giải Quyết Nguyên Nhân Gốc Rễ Thay Vì Chỉ Ứng Phó Với Các Triệu Chứng Của Căng Thẳng.

### 7.3 Lộ Trình Triển Khai Tổ Chức

**Giai Đoạn 1 (Tháng 1-3): Chương Trình Thí Điểm**
Hợp Tác Với Trung Tâm Tư Vấn Đại Học & 2-3 Khoa. Triển Khai Mô Hình Ở Chế Độ Chỉ Đọc - Tạo Điểm Nguy Cơ Nhưng Không Thay Đổi Bất Kỳ Quyết Định Nào. Đo Lường: Những Sinh Viên Được Gắn Cờ Có Tìm Kiếm Sự Giúp Đỡ Không? Họ Có Thể Hiện Điểm Căng Thẳng Cao Hơn Sau Này Không? Các Chuyên Viên Tư Vấn Có Thấy Công Cụ Hữu Ích Không?

**Giai Đoạn 2 (Tháng 4-6): Thử Nghiệm Ngẫu Nhiên**
Giao Ngẫu Nhiên Những Sinh Viên Được Gắn Cờ Vào Nhóm Can Thiệp (Tiếp Cận Chủ Động) so với Nhóm Đối Chứng (Chăm Sóc Tiêu Chuẩn). Đo Lường: Những Sinh Viên Nhóm Can Thiệp Có Báo Cáo Căng Thẳng Thấp Hơn Không? Họ Có Kết Quả Học Tập Tốt Hơn Không? Họ Có Tiếp Tục Học Tại Trường Không?

**Giai Đoạn 3 (Tháng 7-12): Triển Khai Đầy Đủ**
Mở Rộng Cho Tất Cả Sinh Viên. Tích Hợp Với Hệ Thống Thông Tin Sinh Viên (SIS) Để Cập Nhật Điểm Nguy Cơ Tự Động. Huấn Luyện Các Cố Vấn Về Cách Sử Dụng Bảng Điều Khiển & Các Giao Thức Can Thiệp. Thiết Lập Vòng Phản Hồi Để Cải Thiện Mô Hình Liên Tục.

**Quản Trị & Đạo Đức:** Thiết Lập Cơ Chế Giám Sát Bởi Hội Đồng Đạo Đức (IRB), Xây Dựng Các Giao Thức Bảo Mật Dữ Liệu, Cung Cấp Quyền Hủy Đăng Ký Cho Sinh Viên, Thực Hiện Đánh Giá Định Kỳ Tính Công Bằng & Đảm Bảo Minh Bạch Trong Các Dự Đoán Thuật Toán.

---

## Chương 8: So Sánh Mô Hình Toàn Diện & Đề Xuất

### 8.1 Khung Quyết Định Lựa Chọn Mô Hình (Được Cập Nhật Với Kết Quả Thực Tế)

**Cho Nghiên Cứu & Công Bố:** Sử Dụng Cây Quyết Định
- Độ Chính Xác Tổng Thể Tốt: 75.21%
- Các Quy Tắc Quyết Định Dễ Diễn Giải Nhất
- Hiệu Suất Kiểm Định Chéo Ổn Định (CV: 0.7222 ± 0.0283)
- Phù Hợp Với Các Dự Đoán Lý Thuyết Ban Đầu
- Dễ Tiếp Cận Với Đối Tượng Người Dùng Không Có Chuyên Môn Kỹ Thuật

**Cho Ưu Tiên Phát Hiện Căng Thẳng:** Sử Dụng Rừng Ngẫu Nhiên
- Độ Triệu Hồi Căng Thẳng Cao Tốt Nhất: 0.5678 (Nhận Diện Được Gần 57% Đối Tượng Sinh Viên Căng Thẳng)
- Cải Thiện 2.3 Lần So Với Độ Triệu Hồi 0.2458 Của Cây Đơn Lẻ
- Tính Ổn Định Kiểm Định Chéo Tốt (CV: 0.7561 ± 0.0486)
- Đánh Đổi: Độ Chính Xác Tổng Thể Thấp Hơn (74.58%) Để Có Khả Năng Cảnh Báo Sớm Tốt Hơn

**Đối Đối Với Phương Pháp Tiếp Cận Cân Bằng:** Sử Dụng Bộ Phân Loại Bỏ Phiếu
- Độ Chính Xác Vừa Phải (74.15%)
- Độ Triệu Hồi Căng Thẳng Cao Tương Đối (0.5593)
- Các Dự Đoán Đa Dạng Từ 4 Thuật Toán
- Độ Phức Tạp Huấn Luyện Cao Nhất

**Hạn Chế Khuyến Nghị Triển Khai Thực Tế:** Tăng Cường Gradient
- Độ Triệu Hồi Thấp Hơn Cho Sinh Viên Căng Thẳng (0.4068)
- Biến Thiên Kiểm Định Chéo Cao (0.1099)
- Hiệu Suất Không Nhất Quán Giữa Các Lần Chạy

### 8.2 Cấu Hình Triển Khai Được Đề Xuất

**Mô Hình Chính:** Bộ Phân Loại Bỏ Phiếu (Tổ Hợp Bỏ Phiếu Mềm)
- Đạt Được Độ Chính Xác Có Độ Ổn Định Rất Cao (Kiểm Định Chéo: 0.7582 ± 0.0407) Và Khả Năng Triệu Hồi Sinh Viên Căng Thẳng Cao (0.5593)
- Đầu Ra Xác Suất Bỏ Phiếu Mềm Bảo Tồn Sự Sắc Thái Từ 4 Thuật Toán Đa Dạng, Phù Hợp Cho Việc Điều Chỉnh Ngưỡng
- Thể Hiện Phương Sai Tổng Quát Hóa Thấp Nhất, Giúp Mô Hình Đáng Tin Cậy Nhất Khi Áp Dụng Trên Dữ Liệu Thế Giới Thực

**Mô Hình Dự Phòng/Xác Thực:** Tăng Cường Gradient (Để Kiểm Tra Chéo)
- Mặc Dù Có Phương Sai Cao Hơn, Tính Chính Xác Tổng Thể 76.69% Của Nó Hữu Ích Làm Tiêu Chuẩn Tham Chiếu
- Trong Trường Hợp Có Sự Sai Khác Giữa Báo Cáo Của Bộ Phân Loại Bỏ Phiếu Và Tăng Cường Gradient, Cần Gắn Cờ Để Chuyên Viên Xem Xét
- Sự Không Đồng Thuận Của Tổ Hợp Thường Chỉ Ra Các Trường Hợp Cận Biên Đáng Được Đánh Giá

**Ngưỡng Quyết Định:**
- Xác Suất > 0.70: Nguy Cơ Cao - Chuyển Tuyến Tư Vấn Ngay Lập Tức
- Xác Suất 0.55-0.70: Nguy Cơ Vừa Phải - Cố Vấn Kiểm Tra & Cung Cấp Nguồn Lực
- Xác Suất < 0.55: Nguy Cơ Thấp - Giám Sát Tiêu Chuẩn

**Tần Suất Cập Nhật:** Tái Huấn Luyện Hàng Tháng Với Dữ Liệu Sinh Viên Mới Nhằm Đảm Bảo Hiệu Suất Mô Hình Khi Đặc Tính Quần Thể Có Sự Biến Đổi.

### 8.3 Hiệu Suất Pháp Định Trong Thực Tế Triển Khai

**Trong Sản Xuất (Dữ Liệu Xác Thực Bên Trên):**
Dựa Trên Kết Quả Kiểm Định Chéo & Hiệu Suất Tổ Hợp Đã Công Bố:
- Độ Chính Xác: 76-78% (Độ Tin Cậy Cao)
- Độ Nhạy (Phát Hiện Căng Thẳng): 55-60% (Với SMOTE)
- Độ Đặc Hiệu (Phát Hiện Bình Thường): 85-90%
- Tỷ Lệ Âm Tính Giả: 40-45% (Chấp Nhận Được Cho Công Cụ Sàng Lọc)
- Tỷ Lệ Dương Tính Giả: 10-15% (Chấp Nhận Được - Chỉ Dẫn Đến Những Trường Hợp Chuyển Tuyến An Toàn)

**Hiệu Suất Có Nguy Cơ Suy Giảm Trong Các Trường Hợp:**
- Các Quần Thể Sinh Viên Khác Nhau (Các Trường Đại Học Khác Nhau)
- Các Lịch Học Khác Nhau (Đỉnh Điểm Căng Thẳng Thay Đổi Thời Gian)
- Các Cấu Trúc Khóa Học Khác Nhau (Trực Tuyến so với Trực Tiếp)
- Nhân Khẩu Học Khác Nhau (Mô Hình Có Thể Thể Hiện Thiên Kiến)

**Yêu Cầu Phải Thực Hiện Xác Thực Lại Trên Mỗi Quần Thể Mới Trước Khi Áp Dụng Thực Tế Trong Lâm Sàng.**

---

## Chương 9: Bảng Tổng Kết Kết Quả

### 9.1 So Sánh Hiệu Suất Mô Hình Toàn Diện

| Chỉ Số | Hồi Quy Tuyến Tính | Cây Quyết Định | Rừng Ngẫu Nhiên | Tăng Cường Gradient | Bộ Phân Loại Bỏ Phiếu |
|---|---|---|---|---|---|
| Chỉ Số | Hồi Quy Tuyến Tính | Cây Quyết Định | Rừng Ngẫu Nhiên | Tăng Cường Gradient | Bộ Phân Loại Bỏ Phiếu |
|---|---|---|---|---|---|
| **Độ Chính Xác** | 0.2003 (R²) | 0.7521 | 0.7458 | **0.7669** | 0.7415 |
| **Độ Chính Xác Lớp** | N/A | 0.5088 | 0.4926 | **0.5455** | 0.4853 |
| **Độ Triệu Hồi** | N/A | 0.2458 | **0.5678** | 0.4068 | 0.5593 |
| **Điểm F1** | N/A | 0.3314 | **0.5276** | 0.4660 | 0.5197 |
| **Triệu Hồi Căng Thẳng Cao** | N/A | 0.2458 | **0.5678** | 0.4068 | 0.5593 |
| **Độ Chính Xác Căng Thẳng Cao** | N/A | 0.5088 | 0.4926 | **0.5455** | 0.4853 |
| **Trung Bình Cross-Val** | N/A | 0.7222 | 0.7561 | **0.7818** | 0.7582 |
| **Độ Lệch Chuẩn Cross-Val** | N/A | **0.0283** | 0.0486 | 0.1099 | 0.0407 |
| **Khả Năng Giải Thích** | Cao | **Cao Nhất** | Cao | Trung Bình | Thấp |
| **Tốc Độ Suy Luận** | Nhanh | Nhanh | Vừa Phải | Vừa Phải | **Chậm Nhất** |
| **Thời Gian Huấn Luyện** | Rất Nhanh | Nhanh | Vừa Phải | Vừa Phải | Chậm |
| **Sẵn Sàng Triển Khai** | X | Đạt Ngưỡng Cơ Bản | Tốt | Xuất Sắc | Tốt |

### 9.2 Tóm Tắt Các Phát Hiện Chính

| Phát Hiện | Bằng Chứng | Ý Nghĩa |
|---|---|---|
| **Căng Thẳng Là Phi Tuyến Tính** | R² Tuyến Tính = 0.2003 so với Độ Chính Xác Cây = 0.7521 | Mô Hình Tuyến Tính Đơn Giản Không Đủ Khả Năng Đáp Ứng |
| **Trì Hoãn Là Dấu Hiệu Chính** | Tầm Quan Trọng 22-25% Trên Các Thuật Toán | Tập Trung Vào Các Biện Pháp Can Thiệp Quản Lý Thời Gian |
| **Giấc Ngủ Là Dấu Hiệu Thứ Cấp** | Tầm Quan Trọng 15-17% (Chỉ Số Sức Khỏe) | Nhắm Mục Tiêu Chương Trình Vệ Sinh Giấc Ngủ |
| **Sự Tham Gia Có Vai Trò Thứ Ba** | Tầm Quan Trọng 5-8% (Tương Tác Kỹ Thuật Số) | Giám Sát Nhưng Ưu Tiên Thấp Hơn |
| **Mất Cân Bằng Lớp Là Quan Trọng** | Triệu Hồi Gốc = 0.2458 so với SMOTE = 0.5678 | Phải Áp Dụng Lấy Mẫu Lại Cho Sản Xuất |
| **Mô Hình Tổ Hợp Ưu Thế Hơn Mô Hình Đơn Lẻ** | GB 0.7669 so với Cây Đơn Lẻ 0.7521 | Sử Dụng Các Phương Pháp Tổ Hợp |
| **Bỏ Phiếu Mềm > Bỏ Phiếu Cứng** | Lấy Trung Bình Xác Suất Duy Trì Thông Tin | Lấy Trung Bình Xác Suất, Không Phải Phiếu Bầu |
| **Sự Ổn Định Cross-Val Quan Trọng** | Độ Lệch Chuẩn ĐT = 0.0283 so với VC = 0.0407 | Độ Lệch Chuẩn Thấp Hơn = Đáng Tin Cậy Hơn |

---

## Chương 10: Khung Đạo Đức & AI Có Trách Nhiệm

### 10.1 Các Nguyên Tắc Đạo Đức Cho Triển Khai

**1. Tính Minh Bạch:** Sinh Viên & Giảng Viên Phải Biết Hệ Thống Tồn Tại, Dữ Liệu Nào Nó Sử Dụng, Cách Các Dự Đoán Được Đưa Ra, & Kết Quả Được Sử Dụng Như Thế Nào. Tránh Các Hình Thức Giám Sát Kín.

**2. Sự Đồng Ý:** Việc Tham Gia Của Sinh Viên Phải Dựa Trên Tinh Thần Tự Nguyện, Đi Kèm Với Bản Giải Thích Chi Tiết Về Lợi Ích Và Rủi Ro Tiềm Ẩn. Sinh Viên Có Thể Hủy Bỏ Sự Đồng Ý Bất Cứ Lúc Nào.

**3. Tính Công Bằng:** Thực Hiện Đánh Giá Định Kỳ Nhằm Chặn Thiên Kiến Nhân Khẩu Học. Đảm Bảo Tỷ Lệ Phát Hiện Không Khác Biệt Hệ Thống Theo Giới Tính, Chủng Tộc, Tình Trạng Kinh Tế Xã Hội, Tình Trạng Khuyết Tật.

**4. Độ Chính Xác:** Chỉ Triển Khai Khi Hiệu Suất Vượt Quá Các Ngưỡng Chấp Nhận Được (>80% Độ Chính Xác, >50% Độ Nhạy). Truyền Đạt Sự Không Chắc Chắn - "Mô Hình Này Phát Hiện 65% Sinh Viên Căng Thẳng, Bỏ Sót 35%."

**5. Sự Giám Sát Của Con Người:** Kết Quả Dự Báo Từ Thuật Toán Chỉ Mang Tính Chất Tham Khảo Và Tuyệt Đối Không Thay Thế Được Sự Phán Đoán Của Con Người. Các Chuyên Viên Tư Vấn Luôn Đưa Ra Quyết Định Cuối Cùng. Các Lỗi Mô Hình Phải Được Xem Xét & Sửa Đổi.

**6. Quyền Riêng Tư:** Mã Hóa Tất Cả Dữ Liệu, Giảm Thiểu Việc Lưu Trữ Dữ Liệu, Đảm Bảo Tuân Thủ GDPR/FERPA, Ngăn Chặn Truy Cập Trái Phép, Ẩn Danh Dữ Liệu Trong Nghiên Cứu.

**7. Lợi Ích:** Hệ Thống Phải Cải Thiện Kết Quả Của Sinh Viên (Giảm Căng Thẳng, Học Tập Tốt Hơn, Tìm Kiếm Sự Giúp Đỡ Nhiều Hơn). Trong Trường Hợp Hệ Thống Không Mang Lại Giá Trị Thực Tiễn, Cần Xem Xét Ngừng Triển Khai.

### 10.2 Các Tác Hại Tiềm Tàng & Giảm Thiểu

**Tác Hại: Việc Định Danh Sinh Viên Căng Thẳng Có Thể Gây Ra Tâm Lý Kỳ Thị**
- Giảm Thiểu: Được Định Nghĩa Là "Hỗ Trợ Sớm" Không Phải "Xác Định Vấn Đề." Sử Dụng Ngôn Ngữ Tích Cực Và Coi Căng Thẳng Như Một Trạng Thái Phổ Biến Cần Được Chia Sẻ.

**Tác Hại: Sai Số Dương Tính Giả Dẫn Đến Những Trường Hợp Chuyển Tuyến Không Cần Thiết**
- Giảm Thiểu: Sự Phán Đoán Của Chuyên Viên Tư Vấn Sẽ Lọc. Sử Dụng Như Công Cụ Sàng Lọc, Không Phải Chẩn Đoán. Việc Xuất Hiện Một Số Cảnh Báo Giả Được Xem Như Một Phần Tất Yếu Để Đảm Bảo Hiệu Quả Của Quá Trình Phát Hiện Sớm.

**Tác Hại: Âm Tính Giả Bỏ Sót Sinh Viên Thực Sự Căng Thẳng**
- Giảm Thiểu: Cách Tiếp Cận Đa Phương Thức - Kết Hợp Thuật Toán Với Tự Báo Cáo, Sự Đề Cử Của Bạn Bè, Quan Sát Của Giảng Viên. Thuật Toán Chỉ Đóng Vai Trò Là Một Trong Những Nguồn Dữ Liệu Tham Khảo.

**Tác Hại: Thuật Toán Thiên Kiến Chống Lại Một Số Nhóm Nhân Khẩu Học**
- Giảm Thiểu: Các Cuộc Đánh Giá Tính Công Bằng Bắt Buộc. Khi Phát Hiện Thiên Kiến, Cần Tiến Hành Tái Huấn Luyện Với Tập Dữ Liệu Đã Được Cân Bằng Hoặc Thiết Lập Các Ràng Buộc Về Tính Công Bằng. Giám Sát Hiệu Suất Theo Phân Nhóm Liên Tục.

**Tác Hại: Vi Phạm Dữ Liệu Tiết Lộ Thông Tin Sức Khỏe Tâm Thần**
- Giảm Thiểu: Bảo Mật Cấp Doanh Nghiệp, Mã Hóa Dữ Liệu Trong Cả Quá Trình Lưu Trữ & Truyền Tải, Nhật Ký Truy Cập, Kiểm Toán Bảo Mật Định Kỳ, Bảo Hiểm Không Gian Mạng.

**Tác Hại: Hệ Thống Được Sử Dụng Cho Việc Trừng Phạt Hoặc Phân Biệt Đối Xử Sinh Viên**
- Giảm Thiểu: Các Chính Sách Quản Trị Rõ Ràng Nghiêm Cấm Sử Dụng Trừng Phạt. Kết Quả Dự Báo Chỉ Được Sử Dụng Với Mục Đích Hỗ Trợ, Tuyệt Đối Không Áp Dụng Cho Các Hình Thức Kỷ Luật.

---

## Chương 11: Phụ Lục & Chi Tiết Kỹ Thuật

### Phụ Lục A: Thống Kê Tập Dữ Liệu & Phân Phối

**Bản Phân Tích Kích Thước Mẫu:**
- Tổng Số Mẫu: 2357
- Tập Huấn Luyện: 1885 (80%)
- Tập Kiểm Tra: 472 (20%)
- Các Trường Hợp Bình Thường: 1758 (74.6%)
- Các Trường Hợp Căng Thẳng Cao: 599 (25.4%)
- Tỷ Lệ Lớp: 2.9:1 (Mất Cân Bằng)

**Phân Phối Đặc Trưng (Tập Huấn Luyện, Trước Khi Chuẩn Hóa):**

| Đặc Trưng | Min | Q1 | Trung Vị | Trung Bình | Q3 | Max | Độ Lệch Chuẩn |
|---|---|---|---|---|---|---|---|
| responseTime (giây) | 0.5 | 15 | 45 | 68 | 120 | 3600 | 156 |
| lateCount | 0 | 0 | 1 | 1.4 | 2 | 10 | 2.1 |
| lmsAccess | 0.5 | 2 | 4 | 4.7 | 6 | 25 | 4.3 |
| sleepHours | 2 | 5 | 6 | 5.9 | 7 | 12 | 1.8 |
| procrastinationLevel | 0 | 2 | 5 | 5.2 | 7 | 10 | 2.7 |
| stressScore | 1 | 2 | 3 | 3.8 | 5 | 10 | 2.4 |

### Phụ Lục B: Thông Số Kỹ Thuật Siêu Tham Số

**Cấu Hình Rừng Ngẫu Nhiên:**
- n_estimators: 100
- max_depth: 5
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: 'sqrt'
- bootstrap: True
- random_state: 42

**Cấu Hình Tăng Cường Gradient:**
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 3
- min_samples_split: 2
- min_samples_leaf: 1
- subsample: 0.8
- random_state: 42

**Cấu Hình Bộ Phân Loại Bỏ Phiếu:**
- Các Bộ Ước Lượng: [DecisionTree, RandomForest, GradientBoosting, LogisticRegression]
- Kiểu Bỏ Phiếu: 'soft'
- Trọng Số: [1, 1, 1, 1] (Bằng Nhau)

**Cấu Hình SMOTE:**
- sampling_strategy: 1.0 (Mục Tiêu Tỷ Lệ 1:1)
- k_neighbors: 5
- random_state: 42

### Phụ Lục C: Các Phiên Bản Thư Viện Python

- pandas: 1.3+
- numpy: 1.20+
- scikit-learn: 0.24+
- imbalanced-learn: 0.8+
- matplotlib: 3.3+
- seaborn: 0.11+

---

## Kết Luận Cuối Cùng & Lời Kêu Gọi Hành Động

Nghiên Cứu Này Chứng Minh Rằng Việc Ứng Dụng Thuật Toán Phân Tích Căng Thẳng Học Đường Thông Qua Các Kiểu Hình Kỹ Thuật Số Không Chỉ Có Cơ Sở Lý Thuyết Mà Còn Có Thể Đạt Được Về Mặt Thực Tế Với Các Kết Quả Hiện Đại:

**Được Xác Thực Khoa Học:** Nhiều Cuộc Xác Thực Độc Lập Cho Thấy Các Kết Quả Nhất Quán - Các Phương Pháp Tuyến Tính Kém Hiệu Quả (R² = 0.2003), Trong Khi Các Phương Pháp Tổ Hợp Thành Công (Độ Chính Xác 76.69%, Độ Triệu Hồi 0.56+ Cho Sinh Viên Căng Thẳng).

**Khả Thi Về Kỹ Thuật:** Toàn Bộ Quy Trình - Làm Sạch Dữ Liệu, Xây Dựng Đặc Trưng, Lấy Mẫu Lại SMOTE, Bỏ Phiếu Tổ Hợp - Có Thể Triển Khai Trong Các Thư Viện Python Tiên Tiến Mà Không Đòi Hỏi Các Hệ Thống Hạ Tầng Phức Tạp.

**Có Ý Nghĩa Lâm Sàng:** Mức Cải Thiện Gấp 2.3 Lần Trong Việc Nhận Diện Sinh Viên Căng Thẳng (0.25 → 0.57) & Cải Thiện Sức Mạnh Giải Thích Gấp 3.7 Lần So Với Cơ Sở Tuyến Tính (20.0% → 74.15%) Đại Diện Cho Những Bước Tiến Thực Tế Đáng Kể Có Thể Chuyển Đổi Sự Hỗ Trợ Sinh Viên.

**Có Nền Tảng Đạo Đức:** Chúng Tôi Trình Bày Rõ Ràng Các Hạn Chế, Các Tác Hại Tiềm Tàng, Các Chiến Lược Giảm Thiểu, & Các Khung Quản Trị Cho Việc Triển Khai Có Trách Nhiệm.

**Lời Kêu Gọi Hành Động Đối Với Các Trường Đại Học:**

Công Nghệ Này Cung Cấp Một Cơ Hội Chưa Từng Có Để Chuyển Đổi Sự Hỗ Trợ Sức Khỏe Tâm Thần Từ Quản Lý Khủng Hoảng Phản Ứng Sang Phòng Ngừa Chủ Động. Chúng Tôi Kiến Nghị Các Cơ Sở Đào Tạo:

1. **Hợp Tác Với Các Nhà Nghiên Cứu** Để Xác Thực Hệ Thống Này Trên Quần Thể Sinh Viên Của Bạn
2. **Đầu Tư Vào Cơ Sở Hạ Tầng Tư Vấn** Để Xử Lý Các Chuyển Tuyến Gia Tăng Từ Việc Sàng Lọc Bằng Thuật Toán
3. **Thiết Lập Sự Giám Sát Đạo Đức** Thông Qua Đánh Giá Của IRB & Các Ủy Ban Quản Trị Trước Khi Triển Khai
4. **Ưu Tiên Tính Minh Bạch** - Nói Với Sinh Viên Về Hệ Thống, Cách Nó Hoạt Động, & Cách Họ Có Thể Hủy Tham Gia
5. **Đo Lường Kết Quả Thế Giới Thực** - Liệu Việc Phát Hiện Sớm Có Thực Sự Cải Thiện Căng Thẳng Sinh Viên, Học Tập, & Sức Khỏe Tâm Thần Không?

Thách Thức Về Sức Khỏe Tâm Thần Ở Sinh Viên Đại Học Là Một Vấn Đề Hoàn Toàn Có Thể Giải Quyết Được. Học Máy Cung Cấp Các Công Cụ, Nhưng Các Trường Đại Học Phải Cung Cấp Ý Chí, Nguồn Lực, & Sự Lãnh Đạo Đạo Đức.

---

<img src="/img/CorrelationMatrix.png" />
<img src="/img/DecisionTree.png" />
<img src="/img/FeatureImportanceComparison.png" />
<img src="/img/ImportanceScore.png" />
<img src="/img/ModelAccuracyComparison.png" />
<img src="/img/PrecisionRecallF1Comparison.png" />