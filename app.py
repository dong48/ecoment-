import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Cấu hình giao diện đồ án
st.set_page_config(page_title="Đồ án: Dự báo doanh số Walmart", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('walmart_gb_model.pkl')

model = load_model()

# 2. Xây dựng sidebar nhập liệu đầu vào
st.sidebar.title("Thông số đầu vào")
store_id = st.sidebar.selectbox("Mã cửa hàng (Store):", range(1, 46))
dept_id = st.sidebar.selectbox("Mã phòng ban (Dept):", range(1, 100))
is_holiday = st.sidebar.radio("Tuần bắt đầu có ngày lễ?", [0, 1], format_func=lambda x: "Có" if x==1 else "Không")

# 3. Giao diện hiển thị chính
st.write(f"### Kết quả dự báo chuỗi cung ứng - Cửa hàng {store_id} | Phòng ban {dept_id}")
st.write("Vui lòng thiết lập các tham số cơ sở để thực hiện dự báo cho 4 tuần tiếp theo.")

col1, col2 = st.columns(2)
with col1:
    lag_4_sales = st.number_input("Doanh số cơ sở 4 tuần trước (USD):", min_value=0.0, value=25000.0)
    temperature = st.number_input("Nhiệt độ dự báo (F):", value=65.0)
with col2:
    size = st.number_input("Diện tích cửa hàng (Sq.ft):", value=150000)
    has_markdown = st.selectbox("Áp dụng chương trình khuyến mãi?", [0, 1], format_func=lambda x: "Có" if x==1 else "Không")

# 4. Xử lý suy luận toán học và in kết quả
if st.button("Chạy mô hình dự báo"):
    predictions = []
    weeks = ["Tuần 1", "Tuần 2", "Tuần 3", "Tuần 4"]
    
    # Sử dụng phương pháp lặp tự hồi quy (Recursive Forecasting)
    # Kết quả của tuần trước sẽ trở thành biến Lag của tuần sau
    current_lag = lag_4_sales
    
    for i in range(4):
        # Khởi tạo ma trận đặc trưng 20 biến (sinh viên dùng zero-padding cho các biến không thu thập được)
        input_features = np.zeros((1, 20)) 
        input_features[0, 0] = store_id
        input_features[0, 1] = dept_id
        input_features[0, 2] = is_holiday if i == 0 else 0  # Giả định chỉ có lễ ở tuần đầu tiên
        input_features[0, 3] = temperature
        input_features[0, 12] = size
        input_features[0, 13] = has_markdown
        input_features[0, 18] = current_lag  # Cập nhật biến Lag liên tục
        
        # Dự báo tuần hiện tại
        pred = model.predict(input_features)[0]
        pred = max(0.0, pred) # Loại bỏ doanh số âm
        predictions.append(pred)
        
        # Cập nhật doanh số dự báo làm đầu vào cho vòng lặp tuần tiếp theo
        current_lag = pred
        
    # Tạo Dataframe hiển thị kết quả 4 tuần
    df_results = pd.DataFrame({
        "Thời gian dự báo": weeks,
        "Doanh thu dự kiến (USD)": [f"{val:,.2f}" for val in predictions]
    })
    
    st.write("#### Bảng dự báo doanh thu 4 tuần tiếp theo")
    st.table(df_results)
    
    # Đánh giá tổng quan
    total_sales = sum(predictions)
    st.write(f"**Tổng doanh thu dự kiến (4 tuần):** {total_sales:,.2f} USD")
    
    # Khối logic đánh giá nghiệp vụ
    growth_rate = ((predictions[0] - lag_4_sales) / lag_4_sales) * 100
    
    st.write("#### Nhận xét:")
    if growth_rate > 20:
        st.write(f"- Tốc độ tăng trưởng tuần 1 dự kiến đạt {growth_rate:.1f}%.")
        st.write("- Khuyến nghị: Cần lên kế hoạch mở rộng tồn kho an toàn (Safety Stock) để tránh đứt gãy nguồn cung hàng hóa.")
    elif growth_rate < -20:
        st.write(f"- Sức mua tuần 1 dự kiến suy giảm {growth_rate:.1f}%.")
        st.write("- Khuyến nghị: Điều chỉnh giảm định mức nhập hàng, xem xét áp dụng thêm các chiến dịch giảm giá (MarkDown).")
    else:
        st.write(f"- Biên độ dao động doanh số ổn định ({growth_rate:.1f}%).")
        st.write("- Khuyến nghị: Duy trì mức vận hành và quy trình nhập hàng như hiện tại.")