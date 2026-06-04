import os
import sys
import requests
import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, HeatMapWithTime, DualMap
from streamlit_folium import st_folium

# LƯU Ý: BẮT BUỘC GIỮ NGUYÊN MÃ CODE SETUP MÔI TRƯỜNG SPARK
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
python_path = sys.executable.replace('\\', '/')
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
os.environ['PYSPARK_PIN_THREAD'] = 'true'

# Monkey-patch cho HeatMapWithTime để sửa lỗi get_bounds() (Bug thư viện Folium)
def patched_hmwt_get_self_bounds(self):
    from branca.utilities import none_min, none_max
    bounds = [[None, None], [None, None]]
    for step in self.data:
        for point in step:
            bounds = [
                [none_min(bounds[0][0], point[0]), none_min(bounds[0][1], point[1])],
                [none_max(bounds[1][0], point[0]), none_max(bounds[1][1], point[1])],
            ]
    return bounds
HeatMapWithTime._get_self_bounds = patched_hmwt_get_self_bounds

# Cấu hình giao diện Web
st.set_page_config(
    page_title="Dự Báo Tắc Nghẽn Giao Thông",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# QUẢN LÝ TRẠNG THÁI (SESSION STATE) & THEME SETTINGS
# ================================================================
if 'pickup' not in st.session_state: st.session_state.pickup = (40.7589, -73.9851) # Times Square
if 'dropoff' not in st.session_state: st.session_state.dropoff = (40.7812, -73.9665) # Central Park
# Tọa độ trung tâm để render bản đồ.
center_lat = (st.session_state.pickup[0] + st.session_state.dropoff[0]) / 2
center_lon = (st.session_state.pickup[1] + st.session_state.dropoff[1]) / 2
if 'click_step' not in st.session_state: st.session_state.click_step = 0
if 'show_results' not in st.session_state: st.session_state.show_results = False
if 'prediction_history' not in st.session_state: st.session_state.prediction_history = []
if 'theme' not in st.session_state: st.session_state.theme = "Cyberpunk"
if 'active_tab' not in st.session_state: st.session_state.active_tab = "🏠 Trang chủ"

# Định nghĩa 5 Chủ đề (CSS Variables) - Sử dụng Glassmorphism
themes_dict = {
    "Cyberpunk": {"bg": "linear-gradient(135deg, #0f0c29, #302b63, #24243e)", "text": "#00ffcc", "card": "rgba(0, 255, 204, 0.05)"},
    "Eco Green": {"bg": "linear-gradient(135deg, #d4fc79, #96e6a1)", "text": "#1b4332", "card": "rgba(255, 255, 255, 0.4)"},
    "Dark Mode": {"bg": "linear-gradient(160deg, #0a0a1a 0%, #111133 40%, #1a1a3e 70%, #0d0d24 100%)", "text": "#ffffff", "card": "rgba(255,255,255,0.05)"},
    "Light Mode": {"bg": "linear-gradient(160deg, #f8f9fa 0%, #e9ecef 100%)", "text": "#333333", "card": "rgba(0,0,0,0.05)"},
    "Ocean": {"bg": "linear-gradient(160deg, #004e92 0%, #000428 100%)", "text": "#ffffff", "card": "rgba(255,255,255,0.08)"}
}
current_theme = themes_dict[st.session_state.theme]

# Inject Custom CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    .stApp {{ background: {current_theme['bg']} !important; font-family: 'Inter', sans-serif; color: {current_theme['text']} !important; }}
    h1, h2, h3, h4, .stMarkdown p, .stMetricValue {{ color: {current_theme['text']} !important; }}

    /* Hiệu ứng Glassmorphism */
    .glass-card {{
        background: {current_theme['card']}; backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 20px; margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important;
        border: none !important; border-radius: 14px !important; font-weight: 700 !important;
        transition: all 0.3s ease !important; width: 100%; padding: 0.8rem !important;
    }}
    .stButton > button:hover {{ transform: translateY(-3px) scale(1.02) !important; box-shadow: 0 10px 20px rgba(0,0,0,0.2); }}
</style>
""", unsafe_allow_html=True)

# ================================================================
# TÍCH HỢP OSRM API - ĐẢM BẢO ĐƯỜNG UỐN LƯỢN THỰC TẾ
# ================================================================
@st.cache_data(ttl=3600)
def get_route_osrm(start_coords, end_coords, alternatives=True):
    alt_param = "true" if alternatives else "false"
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson&alternatives={alt_param}"
    try:
        r = requests.get(url)
        res = r.json()
        if res.get('code') == 'Ok':
            routes = []
            for route_info in res['routes']:
                route = [[p[1], p[0]] for p in route_info['geometry']['coordinates']]
                dist = route_info['distance'] / 1000.0
                routes.append((route, dist))
            return routes
    except Exception:
        pass
    return [([start_coords, end_coords], 0.0)]

# ================================================================
# KHỞI TẠO SPARK VÀ LOAD MODEL TỪ THƯ MỤC CỦA DỰ ÁN ĐÃ CHỈ ĐỊNH
# ================================================================
@st.cache_resource
def load_spark_and_model():
    try:
        import findspark
        findspark.init()
    except ImportError:
        pass
    from pyspark.sql import SparkSession
    from pyspark.ml.regression import GBTRegressionModel

    spark = None
    loaded_model = None

    try:
        spark = SparkSession.builder \
            .appName("Traffic_Web_App").master("local[1]") \
            .config("spark.driver.memory", "10g").config("spark.driver.bindAddress", "127.0.0.1").config("spark.driver.host", "127.0.0.1") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    except Exception:
        pass

    if spark:
        # Ưu tiên sử dụng model từ thư mục Google Drive của bạn
        model_paths = [
            "/content/drive/MyDrive/Du_bao_tac_duong_giao_thong_do_thi/Models/gbt_model",
            "Models/gbt_model"
        ]
        for path in model_paths:
            if os.path.exists(path):
                try:
                    loaded_model = GBTRegressionModel.load(path)
                    break
                except Exception:
                    continue

    return spark, loaded_model

spark, model = load_spark_and_model()

# XỬ LÝ LOGIC ĐÈN TÍN HIỆU THỰC TẾ (LIVE STATUS)
if spark is not None and model is not None:
    status_bg = "rgba(0,204,102,0.2)"
    status_color = "#00cc66"
    status_icon = "🟢"
    status_text = "Live: Spark Session Active & GBT Model Online"
elif spark is not None and model is None:
    status_bg = "rgba(255,75,75,0.2)"
    status_color = "#ff4b4b"
    status_icon = "🔴"
    status_text = "Cảnh báo: Spark Active nhưng Model Offline (Thiếu File)"
else:
    status_bg = "rgba(255,75,75,0.2)"
    status_color = "#ff4b4b"
    status_icon = "🔴"
    status_text = "Offline: Lỗi kết nối Spark Session"

# ================================================================
# HEADER TRANG WEB & NAVIGATION BAR
# ================================================================
st.markdown(f"""
<div style="text-align: center; margin-top: 10px;">
    <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="40" stroke="{current_theme['text']}" stroke-width="3" fill="none" opacity="0.6">
            <animate attributeName="r" from="20" to="45" dur="1.5s" repeatCount="indefinite"/>
            <animate attributeName="opacity" from="0.8" to="0" dur="1.5s" repeatCount="indefinite"/>
        </circle>
        <rect x="35" y="25" width="30" height="50" rx="5" fill="#222" stroke="{current_theme['text']}" stroke-width="2"/>
        <circle cx="50" cy="35" r="6" fill="#ff4b4b"><animate attributeName="opacity" values="1;0.2;1" dur="1s" repeatCount="indefinite"/></circle>
        <circle cx="50" cy="50" r="6" fill="#ffa600" />
        <circle cx="50" cy="65" r="6" fill="#00cc66" />
    </svg>
    <h1 style="text-shadow: 0 0 15px {current_theme['text']}; font-weight: 800; margin: 10px 0;">Hệ Thống Dự Báo Tắc Nghẽn Đô Thị AI</h1>
</div>
<div style="text-align:center; margin-bottom: 20px;">
    <span style="background: {status_bg}; color:{status_color}; padding: 5px 15px; border-radius: 20px; font-weight:bold; font-size: 0.9em; border: 1px solid {status_color};">
        {status_icon} {status_text}
    </span>
</div>
""", unsafe_allow_html=True)

# Các tùy chọn Menu
menu_options = ["🏠 Trang chủ", "📖 Tài liệu API OSRM", "📍 Chọn địa điểm (Khu 1)", "🧠 Phân tích & Dự đoán", "🕒 Lịch sử"]

# Chọn Navigation Bar
nav_choice = st.radio("Điều hướng:", menu_options, horizontal=True, label_visibility="collapsed", index=menu_options.index(st.session_state.active_tab))
st.session_state.active_tab = nav_choice

# ================================================================
# THIẾT LẬP SIDEBAR (THANH BÊN)
# ================================================================
with st.sidebar:
    st.header("⚙️ Cài đặt & Tác vụ")

    selected_theme = st.selectbox("🎨 Settings Theme", list(themes_dict.keys()), index=list(themes_dict.keys()).index(st.session_state.theme))
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.markdown("---")
    st.subheader("🕐 Thông số chuyến đi")
    month = st.selectbox("Tháng", range(1, 13), index=4)
    day_of_week = st.selectbox("Ngày trong tuần (1=CN, 2=T2...)", range(1, 8), index=1)
    hour = st.slider("Giờ khởi hành", 0, 23, 17)
    passenger_count = st.number_input("🧑‍🤝‍🧑 Số lượng hành khách", min_value=1, max_value=6, value=1)

    st.markdown("---")
    st.subheader("📍 Nhập tọa độ thủ công")

    def update_coords():
        st.session_state.pickup = (st.session_state.lat_a, st.session_state.lon_a)
        st.session_state.dropoff = (st.session_state.lat_b, st.session_state.lon_b)

    c1, c2 = st.columns(2)
    with c1: st.number_input("Vĩ độ (A)", value=st.session_state.pickup[0], format="%.4f", step=0.001, key="lat_a", on_change=update_coords)
    with c2: st.number_input("Kinh độ (A)", value=st.session_state.pickup[1], format="%.4f", step=0.001, key="lon_a", on_change=update_coords)

    c3, c4 = st.columns(2)
    with c3: st.number_input("Vĩ độ (B)", value=st.session_state.dropoff[0], format="%.4f", step=0.001, key="lat_b", on_change=update_coords)
    with c4: st.number_input("Kinh độ (B)", value=st.session_state.dropoff[1], format="%.4f", step=0.001, key="lon_b", on_change=update_coords)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Dự báo & gợi ý lộ trình")
    if predict_btn:
        st.session_state.show_results = True
        st.session_state.force_save = True
        # Ép chuyển trang sang Phân tích & Dự đoán
        st.session_state.active_tab = "🧠 Phân tích & Dự đoán"
        st.rerun()

center_lat = (st.session_state.pickup[0] + st.session_state.dropoff[0]) / 2
center_lon = (st.session_state.pickup[1] + st.session_state.dropoff[1]) / 2

# ================================================================
# QUẢN LÝ NỘI DUNG TỪNG TRANG QUA NAVIGATION BAR
# ================================================================
if st.session_state.active_tab == "🏠 Trang chủ":
    st.markdown(f"""
    <div class="glass-card">
        <h2>🚩 Giới thiệu chung</h2>
        <p>Chào mừng bạn đến với nền tảng dự báo giao thông thế hệ mới. Trong bối cảnh các siêu đô thị đang phải đối mặt với tình trạng ùn tắc ngày càng nghiêm trọng, việc biết trước tình trạng giao thông không chỉ là một tiện ích, mà là một nhu cầu thiết yếu để tối ưu hóa cuộc sống. Hệ thống của chúng tôi không chỉ dừng lại ở việc hiển thị bản đồ; chúng tôi sử dụng "bộ não" AI để phân tích và đưa ra những dự đoán có độ chính xác cao về thời gian và vận tốc di chuyển.</p>

        <h2>🎯 Mục đích của hệ thống</h2>
        <ul>
            <li><strong>Dự báo chủ động:</strong> Thay vì chỉ phản ứng với kẹt xe khi nó đã xảy ra, AI của chúng tôi phân tích các yếu tố như: thời gian trong ngày, ngày trong tuần, tháng trong năm và vị trí địa lý để đưa ra cảnh báo trước khi bạn bắt đầu hành trình.</li>
            <li><strong>Tối ưu hóa thời gian di chuyển (ETA):</strong> Sử dụng thuật toán học máy GBT (Gradient Boosted Trees) được huấn luyện trên hàng triệu bản ghi dữ liệu vận tải đô thị, giúp tính toán thời gian dự kiến một cách thực tế nhất, bao gồm cả các độ trễ tiềm ẩn do ùn tắc.</li>
            <li><strong>Hỗ trợ ra quyết định:</strong> Cung cấp cho người dùng cái nhìn trực quan qua bản đồ nhiệt (Heatmap) và các chỉ số KPI, giúp bạn quyết định nên khởi hành ngay hay chờ đợi thời điểm thông thoáng hơn.</li>
        </ul>

        <h2>🛡️ Nền tảng công nghệ</h2>
        <p>Chúng tôi tự hào ứng dụng các công nghệ hàng đầu thế giới:</p>
        <ul>
            <li><strong>Apache Spark (PySpark):</strong> Xử lý dữ liệu lớn (Big Data) với tốc độ vượt trội.</li>
            <li><strong>Machine Learning (GBT Model):</strong> Thuật toán tối ưu cho các bài toán hồi quy phức tạp về thời gian di chuyển.</li>
            <li><strong>OSRM Routing:</strong> Đảm bảo mọi lộ trình đều dựa trên mạng lưới đường bộ thực tế, không phải đường chim bay.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.active_tab == "📖 Tài liệu API OSRM":
    st.markdown(f"""
    <div class="glass-card">
        <h2>Tiện ÍCH API OSRM (OSRM Utilities)</h2>
        <h3>OSRM – Xương Sống Của Hệ Thống Định Tuyến Thực Tế</h3>

        <h4>🧠 OSRM là gì?</h4>
        <p>OSRM (Open Source Routing Machine) là một engine định tuyến hiệu năng cực cao, được thiết kế để tìm đường đi ngắn nhất hoặc nhanh nhất trong mạng lưới đường bộ của OpenStreetMap. Đây chính là công cụ giúp hệ thống của chúng tôi hiểu được cấu trúc "uốn lượn" của các tuyến phố đô thị.</p>

        <h4>🛠️ Tại sao chúng tôi sử dụng OSRM thay vì tính toán thông thường?</h4>
        <ul>
            <li><strong>Mạng lưới đường bộ thực:</strong> OSRM giúp chúng tôi lấy được quãng đường chính xác từng mét, đi qua từng ngã rẽ, cầu vượt và hầm chui.</li>
            <li><strong>Tọa độ hình học (Geometry):</strong> API cung cấp chuỗi tọa độ (Polyline) giúp vẽ đường đi uốn lượn mềm mại trên bản đồ Hybrid, mang lại trải nghiệm thị giác chuyên nghiệp cho người dùng.</li>
            <li><strong>Thuật toán Alternative Routes:</strong> Đây là tính năng đặc biệt mà chúng tôi tích hợp. Khi lộ trình chính được AI dự báo là "Đỏ" (Tắc nghẽn), API OSRM sẽ được kích hoạt để tìm kiếm các lộ trình phụ (đường nhánh) giúp bạn thoát khỏi vùng ùn tắc một cách nhanh chóng nhất.</li>
        </ul>

        <h4>🔗 Quy trình liên kết dữ liệu</h4>
        <ol>
            <li>Tọa độ được gửi đến OSRM Server.</li>
            <li>Server trả về Quãng đường thực tế và Mảng tọa độ.</li>
            <li>Các thông tin này được đẩy vào Model AI GBT để tính toán vận tốc dựa trên bối cảnh thời gian.</li>
        </ol>
        <p>Kết quả cuối cùng là một lộ trình thông minh, có tính toán đến từng khúc cua.</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.active_tab == "📍 Chọn địa điểm (Khu 1)":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("1️⃣ Khu vực 1: Bản đồ Tọa độ (Click để chọn Điểm A & B)")
    st.caption("Sau khi chọn xong, hãy nhấn nút '🚀 Dự báo & gợi ý lộ trình' ở thanh bên.")

    m_input = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google')
    folium.Marker(st.session_state.pickup, popup="📍 Điểm đón (A)", icon=folium.Icon(color="green", icon="play")).add_to(m_input)
    folium.Marker(st.session_state.dropoff, popup="🏁 Điểm trả (B)", icon=folium.Icon(color="red", icon="stop")).add_to(m_input)

    map_data = st_folium(m_input, height=500, use_container_width=True, returned_objects=["last_clicked"], key="input_map")
    if map_data and map_data.get("last_clicked"):
        clicked_coord = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
        if st.session_state.get('last_click') != clicked_coord:
            st.session_state.last_click = clicked_coord
            if st.session_state.click_step == 0:
                st.session_state.pickup = clicked_coord
                st.session_state.click_step = 1
            else:
                st.session_state.dropoff = clicked_coord
                st.session_state.click_step = 0
            st.session_state.show_results = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == "🧠 Phân tích & Dự đoán":
    if not st.session_state.show_results:
        st.info("👈 Vui lòng cấu hình tọa độ và nhấn nút **'Dự báo & gợi ý lộ trình'** ở Sidebar để xem kết quả phân tích.")
    else:
        # Lấy lộ trình OSRM thật
        routes_data = get_route_osrm(st.session_state.pickup, st.session_state.dropoff, alternatives=True)
        primary_route, primary_dist = routes_data[0]

        if model is None:
            st.error("⚠️ Lỗi: Không tìm thấy GBT Model tại thư mục `/content/drive/MyDrive/Du_bao_tac_duong_giao_thong_do_thi/Models/gbt_model`.")
        else:
            # --- Tính Feature đưa vào Model GBT ---
            lon1, lat1, lon2, lat2 = map(math.radians, [st.session_state.pickup[1], st.session_state.pickup[0], st.session_state.dropoff[1], st.session_state.dropoff[0]])
            haversine_dist = 2 * 6371 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
            mean_lat = math.radians(center_lat)
            manhattan_dist = abs(st.session_state.pickup[0] - st.session_state.dropoff[0])*111.045 + abs(st.session_state.pickup[1] - st.session_state.dropoff[1])*111.045*math.cos(mean_lat)

            input_data = {
                "passenger_count": float(passenger_count),
                "pickup_longitude": float(st.session_state.pickup[1]), "pickup_latitude": float(st.session_state.pickup[0]),
                "dropoff_longitude": float(st.session_state.dropoff[1]), "dropoff_latitude": float(st.session_state.dropoff[0]),
                "distance_km": float(haversine_dist), "manhattan_distance_km": float(manhattan_dist),
                "day_of_week": float(day_of_week), "month": float(month),
                "hour_sin": float(math.sin(2 * math.pi * hour / 24)), "hour_cos": float(math.cos(2 * math.pi * hour / 24)),
                "vendor_id_indexed": 0.0
            }

            from pyspark.ml.feature import VectorAssembler
            req_df = spark.createDataFrame(pd.DataFrame([input_data]))
            assembler = VectorAssembler(inputCols=list(input_data.keys()), outputCol="features")
            pred_log = model.transform(assembler.transform(req_df)).select("prediction").head()[0]
            pred_result = math.exp(pred_log) - 1 if pred_log > 0 else 0

            minutes, seconds = int(pred_result // 60), int(pred_result % 60)
            actual_dist = primary_dist if primary_dist > 0 else haversine_dist
            speed_kmh = actual_dist / (pred_result / 3600) if pred_result > 0 else 0
            prob_congestion = round(max(0, min(100, (25 - speed_kmh) / 20 * 100)), 1)

            # Logic Trạng thái Giao thông
            if speed_kmh > 30:
                route_color, status = "#00cc66", "Thông Suốt" # Xanh
            elif speed_kmh > 15:
                route_color, status = "#ffa600", "Ùn Ứ / Chậm" # Vàng
            else:
                route_color, status = "#ff4b4b", "Tắc Nghẽn" # Đỏ

            # KHU VỰC 3 -> KHU VỰC 4 -> KHU VỰC 5 XUẤT HIỆN SAU KHI DỰ BÁO

            # --- KHU VỰC 3: KHỐI KẾT QUẢ KPI ---
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("3️⃣ Khu vực 3: Kết Quả Dự Báo KPI")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📏 Distance (OSRM)", f"{actual_dist:.2f} km")
            m2.metric("⏱️ ETA (Đã tính độ trễ)", f"{minutes}p {seconds}s")
            m3.metric("🚗 Avg Speed", f"{speed_kmh:.1f} km/h")
            m4.metric("🚥 Congestion Index", f"{prob_congestion}% - {status}")
            st.markdown('</div>', unsafe_allow_html=True)

            # --- KHU VỰC 4: BẢN ĐỒ KẾT QUẢ & ĐƯỜNG GỢI Ý ---
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("4️⃣ Khu vực 4: Bản Đồ Giao Thông Hybrid (Google Maps)")

            show_alt = False
            if route_color != "#00cc66" and len(routes_data) > 1:
                st.warning(f"⚠️ Phát hiện đoạn đường đang bị {status}. OSRM đã tính toán một đường nhánh thay thế!")
                show_alt = st.toggle("🌟 Kích hoạt Gợi ý lộ trình mới (Đường Xanh lá)", value=True)

            m_result = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google')
            folium.Marker(st.session_state.pickup, popup="📍 Điểm đón (A)", icon=folium.Icon(color="green", icon="play")).add_to(m_result)
            folium.Marker(st.session_state.dropoff, popup="🏁 Điểm trả (B)", icon=folium.Icon(color="red", icon="stop")).add_to(m_result)

            # Vẽ đường thật từ GeoJSON
            folium.PolyLine(locations=primary_route, color=route_color, weight=7, opacity=0.9, tooltip=f"Lộ trình chính: {status}").add_to(m_result)

            # Gợi ý lộ trình nếu bật
            if show_alt and len(routes_data) > 1:
                folium.PolyLine(locations=routes_data[1][0], color="#00cc66", weight=6, dash_array='10', opacity=0.9, tooltip="Lộ trình Gợi ý (Thông Suốt)").add_to(m_result)

            st_folium(m_result, height=450, use_container_width=True, key="result_map")
            st.markdown('</div>', unsafe_allow_html=True)

            # --- KHU VỰC 5: LỊCH SỬ DỰ ĐOÁN HIỆN TẠI ---
            if st.session_state.get('force_save', False):
                st.session_state.prediction_history.append({
                    "Thời gian": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Quãng đường": f"{actual_dist:.2f} km",
                    "ETA": f"{minutes}p {seconds}s",
                    "Tốc độ": f"{speed_kmh:.1f} km/h",
                    "Ùn tắc": f"{prob_congestion}%",
                    "Trạng thái": status
                })
                st.session_state.force_save = False

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("5️⃣ Khu vực 5: Lịch Sử Lần Dự Đoán Vừa Rồi")
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- KHU VỰC 2: 3 BẢN ĐỒ HEATMAP (Luôn nằm dưới cùng của Tab Phân tích) ---
    st.markdown("---")
    st.header("🔥 Khu vực 2: Hệ thống 3 Bản đồ Heatmap")
    t1, t2, t3 = st.tabs(["📊 Heatmap 1: Mật độ quá khứ (Historical)", "⏳ Heatmap 2: Diễn biến thời gian thực", "⚖️ Heatmap 3: DualMap (Tốc độ vs Kẹt xe)"])

    with t1:
        # 1. Đường dẫn đến file CSV mật độ lịch sử (Top 100 hotspots)
        path_hist_data = "/content/drive/MyDrive/Du_bao_tac_duong_giao_thong_do_thi/KetQua_HuanLuyen/CSV/2_3_hotspot_density.csv"
        # 2. Khởi tạo bản đồ nền
        m_hist = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                            tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                            attr='Google')
        # 3. Đọc dữ liệu thật
        if os.path.exists(path_hist_data):
            df_hist = pd.read_csv(path_hist_data)
            # Chuyển đổi dữ liệu sang định dạng list: [vĩ độ, kinh độ, trọng số/mật độ]
            # Sử dụng cột 'count' làm trọng số cho HeatMap
            hist_data_real = df_hist[['pickup_latitude', 'pickup_longitude', 'count']].values.tolist()
            # Vẽ HeatMap với dữ liệu thật
            HeatMap(hist_data_real, radius=15, blur=10,
                    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m_hist)
        else:
            st.error(f"❌ Không tìm thấy dữ liệu lịch sử tại: {path_hist_data}")
        # 4. Hiển thị bản đồ
        st_folium(m_hist, height=400, use_container_width=True, key="hist_map_real")

    with t2:
        m_time = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google')
        # 1. Trỏ đến file CSV 24h thực tế vừa xuất từ Spark
        path_hourly_data = "/content/drive/MyDrive/Du_bao_tac_duong_giao_thong_do_thi/KetQua_HuanLuyen/CSV/2_4_hourly_hotspot.csv"
        if os.path.exists(path_hourly_data):
            df_hourly = pd.read_csv(path_hourly_data)
            # 2. Tạo đúng 24 khung giờ thực tế ('00:00', '01:00', ..., '23:00')
            time_index = [f"{h:02d}:00" for h in range(24)]
            heat_data_time = []
            # Lấy giá trị lớn nhất để chuẩn hóa màu đậm/nhạt
            max_count_time = df_hourly['count'].max()
            # Phân tách dữ liệu thực vào 24 khung hình
            for h in range(24):
                # Lọc dữ liệu của giờ tương ứng
                df_hour = df_hourly[df_hourly['hour'] == h]
                step_data = []
                for _, row in df_hour.iterrows():
                    # Chỉ dùng dữ liệu thật, KHÔNG DÙNG math.sin hay random
                    weight = row['count'] / max_count_time
                    step_data.append([row['lat_r'], row['lon_r'], weight])
                heat_data_time.append(step_data)
            # 4. Vẽ HeatMapWithTime
            HeatMapWithTime(heat_data_time, index=time_index, radius=12, auto_play=True).add_to(m_time)
        else:
            st.warning("⚠️ Chưa tìm thấy file `2_4_hourly_hotspot.csv`. Vui lòng chạy lại Bước 1 bên file Spark để xuất dữ liệu.")
        # Hiển thị lên Streamlit
        import streamlit.components.v1 as stc
        stc.html(m_time._repr_html_(), height=400)    # 4. Hiển thị lên Streamlit
    import streamlit.components.v1 as stc
    stc.html(m_time._repr_html_(), height=400)

    with t3:
        # Đường dẫn đến file CSV mật độ điểm nóng đã lưu từ Spark
        # Lưu ý: Kiểm tra kỹ đường dẫn này có khớp với thư mục trên Drive của bạn không
        path_real_data = "/content/drive/MyDrive/Du_bao_tac_duong_giao_thong_do_thi/KetQua_HuanLuyen/CSV/2_3_hotspot_density.csv"

        # Kiểm tra file tồn tại để tránh crash app
        if os.path.exists(path_real_data):
            df_hotspot = pd.read_csv(path_real_data)

            # Trích xuất dữ liệu thật: [vĩ độ, kinh độ, số lượng/mật độ]
            # data_speed và data_jam sẽ dùng chung dữ liệu mật độ thực tế này
            data_speed = df_hotspot[['pickup_latitude', 'pickup_longitude', 'count']].values.tolist()
            data_jam = df_hotspot[['pickup_latitude', 'pickup_longitude', 'count']].values.tolist()
        else:
            st.error(f"❌ Không tìm thấy dữ liệu thật tại: {path_real_data}")
            data_speed, data_jam = [], []

        m_dual = DualMap(location=[center_lat, center_lon], zoom_start=13, tiles=None)
        google_tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
        google_attr = 'Google'
        folium.TileLayer(tiles=google_tiles, attr=google_attr, name='Google Maps').add_to(m_dual.m1)
        folium.TileLayer(tiles=google_tiles, attr=google_attr, name='Google Maps').add_to(m_dual.m2)
        HeatMap(data_speed, radius=12, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m_dual.m1)
        HeatMap(data_jam, radius=12, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m_dual.m2)

        stc.html(m_dual._repr_html_(), height=400)

elif st.session_state.active_tab == "🕒 Lịch sử":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🕒 Lịch sử Tra cứu")
    st.write("Toàn bộ dữ liệu bạn đã tra cứu trong phiên làm việc này được lưu trữ tại đây.")
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        csv_data = history_df.to_csv(index=False, sep=',').encode('utf-8-sig')
        st.download_button("📥 Tải xuống Báo cáo (CSV)", data=csv_data, file_name="lich_su_du_bao.csv", mime="text/csv")
    else:
        st.info("Chưa có chuyến đi nào được dự đoán.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
# FOOTER (CHÂN TRANG)
# ================================================================
st.markdown(f"""
<hr style="border-color: rgba(255,255,255,0.1); margin-top: 50px;">
<div style="text-align: center; color: {current_theme['text']}; opacity: 0.8; padding: 20px; font-size: 0.9em;">
    <p style="margin: 5px;"><b>© 2026 Bản quyền:</b> Ngô Hữu Phong – 2212325 – KLK46</p>
    <p style="margin: 5px;">📧 <b>Thông liên hệ:</b> phonghuungo248@gmail.com</p>
    <p style="margin: 5px;">🛠️ <b>Công nghệ:</b> Streamlit | PySpark | OSRM API | Folium</p>
</div>
""", unsafe_allow_html=True)
