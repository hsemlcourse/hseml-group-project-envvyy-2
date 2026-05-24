"""Streamlit интерфейс для предсказания популярности Twitch стримеров."""

from __future__ import annotations

import requests
import streamlit as st

# API configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Twitch Popularity Predictor",
    page_icon="📺",
    layout="wide",
)

st.title("📺 Twitch Popularity Prediction")
st.markdown("Предсказание времени просмотра для Twitch стримеров")


def get_model_info():
    """Get model info from API."""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def check_health():
    """Check API health."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def make_prediction(data: dict):
    """Make prediction via API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


# Check API status
api_healthy = check_health()

if not api_healthy:
    st.warning("⚠️ API недоступен. Запустите API сервер командой: `python api.py`")
    st.info("Или используйте docker-compose: `docker-compose up`")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ О проекте")
    st.write("""
    **Twitch Popularity Prediction**
    
    Модель предсказывает время просмотра (Watch Time) 
    на основе характеристик стримера.
    
    **Параметры:**
    - Время стрима
    - Количество зрителей (пиковое и среднее)
    - Подписчики и их динамика
    - Язык и статус партнёра
    """)
    
    model_info = get_model_info()
    if model_info:
        st.write("**Информация о модели:**")
        st.write(f"- Тип: {model_info.get('model_type', 'N/A')}")
        st.write(f"- Версия: {model_info.get('model_version', 'N/A')}")
        st.write(f"- Фичей: {model_info.get('features_count', 'N/A')}")

# Main input form
st.header("📝 Введите данные стримера")

col1, col2, col3 = st.columns(3)

with col1:
    stream_time = st.number_input(
        "Время стрима (минут)",
        min_value=1,
        max_value=1000000,
        value=1000,
        step=100,
        help="Общее время проведённых стримов в минутах"
    )
    
    peak_viewers = st.number_input(
        "Пиковое число зрителей",
        min_value=0,
        max_value=10000000,
        value=5000,
        step=100,
        help="Максимальное количество одновременных зрителей"
    )
    
    avg_viewers = st.number_input(
        "Среднее число зрителей",
        min_value=0,
        max_value=10000000,
        value=1000,
        step=100,
        help="Среднее количество зрителей за стрим"
    )

with col2:
    followers = st.number_input(
        "Подписчики",
        min_value=0,
        max_value=100000000,
        value=100000,
        step=1000,
        help="Общее количество подписчиков"
    )
    
    followers_gained = st.number_input(
        "Получено подписчиков",
        min_value=-1000000,
        max_value=100000000,
        value=10000,
        step=500,
        help="Подписчики gained за период"
    )
    
    views_gained = st.number_input(
        "Получено просмотров",
        min_value=0,
        max_value=10000000000,
        value=1000000,
        step=10000,
        help="Просмотры gained за период"
    )

with col3:
    partnered = st.selectbox(
        "Партнёр Twitch",
        options=[True, False],
        index=0,
        help="Является ли стример партнёром Twitch"
    )
    
    mature = st.selectbox(
        "Возрастной контент",
        options=[True, False],
        index=1,
        help="Есть ли возрастной контент"
    )
    
    language = st.selectbox(
        "Язык",
        options=[
            "English", "Russian", "German", "Spanish", "French",
            "Portuguese", "Korean", "Chinese", "Japanese", "Other"
        ],
        index=0,
        help="Основной язык стрима"
    )

# Prediction button
st.markdown("---")

if st.button("🔮 Предсказать", type="primary", use_container_width=True):
    if not api_healthy:
        st.error("❌ API недоступен. Проверьте, что сервер запущен.")
    else:
        with st.spinner("Делаем предсказание..."):
            data = {
                "stream_time_minutes": float(stream_time),
                "peak_viewers": int(peak_viewers),
                "average_viewers": int(avg_viewers),
                "followers": int(followers),
                "followers_gained": int(followers_gained),
                "views_gained": int(views_gained),
                "partnered": partnered,
                "mature": mature,
                "language": language,
            }
            
            result = make_prediction(data)
            
            if "error" in result:
                st.error(f"❌ Ошибка: {result['error']}")
            else:
                st.success("✅ Предсказание получено!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Время просмотра (минуты)",
                        value=f"{result['watch_time_minutes']:,.0f}"
                    )
                
                with col2:
                    hours = result['watch_time_minutes'] / 60
                    st.metric(
                        label="Время просмотра (часы)",
                        value=f"{hours:,.1f}"
                    )
                
                st.info(f"Модель: {result['model_version']}")
                
                # Interpretation
                st.markdown("### 📊 Интерпретация")
                
                if hours < 1:
                    st.write("🔵 **Низкая популярность** — стример только начинает или имеет небольшую аудиторию.")
                elif hours < 100:
                    st.write("🟡 **Средняя популярность** — стример имеет стабильную, но не очень большую аудиторию.")
                elif hours < 1000:
                    st.write("🟠 **Высокая популярность** — стример популярен и имеет значительную аудиторию.")
                else:
                    st.write("🟢 **Очень высокая популярность** — топовый стример с огромной аудиторией!")

# Example predictions section
st.markdown("---")
st.header("📋 Примеры известных стримеров")

examples = [
    {"name": "xQcOW", "stream_time": 215250, "peak_viewers": 222720, "avg_viewers": 27716, 
     "followers": 3246298, "followers_gained": 1734810, "views_gained": 93036735, 
     "partnered": True, "mature": False, "language": "English"},
    {"name": "Gaules", "stream_time": 515280, "peak_viewers": 387315, "avg_viewers": 10976,
     "followers": 1767635, "followers_gained": 1023779, "views_gained": 102611607,
     "partnered": True, "mature": True, "language": "Portuguese"},
    {"name": "Shroud", "stream_time": 180000, "peak_viewers": 150000, "avg_viewers": 20000,
     "followers": 9500000, "followers_gained": 500000, "views_gained": 50000000,
     "partnered": True, "mature": False, "language": "English"},
]

for ex in examples:
    with st.expander(f"🎮 {ex['name']}"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Время стрима:** {ex['stream_time']:,} мин")
            st.write(f"**Пик зрителей:** {ex['peak_viewers']:,}")
            st.write(f"**Средние зрители:** {ex['avg_viewers']:,}")
        with col2:
            st.write(f"**Подписчики:** {ex['followers']:,}")
            st.write(f"**Партнёр:** {'Да' if ex['partnered'] else 'Нет'}")
            st.write(f"**Язык:** {ex['language']}")
        
        if st.button(f"Предсказать для {ex['name']}", key=ex['name']):
            if not api_healthy:
                st.error("❌ API недоступен")
            else:
                data = {
                    "stream_time_minutes": float(ex['stream_time']),
                    "peak_viewers": ex['peak_viewers'],
                    "average_viewers": ex['avg_viewers'],
                    "followers": ex['followers'],
                    "followers_gained": ex['followers_gained'],
                    "views_gained": ex['views_gained'],
                    "partnered": ex['partnered'],
                    "mature": ex['mature'],
                    "language": ex['language'],
                }
                
                with st.spinner("Предсказание..."):
                    result = make_prediction(data)
                    if "error" not in result:
                        st.success(f"📺 Предсказанное время просмотра: **{result['watch_time_minutes']:,.0f} минут** ({result['watch_time_minutes']/60:,.1f} часов)")
                    else:
                        st.error(f"Ошибка: {result['error']}")

# Footer
st.markdown("---")
st.markdown("Сделано с ❤️ для курса ML | 2024")