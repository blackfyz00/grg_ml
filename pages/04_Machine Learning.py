import streamlit as st
import pandas as pd
import pickle
import os
import lightgbm

# Установка заголовка страницы
st.set_page_config(page_title="Trip Duration Prediction", layout="centered")
st.title("🚗 Прогнозирование длительности поездки")

st.markdown("Загрузите CSV-файл или введите данные вручную для получения прогноза.")

# Путь к папке с моделями
models_dir = "models"

# Проверка наличия папки models
if not os.path.exists(models_dir):
    st.error(f"Папка с моделями не найдена: {models_dir}")
else:
    # Получаем список всех .pkl файлов
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    if not model_files:
        st.warning("В папке models нет моделей (.pkl файлов)")
    else:
        # Выбор модели пользователем
        selected_model = st.selectbox("Выберите модель", model_files)

        # Загрузка модели
        model_path = os.path.join(models_dir, selected_model)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success(f"✅ Модель '{selected_model}' загружена")

        # Разделение на две колонки: загрузка файла и ручной ввод
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Загрузите CSV-файл")
            uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
            if uploaded_file:
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    st.success("Файл успешно загружен!")
                    st.write("Предпросмотр данных:")
                    st.dataframe(df_uploaded.head())

                    # Предсказание по файлу
                    if 'trip_duration' in df_uploaded.columns:
                        X = df_uploaded.drop(columns=['trip_duration'])
                    else:
                        X = df_uploaded

                    predictions = model.predict(X)
                    df_uploaded['predicted_trip_duration'] = predictions
                    st.download_button(
                        label="💾 Скачать с предсказаниями",
                        data=df_uploaded.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Ошибка при обработке файла: {e}")

        with col2:
            st.subheader("✍️ Введите данные вручную")

            passenger_count = st.number_input("Количество пассажиров", min_value=1, max_value=10, value=1)
            trip_distance_km = st.number_input("Расстояние поездки, км", min_value=0.1, max_value=100.0, value=5.0)
            month = st.selectbox("Месяц", list(range(1, 13)), format_func=lambda x: f"{x} месяц")
            date = st.slider("Дата (число месяца)", 1, 31, 15)

            country = st.radio("Страна", ["Canada", "United States of America"])
            country_Canada = 1 if country == "Canada" else 0
            country_United_States_of_America = 1 if country == "United States of America" else 0

            daytime = st.selectbox("Время суток", ["morning", "afternoon", "evening", "night"])
            daytime_morning = 1 if daytime == "morning" else 0
            daytime_afternoon = 1 if daytime == "afternoon" else 0
            daytime_evening = 1 if daytime == "evening" else 0
            daytime_night = 1 if daytime == "night" else 0

            if st.button("🔮 Получить предсказание"):
                input_data = pd.DataFrame({
                    'passenger_count': [passenger_count],
                    'trip_distance_km': [trip_distance_km],
                    'month': [month],
                    'date': [date],
                    'country_Canada': [country_Canada],
                    'country_United States of America': [country_United_States_of_America],
                    'daytime_morning': [daytime_morning],
                    'daytime_afternoon': [daytime_afternoon],
                    'daytime_evening': [daytime_evening],
                    'daytime_night': [daytime_night]
                })

                prediction = model.predict(input_data)[0]
                st.success(f"⏳ Прогнозируемая продолжительность поездки: **{round(prediction/60, 2)} минут**")

# Подвал
st.markdown("---")
st.markdown("© 2025 — ML Trip Predictor App")