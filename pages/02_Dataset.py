# pages/2_Информация_о_разработчике.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Описание данных", layout="wide")
st.title("📊 Описание датасета — Поездки на такси")

st.markdown("""
Данные представляют собой набор информации о поездках такси. Цель — прогнозирование длительности поездки (`trip_duration`) 
на основе ряда признаков. Датасет содержит как числовые, так и категориальные данные, закодированные с помощью one-hot encoding.
""")

st.header("🔢 Структура датасета")

data_description = {
    "passenger_count": "Количество пассажиров в автомобиле (целое число от 1 до 10)",
    "trip_duration": "Длительность поездки в **минутах** (вещественное число). *Целевая переменная*",
    "month": "Месяц совершения поездки (число от 1 до 12)",
    "date": "Число месяца (от 1 до 31)",
    "trip_distance_km": "Расстояние поездки в **километрах** (вещественное число)",
    "country_Canada": "Флаг, указывающий, была ли поездка в Канаде (1 — да, 0 — нет)",
    "country_United States of America": "Флаг, указывающий, была ли поездка в США (1 — да, 0 — нет)",
    "daytime_afternoon": "Флаг, указывающий, была ли поездка днём (1 — да, 0 — нет)",
    "daytime_evening": "Флаг, указывающий, была ли поездка вечером (1 — да, 0 — нет)",
    "daytime_morning": "Флаг, указывающий, была ли поездка утром (1 — да, 0 — нет)",
    "daytime_night": "Флаг, указывающий, была ли поездка ночью (1 — да, 0 — нет)"
}

st.subheader("📌 Признаки и их описание:")
for feature, desc in data_description.items():
    st.markdown(f"**{feature}**: {desc}")

st.subheader("🎯 Целевая переменная")
st.markdown("`trip_duration` — длительность поездки в минутах.")

st.subheader("🧮 Тип задачи")
st.markdown("Задача регрессии: предсказание непрерывной числовой величины — времени поездки.")

st.subheader("📎 Пример строки из датасета")
example_row = """
| passenger_count | trip_duration | month | date | trip_distance_km | country_Canada | country_United States of America | daytime_afternoon | daytime_evening | daytime_morning | daytime_night |
|------------------|---------------|-------|------|-------------------|----------------|----------------------------------|-------------------|-----------------|-----------------|---------------|
|        2         |     15.6      |   4   |  12  |        7.3        |       0        |                1                 |         0         |        1        |        0        |       0       |
"""

st.markdown(example_row)
st.write(f"Количество признаков: {len(data_description)}")

st.markdown("---")

st.header("Этапы предобработки данных (EDA)")
st.markdown("""
- Пропущенные значения (`None`) были заполнены средним значением по признаку, за исключением случаев, связанных с выбросами.
- Выявленные выбросы были удалены, при этом сохранены данные, не являющиеся аномальными с точки зрения предметной области.
- Проведён анализ признаков с использованием графиков: `boxplot`, `heatmap` и `scatter plot`.
- Установлено, что признаки `trip_duration` (длительность поездки) и `trip_distance_km` (расстояние поездки) имеют распределение, близкое к нормальному.
- Географические координаты были преобразованы в названия стран и расстояние поездки с использованием библиотеки `geopandas`.
- Признаки, не влияющие на модель (например, идентификаторы), были удалены из датасета.
- Дата и время были переведены в категориальные признаки, затем закодированы в бинарное представление.
- Обнаружена корреляция между длительностью поездки и её расстоянием, особенно в вечернее и ночное время.
""")