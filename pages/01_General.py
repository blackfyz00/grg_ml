# pages/2_Информация_о_разработчике.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("👤 Информация о разработчике")
st.write("ФИО: [Ваше имя]")
st.write("Группа: МО-231")
# st.image("path_to_photo.jpg", caption="Фото разработчика")
st.write("Тема РГР: Разработка Web-приложения для инференса моделей ML и анализа данных")

st.markdown("---")  # Разделитель

st.title("📦 Информация о наборе данных")
st.write("Предметная область: [описание]")
st.write("Признаки: [список признаков]")
st.write("Предобработка: [описание шагов]")

st.markdown("---")

st.title("📊 Визуализации данных")

# Пример загрузки данных (если нужно)
# data = pd.read_csv("data.csv")  # раскомментируй, если используешь свои данные

# Пример графика
# fig, ax = plt.subplots()
# sns.histplot(data['target'], ax=ax)
# st.pyplot(fig)    