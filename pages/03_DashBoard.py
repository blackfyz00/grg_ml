import streamlit as st
from PIL import Image
import os

# Установка настроек страницы
st.set_page_config(page_title="Дашборд анализа данных", layout="wide")

# Заголовок страницы
st.title("📊 Дашборд анализа данных и моделирования")
st.markdown("""
На этой странице представлены основные результаты анализа данных и оценки моделей машинного обучения для задачи прогнозирования длительности поездки.
""")

image_dir = "C:\\Users\\MyPC\\Code\\python_4sem\\AI\\Dashboard"
# Функция для отображения изображения с подписью
def display_image_with_caption(image_path, caption):
    image = Image.open(os.path.join(image_dir, image_path))
    st.image(image, caption=caption, use_container_width=True)

# Создание сетки для изображений
col1, col2, col3 = st.columns(3)

# Размещение графиков в сетке
with col1:
    # Bagging_regressor.png
    display_image_with_caption(
        "Bagging_regressor.png",
        "Метрики Bagging Regressor\n"
        "- **R² (R-squared)**: Мера объяснённой дисперсии.\n"
        "- **RMSE (Root Mean Squared Error)**: Чем меньше, тем лучше точность модели.\n"
        "- **MAE (Mean Absolute Error)**: Средняя абсолютная ошибка.\n"
    )

    # Gradient_boosting.png
    display_image_with_caption(
        "Gradient_boosting.png",
        "График Gradient Boosting\n"
        "- Показывает деревья принятия решений.\n"
        "- Используется для понимания сложности модели и её структуры.\n"
    )

    # Trees_metrics.png
    display_image_with_caption(
        "Trees_metrics.png",
        "Метрики дерева решений\n"
        "- Включает R², RMSE и MAE.\n"
        "- Позволяет сравнить качество дерева решений с другими моделями.\n"
    )

with col2:
    # corr_matrix.png
    display_image_with_caption(
        "corr_matrix.png",
        "Тепловая карта корреляций\n"
        "- Показывает взаимосвязь между признаками.\n"
        "- Цвета отражают силу корреляции: от -1 до 1.\n"
        "- Выявляет наиболее значимые признаки.\n"
    )

    # LGBMTree.png
    display_image_with_caption(
        "LGBMTree.png",
        "Дерево LightGBM\n"
        "- Визуализация одного из деревьев LightGBM.\n"
        "- Помогает понять, какие признаки используются для принятия решений.\n"
    )

    # Ridge_regressor.png
    display_image_with_caption(
        "Ridge_regressor.png",
        "Метрики Ridge Regressor\n"
        "- Включает R², RMSE и MAE.\n"
        "- Регуляризованная модель, устойчивая к мультиколлинеарности.\n"
    )

with col3:
    # ridge_metrics.png
    display_image_with_caption(
        "ridge_metrics.png",
        "Дополнительные метрики Ridge Regressor\n"
        "- Показывает детальный анализ ошибок и качества модели.\n"
        "- Включает распределение остатков и другие метрики.\n"
    )

    # Stacking_regressor.png
    display_image_with_caption(
        "Stacking_regressor.png",
        "Стекинг регрессоров\n"
        "- Комбинирует несколько моделей для улучшения предсказательной способности.\n"
        "- Показывает метрики стекинга.\n"
    )