import streamlit as st
import pandas as pd
from PIL import Image
from model import open_data, preprocess_data, get_X_y_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open(
        "data/page_icon_car.png")

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Cars",
        page_icon=image,

    )

    st.write(
        """
        # Предсказание цены автомобиля
        Определение цены автомобиля с использованием линейной регрессии 
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction):
    st.write("## Предсказание")
    st.write(prediction)


def process_side_bar_inputs():
    st.sidebar.header(
        "Задайте параметры автомобиля")
    user_input_df = sidebar_input_features()
    write_user_data(user_input_df)

    data = open_data()
    X_df, _ = get_X_y_data(data)
    full_X_df = pd.concat((user_input_df, X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]

    prediction = load_model_and_predict(user_X_df)
    write_prediction(prediction)


def sidebar_input_features():
    year = st.sidebar.slider("Год выпуска",
                             min_value=1950, max_value=2015, step=1)
    km_driven = st.sidebar.number_input(
        "Пробег на дату продажи")
    fuel = st.sidebar.selectbox(
        "Тип используемого топлива", ("Дизель", "Бензин"))
    transmission = st.sidebar.selectbox(
        "Тип трансмиссии", ("Ручная", "Автоматическая"))
    seller_type = st.sidebar.selectbox(
        "Продавец", ("Официальный дилер", "Физ. лицо", "Дилер",))
    owner = st.sidebar.selectbox("Количество владельцев", (
        "Один", "Два", "Три", "Четыре и более"))
    mileage = st.sidebar.number_input("Текущий пробег")
    engine = st.sidebar.number_input("Объем двигателя")
    max_power = st.sidebar.number_input(
        "Пиковая мощность двигателя")
    seats = st.sidebar.slider(
        "Количество мест", min_value=1, max_value=10, step=1)

    translatetion = {
        "Дизель": "Diesel",
        "Бензин": "Fuel",
        "Ручная": "Manual",
        "Официальный дилер": "Trustmark Dealer",
        "Физ. лицо": "Individual",
        "Дилер": "Dealer",
        "Автоматическая": "Automatic",
        "Один": "First Owner",
        "Два": "Second Owner",
        "Три": "Third Owner",
        "Четыре и более": "Fourth & Above Onwer",
    }

    data = {
        "year": year,
        "km_driven": km_driven,
        "fuel": translatetion[fuel],
        "transmission": translatetion[transmission],
        "seller_type": translatetion[seller_type],
        "owner": translatetion[owner],
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
