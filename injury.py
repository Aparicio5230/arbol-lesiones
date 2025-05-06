import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Cargar el CSV por defecto al iniciar la app
@st.cache
def cargar_datos():
    # Cargar datos desde el archivo CSV (por defecto en el mismo directorio)
    try:
        return pd.read_csv('injury_data.csv')
    except FileNotFoundError:
        st.warning("El archivo 'injury_data.csv' no se encontró. Asegúrate de que el archivo esté en el directorio.")
        return pd.DataFrame()  # Retorna un DataFrame vacío si no se encuentra el archivo

# Cargar los datos al inicio
data = cargar_datos()

# Separar variables independientes (X) y la variable dependiente (y)
if not data.empty:
    X = data[['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity']]
    y = data['Recovery_Time']

    # Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo con una profundidad máxima para evitar sobreajuste
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    # Subir nuevos datos para predicción
    st.header("Sube nuevos datos para hacer predicción")
    uploaded_file = st.file_uploader("Cargar un archivo CSV con nuevos datos", type=["csv"])

    if uploaded_file is not None:
        # Leer el nuevo archivo CSV
        new_data = pd.read_csv(uploaded_file)

        # Asegúrate de que el archivo cargado tenga las mismas columnas que el dataset original
        required_columns = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity']
        if all(col in new_data.columns for col in required_columns):
            # Realizar predicciones sobre los nuevos datos
            new_data['Predicted_Recovery_Time'] = model.predict(new_data[required_columns])

            # Mostrar las predicciones
            st.write("Predicciones sobre los nuevos datos:")
            st.write(new_data[['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity', 'Predicted_Recovery_Time']])

            # Guardar los nuevos datos y sus predicciones en el archivo CSV original
            data = pd.concat([data, new_data], ignore_index=True)
            data.to_csv('injury_data.csv', index=False)
            st.success("Nuevos datos guardados en 'injury_data.csv'.")
        else:
            st.error(f"El archivo debe contener las siguientes columnas: {', '.join(required_columns)}.")
    else:
        st.info("Sube un archivo CSV con nuevos datos para hacer la predicción.")
else:
    st.warning("No se encontraron datos previos. Asegúrate de que 'injury_data.csv' esté presente.")
