import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la aplicación
st.title("Predicción de Tiempo de Recuperación de Jugadores")

# Subir el archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")
if uploaded_file is not None:
    # Cargar datos
    data = pd.read_csv(uploaded_file)

    # Mostrar las primeras filas de los datos
    st.write(data.head())

    # Separar variables independientes (X) y dependientes (y)
    X = data[['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity']]
    y = data['Recovery_Time']

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Mostrar las métricas
    st.write(f"Error absoluto medio (MAE): {mae:.2f} días")
    st.write(f"Coeficiente de determinación (R²): {r2:.2f}")

    # Mostrar la comparación de predicciones
    predictions = pd.DataFrame({'Real': y_test.values, 'Predicho': y_pred})
    st.write("Comparación entre valores reales y predichos", predictions)

    # Visualizar la importancia de las características
    st.subheader('Importancia de las Características')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=X.columns)
    st.pyplot()

    # Visualizar el árbol de decisión
    st.subheader('Árbol de Decisión')
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, filled=True, feature_names=X.columns)
    st.pyplot()

