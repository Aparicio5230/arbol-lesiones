import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV
data = pd.read_csv('injury_data.csv')

# Separar variables independientes (X) y la variable dependiente (y)
X = data[['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity']]
y = data['Recovery_Time']

# Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo con una profundidad máxima para evitar sobreajuste
model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas
st.write(f"Error absoluto medio (MAE): {mae:.2f} días")
st.write(f"Coeficiente de determinación (R²): {r2:.2f}")

# Comparación entre valores reales y predichos
predictions = pd.DataFrame({'Real': y_test.values, 'Predicho': y_pred})
st.write(predictions)

# Exportar el árbol de decisión como un archivo .dot
dot_data = export_graphviz(model, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)

# Crear el gráfico de Graphviz
graph = graphviz.Source(dot_data)

# Mostrar el árbol de decisión en Streamlit
st.graphviz_chart(graph)
