import torch
import matplotlib.pyplot as plt

def plot_results(model, distances, times):
    """
    Grafica los puntos de datos reales y la línea predicha por el modelo para un dataset dado.

    Args:
        model: El modelo de machine learning entrenado para usar en las predicciones.
        distances: Los puntos de datos de entrada (features) para el modelo.
        times: Los puntos de datos objetivo (labels) para el gráfico.
    """
    # Establece el modelo en modo de evaluación (evaluation mode)
    model.eval()

    # Desactiva el cálculo de gradientes para una inferencia eficiente
    with torch.no_grad():
        # Realiza predicciones usando el modelo entrenado
        predicted_times = model(distances)

    # Crea una nueva figura para el gráfico
    plt.figure(figsize=(8, 6))
    
    # Grafica los puntos de datos reales
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='None', label='Actual Delivery Times')
    
    # Grafica la línea predicha por el modelo
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', marker='None', label='Predicted Line')
    
    # Configura el título del gráfico
    plt.title('Actual vs. Predicted Delivery Times')
    # Configura la etiqueta del eje x
    plt.xlabel('Distance (miles)')
    # Configura la etiqueta del eje y
    plt.ylabel('Time (minutes)')
    # Muestra la leyenda
    plt.legend()
    # Añade una cuadrícula (grid) al gráfico
    plt.grid(True)
    # Muestra el gráfico
    plt.show()

    

def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Compara y grafica las predicciones de un modelo frente a nuevos datos no lineales (non-linear).

    Args:
        model: El modelo entrenado que será evaluado.
        new_distances: Los nuevos datos de entrada para generar predicciones.
        new_times: Los valores objetivo reales para la comparación.
    """
    # Establece el modelo en modo de evaluación (evaluation mode)
    model.eval()
    
    # Desactiva el cálculo de gradientes para la inferencia
    with torch.no_grad():
        # Genera predicciones usando el modelo
        predictions = model(new_distances)

    # Crea una nueva figura para el gráfico
    plt.figure(figsize=(8, 6))
    
    # Grafica los puntos de datos reales
    plt.plot(new_distances.numpy(), new_times.numpy(), color='orange', marker='o', linestyle='None', label='Actual Data (Bikes & Cars)')
    
    # Grafica las predicciones del modelo
    plt.plot(new_distances.numpy(), predictions.numpy(), color='green', marker='None', label='Linear Model Predictions')
    
    # Configura el título del gráfico
    plt.title('Linear Model vs. Non-Linear Reality')
    # Configura la etiqueta del eje x
    plt.xlabel('Distance (miles)')
    # Configura la etiqueta del eje y
    plt.ylabel('Time (minutes)')
    # Añade la leyenda al gráfico
    plt.legend()
    # Añade una cuadrícula al gráfico para mejor legibilidad
    plt.grid(True)
    # Muestra el gráfico
    plt.show()