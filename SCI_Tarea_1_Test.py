import numpy as np

def gaussian(x, mu, sigma):
    """
    Calcula la función Gaussiana para un valor dado de x, mu y sigma.

    Parámetros:
    x (float): Valor en el eje x.
    mu (float): Media de la distribución Gaussiana.
    sigma (float): Desviación estándar de la distribución Gaussiana.

    Retorna:
    float: Valor de la función Gaussiana en x.
    """
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    g = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return g

def calcular_gaussianas(mu, sigma_values, x):
    """
    Calcula las funciones Gaussianas para los valores de sigma dados.

    Parámetros:
    mu (float): Media de la distribución Gaussiana.
    sigma_values (list): Lista de valores de sigma (desviación estándar).
    x (numpy.ndarray): Array de valores en el eje x.

    Retorna:
    numpy.ndarray: Matriz donde cada columna corresponde a una función Gaussiana para cada valor de sigma.
    """
    gaussianas_matrix = np.zeros((len(x), len(sigma_values)))
    for i, sigma in enumerate(sigma_values):
        gaussianas_matrix[:, i] = gaussian(x, mu, sigma)
    return gaussianas_matrix

def crear_filtro_gaussiano(sigma):
    """
    Crea un filtro gaussiano de 1D basado en el valor de sigma dado.

    Parámetros:
    sigma (float): Desviación estándar de la distribución gaussiana.

    Retorna:
    numpy.ndarray: Filtro gaussiano normalizado.
    """
    size = int(6 * sigma + 1)
    x = np.linspace(-size // 2, size // 2, size)
    filtro = gaussian(x, 0, sigma)
    filtro /= filtro.sum()
    return filtro

def aplicar_filtro_gaussiano_1d(data, sigma):
    """
    Aplica un filtro gaussiano de 1D a los datos proporcionados.

    Parámetros:
    data (numpy.ndarray): Arreglo de datos a los que se aplicará el filtro.
    sigma (float): Desviación estándar de la distribución gaussiana utilizada para crear el filtro.

    Retorna:
    numpy.ndarray: Arreglo de datos suavizados después de aplicar el filtro gaussiano.
    """
    filtro = crear_filtro_gaussiano(sigma)
    tamaño_filtro = len(filtro)
    mitad_filtro = tamaño_filtro // 2

    smoothed_data = np.zeros_like(data)

    for i in range(len(data)):
        suma = 0
        for j in range(tamaño_filtro):
            k = i + j - mitad_filtro
            if 0 <= k < len(data):
                suma += data[k] * filtro[j]
        smoothed_data[i] = suma

    return smoothed_data

def crear_filtro_gaussiano_2d(size, sigma):
    """
    Crea un filtro Gaussiano bidimensional.

    Parámetros:
    size (int): Tamaño del filtro (número de puntos en cada dimensión).
    sigma (float): Desviación estándar de la distribución Gaussiana.

    Retorna:
    numpy.ndarray: Filtro Gaussiano bidimensional normalizado.
    """
    # Crear una cuadrícula de puntos
    x = np.linspace(-size // 2, size // 2, size)
    gaussian_1d = gaussian(x, 0, sigma)

    # Crear el filtro Gaussiano 2D sin usar np.newaxis
    # Crear una matriz 2D donde cada fila es una copia del vector 1D
    gaussian_2d = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            gaussian_2d[i, j] = gaussian_1d[i] * gaussian_1d[j]

    # Normalizar el filtro para que la suma de todos los elementos sea 1
    return gaussian_2d / gaussian_2d.sum()

def aplicar_filtro_gaussiano_imagen(imagen, filtro):
    """
    Aplica un filtro Gaussiano bidimensional a una imagen, ignorando los bordes.

    Parámetros:
    imagen (numpy.ndarray): Imagen a la cual se aplicará el filtro.
    filtro (numpy.ndarray): Filtro Gaussiano bidimensional.

    Retorna:
    numpy.ndarray: Imagen filtrada.
    """
    img_filtrada = np.zeros_like(imagen)
    offset = filtro.shape[0] // 2

    # Aplicar el filtro ignorando los bordes
    for i in range(offset, imagen.shape[0] - offset):
        for j in range(offset, imagen.shape[1] - offset):
            region = imagen[i - offset:i + offset + 1, j - offset:j + offset + 1]
            img_filtrada[i, j] = np.sum(region * filtro)

    return img_filtrada
