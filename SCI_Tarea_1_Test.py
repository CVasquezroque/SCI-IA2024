# utils.py

import numpy as np

def gaussian_test(x, mu, sigma):
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    g = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return g

def crear_filtro_gaussiano_2d_test(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    gaussian_1d = gaussian_test(x, 0, sigma)
    gaussian_2d = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            gaussian_2d[i, j] = gaussian_1d[i] * gaussian_1d[j]
            return gaussian_2d / gaussian_2d.sum()


def test_gaussian(student_gaussian):
    x = 0
    mu = 0
    sigma = 1
    result = student_gaussian(x, mu, sigma)
    expected = 1 / (np.sqrt(2 * np.pi))
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"
    print("Todos los test pasaron! Sigue así Coder")

def test_calcular_gaussianas(student_calcular_gaussianas):
    x = np.linspace(-10, 10, 1000)
    mu = np.mean(x)
    sigma_values = [2, 4, 8, 16]
    result = student_calcular_gaussianas(mu, sigma_values, x)
    assert result.shape == (1000, 4), f"Expected shape (1000, 4), got {result.shape}"
    assert np.allclose(result[:, 0], gaussian_test(x, mu, sigma_values[0])), "First column does not match expected Gaussian values"
    print("Todos los test pasaron! Sigue así Coder")

def test_crear_filtro_gaussiano(student_crear_filtro_gaussiano):
    sigma = 2
    result = student_crear_filtro_gaussiano(sigma)
    expected_size = int(6 * sigma + 1)
    assert result.shape == (expected_size,), f"Expected shape ({expected_size},), got {result.shape}"
    assert np.isclose(result.sum(), 1), f"Expected sum to be 1, got {result.sum()}"
    print("Todos los test pasaron! Sigue así Coder")

def test_aplicar_filtro_gaussiano_1d(student_aplicar_filtro_gaussiano_1d):
    data = np.array([1, 2, 3, 4, 5])
    sigma = 1
    result = student_aplicar_filtro_gaussiano_1d(data, sigma)
    assert result.shape == data.shape, f"Expected shape {data.shape}, got {result.shape}"
    assert np.all(result >= 0), "All values should be non-negative"
    print("Todos los test pasaron! Sigue así Coder")

def test_crear_filtro_gaussiano_2d(student_crear_filtro_gaussiano_2d):
    size = 5
    sigma = 1
    result = student_crear_filtro_gaussiano_2d(size, sigma)
    assert result.shape == (size, size), f"Expected shape ({size}, {size}), got {result.shape}"
    assert np.isclose(result.sum(), 1), f"Expected sum to be 1, got {result.sum()}"
    print("Todos los test pasaron! Sigue así Coder")

def test_aplicar_filtro_gaussiano_imagen(student_aplicar_filtro_gaussiano_imagen):
    imagen = np.random.rand(100, 100)
    filtro = crear_filtro_gaussiano_2d_test(5, 1)
    result = student_aplicar_filtro_gaussiano_imagen(imagen, filtro)
    assert result.shape == imagen.shape, f"Expected shape {imagen.shape}, got {result.shape}"
    assert np.all(result >= 0), "All values should be non-negative"
    print("Todos los test pasaron! Sigue así Coder")

