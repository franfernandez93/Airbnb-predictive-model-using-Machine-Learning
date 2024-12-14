# Airbnb-predictive-model-using-Machine-Learning
## Modelo Predictivo Usando Machine Learning
Este proyecto tiene como objetivo desarrollar un modelo predictivo para estimar el precio de los alojamientos en Airbnb a partir de un conjunto de datos proporcionado. Utilicé diversas técnicas de análisis y preprocesamiento de datos, así como modelos de Machine Learning para entrenar y validar el modelo predictivo.

Objetivo
El objetivo principal de este proyecto es predecir el precio de los alojamientos en Airbnb, basándonos en una serie de características como el número de habitaciones, el número de camas, los servicios ofrecidos, entre otros. Para ello, el flujo de trabajo sigue los siguientes pasos:

División de los datos: Primero, divido los datos en conjuntos de entrenamiento y prueba para evitar el uso de datos de prueba en el entrenamiento del modelo.
Limpieza de datos: Eliminación de características irrelevantes y tratamiento de valores faltantes.
Preprocesamiento: Escalado de características, codificación de variables categóricas y creación de nuevas características.
Entrenamiento del modelo: Entrenamiento de un modelo de regresión utilizando el conjunto de entrenamiento.
Evaluación del modelo: Validación del modelo con datos de prueba y optimización de hiperparámetros.
Visualización: Análisis exploratorio de los datos y visualización de relaciones clave.
Tecnologías y Librerías Utilizadas
Este proyecto hace uso de diversas tecnologías y librerías de Python para realizar el análisis y desarrollo del modelo predictivo:

Python 3: Lenguaje de programación utilizado para implementar todo el proyecto.
pandas: Librería esencial para la manipulación y análisis de datos.
numpy: Utilizada para operaciones matemáticas y manejo de arrays.
scikit-learn: Librería fundamental para la construcción del modelo de Machine Learning, incluyendo la implementación de algoritmos de regresión y técnicas de preprocesamiento.
matplotlib y seaborn: Librerías utilizadas para la visualización de datos y la creación de gráficos.
sklearn.preprocessing: Utilizada para el escalado de características y la codificación de variables categóricas.
OneHotEncoder: Para transformar variables categóricas en representaciones numéricas a través de codificación one-hot.
Pasos Realizados en el Proyecto
Cargar los datos: Se cargan los datos desde el archivo CSV proporcionado en un DataFrame de pandas.

División de los datos: Los datos se dividen en dos conjuntos: entrenamiento y prueba, utilizando la función train_test_split de scikit-learn. Esto asegura que el modelo se entrene solo con el conjunto de entrenamiento y se valide en el conjunto de prueba.

Limpieza de datos: Se eliminan columnas irrelevantes y se gestionan los valores faltantes. Algunas columnas se completan con la moda de las características correspondientes (por ejemplo, Bathrooms, Bedrooms, Beds), mientras que otras se transforman en nuevas columnas, como Has_Cleaning_Fee y Cleaning_Fee_Amount.

Preprocesamiento:

Se realiza una normalización de datos para escalar las características numéricas solo en el conjunto de entrenamiento.
Se emplea la codificación one-hot para las variables categóricas, como la política de cancelación, convirtiéndolas en variables binarias.
Entrenamiento del modelo: Se entrena un modelo de regresión utilizando el conjunto de entrenamiento y se ajusta utilizando la validación cruzada y la optimización de hiperparámetros mediante GridSearchCV.

Evaluación y visualización: Se evalúa el modelo con el conjunto de prueba y se visualizan las relaciones clave entre las características y el precio de los alojamientos mediante gráficos de dispersión y histogramas. Además, se analizan las características más importantes utilizando el test F y la información mutua.
