# Proyecto Actividad 2 - Machine Learning Supervisado

## 3. Modelo 1: Random Forest

Decidimos utilizar el algoritmo **Random Forest**, el cual consiste en
entrenar múltiples árboles de decisión independientes que trabajan de
manera conjunta. Cada árbol realiza predicciones basadas en una muestra
diferente de los datos y un subconjunto aleatorio de variables, y al
final, el modelo combina todas esas predicciones (votación en
clasificación o promedio en regresión) para obtener un resultado más
robusto y preciso.\
Este enfoque ayuda a reducir el sobreajuste y mejora la capacidad de
generalización del modelo.

En nuestro caso, se configuró con **200 árboles** y el criterio de
partición **Gini**, el cual mide la impureza de las divisiones en las
hojas de los árboles. Elegimos **Gini** sobre *Entropy* ya que ofrece
resultados similares, pero con un menor costo computacional, logrando un
buen balance entre velocidad y precisión.

### Parámetros utilizados:

``` python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    criterion="gini",
    max_features='sqrt'
)
```

### Resultados obtenidos

Los resultados del modelo sobre el conjunto de prueba fueron los
siguientes:

![Texto alternativo](https://github.com/Ang3l1485/Informe_dos_IA/blob/main/taller_2.0/Soluci%C3%B3n%20forest.png?raw=true)

  Métrica                                 Valor
  --------------------------------------- ------------
  Exactitud (Accuracy)                    **0.9022**
  Precisión (Precision)                   **0.8962**
  Sensibilidad / Exhaustividad (Recall)   **0.9314**
  Puntaje F1 (F1-score)                   **0.9135**

### Interpretación

-   **Exactitud (Accuracy):** El modelo clasifica correctamente el
    **90.22%** de los casos.\
-   **Precisión:** De todos los pacientes que el modelo predijo como
    positivos (con enfermedad), el **89.62%** realmente lo son.\
-   **Sensibilidad (Recall):** El modelo detecta correctamente el
    **93.14%** de los pacientes enfermos, lo que es crucial en un
    problema médico.\
-   **F1-score:** Con un balance entre precisión y recall, el modelo
    alcanza un **91.35%**, mostrando un desempeño consistente y
    equilibrado.

## Modelo 2: Red Neuronal

Una Red Neuronal Artificial (RNA) sirve para abordar el problema de clasificación, ya que estos modelos son excelentes para aprender patrones no lineales y complejos en los datos. A diferencia de los modelos de ensamble como Random Forest y Gradient Boosting, una red neuronal opera con capas interconectadas de "neuronas". Cada neurona en una capa recibe entradas de las neuronas de la capa anterior, las procesa y transmite una salida a la siguiente capa. El modelo aprende ajustando los pesos de estas conexiones a través de un proceso de optimización llamado propagación hacia atrás (backpropagation).

La arquitectura que diseñamos es una red neuronal secuencial simple (un modelo Feed-forward) con tres capas densas:

Una capa de entrada que recibe las 20 características procesadas de nuestro dataset.

Una capa oculta con 64 neuronas.

Una segunda capa oculta con 32 neuronas.

Una capa de salida con una sola neurona para la clasificación binaria (enfermedad o no enfermedad).

Utilizamos la función de activación ReLU (Rectified Linear Unit) en las capas ocultas por su eficiencia computacional. Para la capa de salida, usamos la función Sigmoide, que comprime el valor de salida entre 0 y 1, perfecto para problemas de clasificación binaria.

### Arquitectura del modelo
``` python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Parametros utilizado: 
``` python
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=0
)
```

### Resultados obtenidos

Los resultados del modelo sobre el conjunto de prueba fueron los siguientes:

![Texto alternativo](https://github.com/Ang3l1485/Informe_dos_IA/blob/main/taller_2.0/matriz_confusion_nn.png)

Métrica	Valor
Exactitud (Accuracy)	0.8913
Precisión (Precision)	0.8942
Sensibilidad (Recall)	0.9118
Puntaje F1 (F1-score)	0.9029



# Análisis del modelo Gradient Boosting

## Explicación del algoritmo
El **Gradient Boosting** es un algoritmo de ensamble que combina múltiples modelos débiles (generalmente árboles de decisión poco profundos) de forma secuencial.  
A diferencia de *Random Forest*, que construye árboles de manera independiente y vota por mayoría, Gradient Boosting entrena cada nuevo árbol corrigiendo los errores del árbol anterior.  
Esto se logra calculando los **residuos** (errores) del modelo anterior y ajustando el siguiente árbol para reducirlos. Así, el modelo va mejorando paso a paso hasta alcanzar una mejor capacidad de predicción.

En resumen:
- **Random Forest**: muchos árboles entrenados en paralelo → votan en conjunto.  
- **Gradient Boosting**: muchos árboles entrenados en secuencia → cada árbol corrige al anterior.
- **red neuronal**: manejado con pesos y capas.
---

## Parámetros utilizados en la primera configuración
```python
GradientBoostingClassifier(
    n_estimators=300,   # número de árboles pequeños entrenados en secuencia
    learning_rate=0.05, # tasa de aprendizaje, controla cuánto corrige cada árbol
    max_depth=3,        # profundidad máxima de cada árbol débil
    random_state=42
)
```

### Explicación de los hiperparámetros
- **n_estimators**: número de árboles en la secuencia. Más árboles permiten un aprendizaje más detallado, pero también aumentan el riesgo de sobreajuste y el costo computacional.  
- **learning_rate**: controla cuánto aporta cada árbol al modelo final.  
  - Un valor alto = aprendizaje rápido, pero riesgo de sobreajuste.  
  - Un valor bajo = aprendizaje más estable, requiere más árboles.  
- **max_depth**: controla qué tan complejos son los árboles individuales.  
  - Árboles profundos → mayor capacidad de ajuste, pero riesgo de sobreajuste.  
  - Árboles superficiales → más estables, pero necesitan más árboles para lograr buena precisión.

---

## Resultados iniciales
Resultados obtenidos con `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`:

![Texto alternativo](https://github.com/Ang3l1485/Informe_dos_IA/blob/main/taller_2.0/propuesta_1.png)

---

## Segunda configuración optimizada
Después de varias pruebas, se encontró que lo mejor era **reducir el learning_rate y aumentar el número de árboles**, manteniendo árboles poco profundos.  
Esto permite que el modelo se acerque en pasos pequeños al resultado deseado, reduciendo el riesgo de sobreajuste.

Configuración:
```python
GradientBoostingClassifier(
    n_estimators=700,   # más árboles, cada uno corrige un pequeño error
    learning_rate=0.005,# pasos pequeños y estables
    max_depth=2,        # árboles muy simples para evitar sobreajuste
    random_state=42
)
```

---

## Resultados finales
![Texto alternativo](https://github.com/Ang3l1485/Informe_dos_IA/blob/main/taller_2.0/Solucion_ideal.png)

| Métrica              | Valor   |
|----------------------|---------|
| **Exactitud (Accuracy)** | 0.8804 |
| **Precisión (Precision)** | 0.8704 |
| **Sensibilidad (Recall)** | 0.9216 |
| **F1-score**            | 0.8952 |

---

## Conclusiones de estos resultados
- Se observó que al **incrementar demasiado la profundidad o el número de árboles** el modelo caía en **sobreajuste** (overfitting), es decir, memorizaba demasiado los datos de entrenamiento y perdía capacidad de generalización.  
- Con la configuración optimizada (muchos árboles poco profundos y learning_rate bajo), el modelo logró un **buen equilibrio entre precisión y generalización**.  
- Aunque el rendimiento fue ligeramente inferior al de Random Forest en exactitud global, Gradient Boosting mostró una **mayor estabilidad en la detección de pacientes con la condición (Recall = 0.92)**, lo cual puede ser más relevante en contextos médicos.

En conclusión, **Gradient Boosting es un algoritmo poderoso cuando se ajustan correctamente sus hiperparámetros**. Su principal ventaja frente a Random Forest es que puede capturar relaciones más complejas y corregir errores paso a paso, pero a cambio requiere un ajuste más cuidadoso para evitar sobreajuste.

