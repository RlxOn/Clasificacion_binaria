# Clasificacion_binaria
Este script contiene una función que realiza los distintos tipos de clasificación binaria más utilizados (regresión logística, naive bayes, support vector machines, classifier and regression trees y random forest). 
El parámetro clasif indica el tipo de clasificador ("Logist", "Bayes", "svm", "cart" y "rf").
El parámetro de proporción sirve para la partición de conjunto de entrenamiento y prueba. 
El parámetro svm_kernel indica el tipo de kernel ("lineal", "polinomial", "radial" y "sigmoidal").
El parámetro param depende del clasificador elegido: Para "Logist" debe ser un valor que indique el punto de corte, para "Bayes" indica el valor del parámetro de laplace, para "svm" debe indicarse una lista con los parámetros del tipo de kernel* (que pueden ser cost, gamma, coef0, degree), para "cart" y "rf" debe indicarse un vector de dos atributos (el número de folds para CV y el número de tunelength).

*Para más detalles revisar la documentación de la función svm.
