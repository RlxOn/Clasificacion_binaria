################## Clasificacion binaria #######################################
pacman::p_load(ggplot2, tidyverse, rpart, rpart.plot, caret, e1071,
               randomForest, broom)

set.seed(1989)    #Taylor's version

clasificador_binario <- function(formula, data, clasif, prop = 0.66, svm_kernel = NULL, param){
  #Inicializamos variables
  formula <- formula
  data <- data
  clasif <- clasif
  prop <- prop
  svm_kernel <- svm_kernel
  param <- param

  
  #Construimos el conjunto de entrenamiento y el de prueba
  train_test <- function(data, prop){
    #Verificamos que p este entre 0 y 1.
    if(prop <= 0 | prop >= 1){
      stop("La proporcion debe estar entre 0 y 1")
    }
    
    #Generamos los indices para el conjunto de prueba y entrenamiento
    indices <- 1:dim(data)[1]
    ind_train <- sample(indices, ceiling(prop * length(indices)))
    ind_test <- setdiff(indices, ind_train)
    
    #Guardamos los df de prueba y de entrenamiento
    train_set <- data[ind_train,]
    test_set <- data[ind_test,]
    
    #Guardamos en lista para regresarla como argumento
    list_set <- list(train_set, test_set)
    names(list_set) <- c("train_set", "test_set")
    
    return(list_set)
    
  }
  
  tt_sets <- train_test(data, prop)
  train_set <- tt_sets$train_set
  test_set <- tt_sets$test_set
  
  var_resp <- as.character(formula)[2]
  test_resp <- as.factor(test_set[[var_resp]])
  
  #Regresion logistica =========================================================
  if(clasif == "Logist"){
    #Modelo
    logist_model <- glm(formula, data = train_set, family = "binomial")
    
    #Obtenemos los intervalos de confianza
    conf_int <- confint(logist_model, level = 0.95)
    
    #Creamos la tabla de coeficientes
    coef_tab <- broom::tidy(logist_model) %>%
      mutate(Lower = round(conf_int[,1],4),
             Upper = round(conf_int[,2],4),
             is_signf = ifelse(p.value < 0.05, TRUE, FALSE))
    
    print(coef_tab)
    
    #Creamos la tabla de metricas de la regresion
    measure_tab <- broom::glance(logist_model)
    
    print(measure_tab)
    
    #Obtenemos la matriz de confusion y realizamos prediccion con el test set
    pred <- predict.glm(logist_model, newdata = test_set, type = "response")
    
    logist_pred <- data.frame(
      probs = pred,
      category = as.factor(ifelse(pred > param,1,0)) 
    )
    
    conf_mat <- confusionMatrix(data = logist_pred$category, test_resp)
    
    print(conf_mat)
    
    output_list <- list(logist_model,coef_tab, measure_tab, train_set, test_set,
                        logist_pred, conf_mat)
    
    names(output_list) <- c("Modelo","Coeficientes", "Metricas", "Entrenamiento", "Prueba",
                            "Prediccion", "Matriz_Confusion")
    
    return(output_list)
  }
  
  
  #Clasificador de Bayes =======================================================
  if(clasif == "Bayes"){
    #Modelo
    bayes_model <- naiveBayes(formula, data = train_set, laplace = param)
    print(bayes_model)
    
    #Prediccion
    bayes_pred <- predict(bayes_model, newdata = test_set)
    
    #Matriz de confusion
    conf_mat <- confusionMatrix(data = bayes_pred, test_resp)
    print(conf_mat)
    
    output_list <- list(bayes_model, bayes_pred, conf_mat, train_set, test_set)
    names(output_list) <- c("Modelo", "Prediccion", "Matriz_Confusion", "Entrenamiento",
                            "Prueba")
    return(output_list)
  }
  
  #SVM =========================================================================
  if(clasif == "svm"){
    
    #Kernel lineal
    if(svm_kernel == "lineal"){
      #Encontramos el mejor parametro
      svm_tune <- tune("svm", formula, data = train_set, kernel = "linear",
                       ranges = param)
      
      costo <- svm_tune$best.model$cost
      svm_lineal <- svm(formula, data = train_set, kernel = "linear", cost = costo)
      
      #Prediccion y matriz de confusion
      svm_pred <- predict(svm_lineal, test_set)
      conf_mat <- confusionMatrix(data = svm_pred, test_resp)
      
      print(svm_lineal)
      print(conf_mat)
     
      output_list <- list(svm_tune, svm_lineal, svm_pred, conf_mat, train_set, test_set)
      names(output_list) <- c("Ajuste", "Modelo", "Prediccion", "Matriz_Confusion",
                              "Entrenamiento", "Prueba")
      
      return(output_list)
    }
    
    #Kernel Gaussiano
    if(svm_kernel == "radial"){
      #Encontramos el mejor parametro
      svm_tune <- tune("svm", formula, data = train_set, kernel = "radial",
                       ranges = param)
      
      costo <- svm_tune$best.model$cost
      gamma <- svm_tune$best.model$gamma
      
      svm_radial <- svm(formula, data = train_set, kernel = "radial", 
                        cost = costo, gamma = gamma)
      
      #Prediccion y matriz de confusion
      svm_pred <- predict(svm_radial, test_set)
      conf_mat <- confusionMatrix(data = svm_pred, test_resp)
      
      print(svm_radial)
      print(conf_mat)
      
      output_list <- list(svm_tune, svm_radial, svm_pred, conf_mat, train_set, test_set)
      names(output_list) <- c("Ajuste", "Modelo", "Prediccion", "Matriz_Confusion",
                              "Entrenamiento", "Prueba")
      
      return(output_list)
    }
    
    #Kernel polinomial
    if(svm_kernel == "polinomial"){
      #Encontramos el mejor parametro
      svm_tune <- tune("svm", formula, data = train_set, kernel = "polynomial",
                       ranges = param)
      
      costo <- svm_tune$best.model$cost
      gamma <- svm_tune$best.model$gamma
      coef0 <- svm_tune$best.model$coef0
      degree <- svm_tune$best.model$degree
      
      svm_poli <- svm(formula, data = train_set, kernel = "polynomial", 
                        cost = costo, gamma = gamma, coef0 = coef0, degree = degree)
      
      #Prediccion y matriz de confusion
      svm_pred <- predict(svm_poli, test_set)
      conf_mat <- confusionMatrix(data = svm_pred, test_resp)
      
      print(svm_poli)
      print(conf_mat)
      
      output_list <- list(svm_tune, svm_poli, svm_pred, conf_mat, train_set, test_set)
      names(output_list) <- c("Ajuste", "Modelo", "Prediccion", "Matriz_Confusion",
                              "Entrenamiento", "Prueba")
      
      return(output_list)
    }
    
    #Kernel sigmoidal
    if(svm_kernel == "sigmoidal"){
      #Encontramos el mejor parametro
      svm_tune <- tune("svm", formula, data = train_set, kernel = "sigmoid",
                       ranges = param)
      
      costo <- svm_tune$best.model$cost
      gamma <- svm_tune$best.model$gamma
      coef0 <- svm_tune$best.model$coef0
      
      svm_sigm <- svm(formula, data = train_set, kernel = "sigmoid", 
                      cost = costo, gamma = gamma, coef0 = coef0)
      
      #Prediccion y matriz de confusion
      svm_pred <- predict(svm_sigm, test_set)
      conf_mat <- confusionMatrix(data = svm_pred, test_resp)
      
      print(svm_sigm)
      print(conf_mat)
      
      output_list <- list(svm_tune, svm_sigm, svm_pred, conf_mat, train_set, test_set)
      names(output_list) <- c("Ajuste", "Modelo", "Prediccion", "Matriz_Confusion",
                              "Entrenamiento", "Prueba")
      
      return(output_list)
    }
    
  }
  
  #CART ========================================================================
  if(clasif == "cart"){
    #Ajustamos el mejor arbol
    cart_train <- train(formula, data = train_set, method = "rpart",
                        trControl = trainControl(method = "cv", number = param[1]),
                        tuneLength = param[2])
    
    #Guardamos el mejor arbol
    best_cart <- cart_train$finalModel
    
    #Dibujamos el mejor arbol
    print(
    rpart.plot(best_cart, extra = 2, under = TRUE,  varlen = 0, faclen = 0,
               fallen.leaves = TRUE, space = 2, tweak = 1.5, type = 2))
    
    #Revisamos la importancia de las variables
    print(
    dotPlot(varImp(cart_train, compete = FALSE)))
    
    #Prediccion y matriz
    cart_pred <- predict(best_cart, newdata = test_set, type = "class")
    conf_mat <- confusionMatrix(cart_pred, test_resp)
    
    print(cart_train)
    print(conf_mat)
    
    output_list <- list(cart_train, best_cart, cart_pred, conf_mat, train_set, test_set)
    names(output_list) <- c("Ajuste", "Modelo", "Prediccion", "Matriz_Confusion",
                            "Entrenamiento", "Prueba")
    
    return(output_list)
  }
  
  #Bosques aleatorios ==========================================================
  if(clasif == "rf"){
    #Ajustamos el mejor bosque
    rf_train <- train(formula, data = train_set, method = "rf",
                      trControl = trainControl(method = "cv", number = param[1]),
                      tuneLength = param[2])
    
    #Mejores parametros
    mtry <- rf_train$finalModel$mtry
    ntrees <- which.min(rf_train$finalModel$err.rate[,1])
    
    #Mejor modelo
    best_rf <- randomForest(formula, data = train_set, ntree = ntrees, mtry = mtry)
    
    #Importancia de los atributos
    print(
    varImpPlot(best_rf, main = "Importancia de los atributos", color = c("#800000","navy"),
               cex = 1.2))
    
    #Prediccion y matriz
    rf_pred <- predict(best_rf, newdata = test_set, type = "class")
    conf_mat <- confusionMatrix(rf_pred, test_resp)
    
    print(best_rf)
    print(conf_mat)
    
    output_list <- list(rf_train, best_rf, rf_pred, conf_mat, train_set, test_set)
    names(output_list) <- c("Ajuste", "Modelo", "Prediccion", "Matriz_Confusion",
                                "Entrenamiento", "Prueba")
    
    return(output_list)
  }

}