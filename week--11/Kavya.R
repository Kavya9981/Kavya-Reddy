library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                         1, 0)

library(xgboost)
library(caret)

# Function to evaluate XGBoost model with different dataset sizes
evaluate_xgboost <- function(data, size) {
  # Sample data to get requested size
  set.seed(123)  # For reproducibility
  indices <- sample(1:nrow(data), size = min(size, nrow(data)))
  sample_data <- data[indices, ]
  
  # Split data into training (80%) and testing (20%) sets
  train_indices <- createDataPartition(sample_data$outcome, p = 0.8, list = FALSE)
  train_data <- sample_data[train_indices, ]
  test_data <- sample_data[-train_indices, ]
  
  # Prepare matrices for XGBoost
  train_matrix <- as.matrix(train_data[, -ncol(train_data)])
  train_label <- train_data$outcome
  test_matrix <- as.matrix(test_data[, -ncol(test_data)])
  test_label <- test_data$outcome
  
  # Fit XGBoost model with simple cross-validation
  start_time <- Sys.time()
  xgb_cv <- xgb.cv(
    data = train_matrix,
    label = train_label,
    nrounds = 100,
    nfold = 5,
    objective = "binary:logistic",
    eval_metric = "error",
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  # Get optimal number of rounds
  best_nrounds <- which.min(xgb_cv$evaluation_log$test_error_mean)
  
  # Fit final model with optimal rounds
  xgb_model <- xgboost(
    data = train_matrix,
    label = train_label,
    nrounds = best_nrounds,
    objective = "binary:logistic",
    eval_metric = "error",
    verbose = 0
  )
  end_time <- Sys.time()
  time_taken <- difftime(end_time, start_time, units = "secs")
  
  # Make predictions on test set
  predictions <- predict(xgb_model, test_matrix)
  pred_class <- ifelse(predictions > 0.5, 1, 0)
  
  # Calculate metrics
  confusion_matrix <- confusionMatrix(factor(pred_class), factor(test_label))
  accuracy <- confusion_matrix$overall["Accuracy"]
  
  # Return results
  return(list(
    method = "XGBoost with simple cross-validation",
    dataset_size = size,
    test_accuracy = accuracy,
    time_taken_seconds = as.numeric(time_taken)
  ))
}

# Run evaluation for different dataset sizes
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)
results <- lapply(sizes, function(size) evaluate_xgboost(dfdata, size))

# Display results in a nice table
results_df <- do.call(rbind, lapply(results, function(x) data.frame(
  Method = x$method,
  Dataset_Size = x$dataset_size,
  Test_Accuracy = round(x$test_accuracy, 4),
  Time_Taken_Seconds = round(x$time_taken_seconds, 2)
)))

# Print results
print(results_df)


library(caret)
library(xgboost)

# Function to evaluate XGBoost via caret with 5-fold CV
evaluate_xgboost_caret <- function(data, size) {
  # Sample data to get requested size
  set.seed(123)  # For reproducibility
  if (size > nrow(data)) {
    # If requested size is larger than available data, just use all data
    sample_data <- data
  } else {
    indices <- sample(1:nrow(data), size = size)
    sample_data <- data[indices, ]
  }
  
  # Split data into training (80%) and testing (20%) sets
  set.seed(456)
  train_indices <- createDataPartition(sample_data$outcome, p = 0.8, list = FALSE)
  train_data <- sample_data[train_indices, ]
  test_data <- sample_data[-train_indices, ]
  
  # Convert outcome to factor for classification
  train_data$outcome <- as.factor(train_data$outcome)
  test_data$outcome <- as.factor(test_data$outcome)
  
  # Set up training control with 5-fold CV
  train_control <- trainControl(
    method = "cv",
    number = 5,
    verboseIter = FALSE
  )
  
  # Set up a simpler grid with fewer parameters
  tuning_grid <- expand.grid(
    nrounds = 50,
    max_depth = 3,
    eta = 0.3,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
  
  # Train the model and time it
  start_time <- Sys.time()
  # Use try() to catch any errors
  model_result <- try({
    xgb_model <- train(
      x = train_data[, -ncol(train_data)],  # Predictors only
      y = train_data$outcome,               # Outcome
      method = "xgbTree",
      trControl = train_control,
      tuneGrid = tuning_grid,
      verbose = FALSE
    )
    
    # Make predictions on test set
    predictions <- predict(xgb_model, test_data)
    
    # Calculate metrics
    confusion_matrix <- confusionMatrix(predictions, test_data$outcome)
    accuracy <- confusion_matrix$overall["Accuracy"]
    
    list(model = xgb_model, accuracy = accuracy)
  }, silent = TRUE)
  
  end_time <- Sys.time()
  time_taken <- difftime(end_time, start_time, units = "secs")
  
  # Check if model training was successful
  if (inherits(model_result, "try-error")) {
    return(list(
      method = "XGBoost via caret with 5-fold CV",
      dataset_size = size,
      test_accuracy = NA,
      time_taken_seconds = as.numeric(time_taken),
      error = TRUE
    ))
  } else {
    return(list(
      method = "XGBoost via caret with 5-fold CV",
      dataset_size = size,
      test_accuracy = model_result$accuracy,
      time_taken_seconds = as.numeric(time_taken),
      error = FALSE
    ))
  }
}

# Run evaluation for different dataset sizes
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)
results <- list()

for (size in sizes) {
  cat("Processing dataset size:", size, "\n")
  result <- evaluate_xgboost_caret(dfdata, size)
  results[[length(results) + 1]] <- result
}

# Display results in a nice table
results_df <- do.call(rbind, lapply(results, function(x) {
  data.frame(
    Method = x$method,
    Dataset_Size = x$dataset_size,
    Test_Accuracy = if (x$error) NA else round(x$test_accuracy, 4),
    Time_Taken_Seconds = round(x$time_taken_seconds, 2),
    Status = if (x$error) "Failed" else "Success"
  )
}))

# Print results
print(results_df)