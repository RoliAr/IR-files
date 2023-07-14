#install.packages(c("tidyverse", "caret", "Metrics", "ggplot2", "fastDummies"))
library(tidyverse)
library(caret)
library(Metrics)
library(ggplot2)
#install.packages("rpart")
library(rpart)
library(rpart.plot)


# Start the timer
start_time <- Sys.time()

data <- read.csv("C:/Users/ROEYE/Downloads/insurance.csv")

# Convert categorical variables to factor
data$sex <- as.factor(data$sex)
data$smoker <- as.factor(data$smoker)
data$region <- as.factor(data$region)

# One-hot encoding
data <- fastDummies::dummy_cols(data, select_columns = c("sex", "smoker", "region"))

N <- nrow(data)
set.seed(123)

# Set training size
training_size <- round(N * 0.6)

# Set training rows
training_cases <- sample(N, training_size)

# Make training df
training <- data[training_cases,]

# Make test df = not in training cases
test <- data[-training_cases,]

#####################Regression#############################################################################

model <- lm(charges ~ ., data = training)
summary(model)

# Predict
predictions <- predict(model, test)

# Evaluate
# Find true car values
observations <- test$charges

# Compute the errors
errors <- observations - predictions

# Compute KPIS = rmse = root mean squared error
rmse <- sqrt(mean(errors^2))

# Mean absolute % error
mape <- mean(abs(errors / observations))

# Create residuals
residuals <- test$charges - predictions

# Create data frame for plotting residuals
residuals_data <- data.frame(Predicted = predictions, Residuals = residuals)

# Plot Actual vs Predicted charges
ggplot(residuals_data, aes(x = observations, y = predictions)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  ggtitle("Actual vs Predicted charges") +
  xlab("Actual charges") +
  ylab("Predicted charges")

# Plot residuals vs predicted values
ggplot(residuals_data, aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  ggtitle("Residuals vs Predicted charges") +
  xlab("Predicted charges") +
  ylab("Residuals")

# End the timer
end_time <- Sys.time()

# Print the time taken
print(end_time - start_time)


##########################Decision Tree###############################################################
#install.packages("rpart")
library(rpart)
library(rpart.plot)

# Start the timer
start_time_tree <- Sys.time()


# Create decision tree model
model_tree <- rpart(charges ~ ., data = training, method = "anova")
printcp(model_tree)
summary(model_tree)

# Predict
predictions_tree <- predict(model_tree, test)

# Compute the errors for decision tree model
errors_tree <- test$charges - predictions_tree

# Compute RMSE and MAPE for decision tree model
rmse_tree <- sqrt(mean(errors_tree^2))
mape_tree <- mean(abs(errors_tree / test$charges))

# Print metrics for decision tree model
print(paste("Decision Tree - RMSE: ", rmse_tree))
print(paste("Decision Tree - MAPE: ", mape_tree))

# Plot the tree
rpart.plot(model_tree)

# Plot actual vs predicted values
ggplot(data.frame(Actual = test$charges, Predicted = predictions_tree), aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  ggtitle("Actual vs Predicted charges (Decision Tree)") +
  xlab("Actual charges") +
  ylab("Predicted charges")

# Calculate residuals
residuals_tree <- test$charges - predictions_tree

# Create data frame for plotting
residuals_data_tree <- data.frame(Predicted = predictions_tree, Residuals = residuals_tree)

# Plot residuals vs predicted values
ggplot(residuals_data_tree, aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  ggtitle("Residuals vs Predicted charges (Decision Tree)") +
  xlab("Predicted charges") +
  ylab("Residuals")

# End the timer
end_time_tree <- Sys.time()

# Print the time taken
print(end_time_tree - start_time_tree)

###################Random Forest#####################################################################################

# Install necessary package
#install.packages("randomForest")
# Load the package
library(randomForest)

# Start the timer
start_time_rf <- Sys.time()

# Create random forest model
model_rf <- randomForest(charges ~ ., data = training, ntree=500)

# Print model summary
print(summary(model_rf))

# Predict
predictions_rf = predict(model_rf, test)

# Compute the errors for random forest model
errors_rf = test$charges - predictions_rf

# Compute RMSE and MAPE for random forest model
rmse_rf = sqrt(mean(errors_rf^2))
mape_rf = mean(abs(errors_rf/test$charges))

# Print metrics for random forest model
print(paste("Random Forest - RMSE: ", rmse_rf))
print(paste("Random Forest - MAPE: ", mape_rf))
# Create a data frame to hold the actual and predicted values
df <- data.frame(Actual = test$charges, Predicted = predictions_rf)

# Plot actual vs predicted values
library(ggplot2)
ggplot(df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  ggtitle("Actual vs Predicted") +
  xlab("Actual Charges") +
  ylab("Predicted Charges")

# Calculate residuals
residuals_rf <- test$charges - predictions_rf

# Create data frame for plotting
residuals_data_rf <- data.frame(Predicted = predictions_rf, Residuals = residuals_rf)

# Plot residuals vs predicted values
ggplot(residuals_data_rf, aes(x = Predicted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  ggtitle("Residuals vs Predicted charges (Random Forest)") +
  xlab("Predicted charges") +
  ylab("Residuals")

# End the timer
end_time_rf <- Sys.time()

# Print the time taken
print(end_time_rf - start_time_rf)


########################################################################################################

###########Neural Network#######################################################################################
# Load the neuralnet package
library(nnet)

# Start the timer
start_time_nn <- Sys.time()

model_nn <- nnet(charges ~ ., data = training, size = 16, maxit=1000)
#model_nn <- nnet(charges ~ ., data = training, size = c(6, 1))

summary(model_nn)

# Predict
predictions_nn <- predict(model_nn, test, type = "raw")

# Evaluate
# Find true car values
observations <- test$charges

# Compute the errors
errors_nn <- observations - predictions

# Compute KPIS
rmse_nn <- sqrt(mean(errors^2))
mape_nn <- mean(abs(errors / observations))

# Print evaluation metrics
print(paste("Neural Network Model - RMSE:", rmse_nn))
print(paste("Neural Network Model - MAPE:", mape_nn))

# Plot actual vs predicted
plot_data <- data.frame(Actual = observations, Predicted = predictions_nn)
plot(plot_data$Actual, plot_data$Predicted, xlab = "Actual charges", ylab = "Predicted charges",
     main = "Actual vs Predicted charges (Neural Network)", col = "blue", pch = 16)
abline(a = 0, b = 1, col = "red")  # Adding a diagonal reference line

# Plot residuals vs predicted
residuals_nn <- errors_nn
plot(predictions_nn, residuals_nn, xlab = "Predicted charges", ylab = "Residuals",
     main = "Residuals vs Predicted charges (Neural Network)", col = "blue", pch = 16)
abline(h = 0, col = "red")  # Adding a horizontal line at y = 0

# End the timer
end_time_nn <- Sys.time()

# Print the time taken
print(end_time_nn - start_time_nn)

