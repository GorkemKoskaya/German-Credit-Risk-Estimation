# Required libraries
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(pROC)
library(ROCR)
library(MLmetrics)
library(reshape2)
library(ggplot2)

# Load the dataset
data <- read.csv("german_credit.csv")

# Convert target variable to factor
data$Class <- as.factor(data$Class)

# Train/test split
set.seed(123)
index <- createDataPartition(data$Class, p = 0.8, list = FALSE)
train <- data[index, ]
test <- data[-index, ]

# Model 1: Decision Tree (rpart)
dt_model <- rpart(Class ~ ., data = train, method = "class")
dt_pred <- predict(dt_model, test, type = "prob")[,2]
dt_auc <- roc(test$Class, dt_pred)$auc
dt_acc <- mean(predict(dt_model, test, type = "class") == test$Class)

# Model 2: Random Forest
rf_model <- randomForest(Class ~ ., data = train)
rf_pred <- predict(rf_model, test, type = "prob")[,2]
rf_auc <- roc(test$Class, rf_pred)$auc
rf_acc <- mean(predict(rf_model, test) == test$Class)

# Model 3: SVM (linear)
svm_model <- svm(Class ~ ., data = train, probability = TRUE)
svm_pred <- attr(predict(svm_model, test, probability = TRUE), "probabilities")[,2]
svm_auc <- roc(test$Class, svm_pred)$auc
svm_acc <- mean(predict(svm_model, test) == test$Class)

# Performance table
results <- data.frame(
  Model = c("Decision Tree", "Random Forest", "SVM"),
  Accuracy = c(dt_acc, rf_acc, svm_acc),
  AUC = c(dt_auc, rf_auc, svm_auc)
)

print(results)

# Draw ROC curves
plot(roc(test$Class, dt_pred), col="blue", main="ROC Curves", legacy.axes = TRUE)
plot(roc(test$Class, rf_pred), col="green", add=TRUE)
plot(roc(test$Class, svm_pred), col="purple", add=TRUE)
legend("bottomright", legend=c("Decision Tree", "Random Forest", "SVM"),
       col=c("blue", "green", "purple", "red"), lwd=2)

# Accuracy chart
ggplot(results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  coord_cartesian(ylim = c(0.6, 0.9)) +
  labs(
    title = "Model Accuracy Comparison",
    x = "Model",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5)

# Class distribution
class_counts <- table(data$Class)
class_df <- as.data.frame(class_counts)
colnames(class_df) <- c("Class", "Count")

# Create percentages and labels
class_df$Percentage <- round(100 * class_df$Count / sum(class_df$Count), 1)
class_df$Label <- paste0(class_df$Class, "\n", class_df$Count, " (", class_df$Percentage, "%)")

# Soft pastel color palette
soft_colors <- c("Good" = "#A5D8FF",  # Pastel Blue
                 "Bad" = "#FFD6A5")   # Pastel Orange

# Pie chart
ggplot(class_df, aes(x = "", y = Count, fill = Class)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  labs(title = "Class Distribution") +
  theme_void() +
  geom_text(aes(label = Label), position = position_stack(vjust = 0.5), size = 5) +
  scale_fill_manual(values = soft_colors) +
  theme(legend.position = "none")

# RANDOM FOREST sampling methods
ctrl_up <- trainControl(method = "none", sampling = "up", classProbs = TRUE)
ctrl_down <- trainControl(method = "none", sampling = "down", classProbs = TRUE)
ctrl_smote <- trainControl(method = "none", sampling = "smote", classProbs = TRUE)
ctrl_rose <- trainControl(method = "none", sampling = "rose", classProbs = TRUE)

# UpSampling
rf_up <- train(Class ~ ., data = train, method = "rf", trControl = ctrl_up)
pred_up <- predict(rf_up, test)
prob_up <- predict(rf_up, test, type = "prob")[, "Bad"]

# DownSampling
rf_down <- train(Class ~ ., data = train, method = "rf", trControl = ctrl_down)
pred_down <- predict(rf_down, test)
prob_down <- predict(rf_down, test, type = "prob")[, "Bad"]

# SMOTE
rf_smote <- train(Class ~ ., data = train, method = "rf", trControl = ctrl_smote)
pred_smote <- predict(rf_smote, test)
prob_smote <- predict(rf_smote, test, type = "prob")[, "Bad"]

# ROSE
rf_rose <- train(Class ~ ., data = train, method = "rf", trControl = ctrl_rose)
pred_rose <- predict(rf_rose, test)
prob_rose <- predict(rf_rose, test, type = "prob")[, "Bad"]

# Create performance table
performance <- data.frame(
  Method = c("UpSampling", "DownSampling", "SMOTE", "ROSE"),
  Accuracy = c(
    mean(pred_up == test$Class),
    mean(pred_down == test$Class),
    mean(pred_smote == test$Class),
    mean(pred_rose == test$Class)
  ),
  AUC = c(
    roc(test$Class, prob_up)$auc,
    roc(test$Class, prob_down)$auc,
    roc(test$Class, prob_smote)$auc,
    roc(test$Class, prob_rose)$auc
  ),
  F1_Bad = c(
    F1_Score(pred_up, test$Class, positive = "Bad"),
    F1_Score(pred_down, test$Class, positive = "Bad"),
    F1_Score(pred_smote, test$Class, positive = "Bad"),
    F1_Score(pred_rose, test$Class, positive = "Bad")
  )
)

print(performance)

# Evaluation chart using real performance values
performance <- data.frame(
  Method = c("UpSampling", "DownSampling", "SMOTE", "ROSE"),
  Accuracy = c(0.780, 0.680, 0.780, 0.535),
  AUC = c(0.8170238, 0.7993452, 0.8195238, 0.7388095),
  F1_Bad = c(0.6140351, 0.6000000, 0.5849057, 0.5326633)
)

# Convert to long format for plotting
perf_long <- melt(performance, id.vars = "Method", variable.name = "Metric", value.name = "Value")

# Plot: Performance Comparison by Sampling Method
ggplot(perf_long, aes(x = Method, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = 0.8),
            vjust = -0.5, size = 4.5) +
  labs(
    title = "Performance Comparison by Sampling Method",
    x = "Sampling Method",
    y = "Value"
  ) +
  coord_cartesian(ylim = c(0.5, 0.85)) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 13),
    legend.text = element_text(size = 12)
  )

# Final Model (Random Forest with UpSampling and Tuning)

# Ensure class levels are consistent
train$Class <- factor(train$Class, levels = c("Bad", "Good"))
test$Class <- factor(test$Class, levels = c("Bad", "Good"))

# TrainControl: UpSampling + 10-fold CV + ROC scoring
ctrl <- trainControl(
  method = "cv",
  number = 10,
  sampling = "up",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

# Grid for hyperparameter tuning (mtry values)
grid <- expand.grid(mtry = c(2, 4, 6, 8, 10, 12, 14))

# Train the model
set.seed(42)
rf_up_final <- train(
  Class ~ .,
  data = train,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid
)

print(rf_up_final)
plot(rf_up_final)

# Evaluate final model on test data
pred_up_final <- predict(rf_up_final, test)
prob_up_final <- predict(rf_up_final, test, type = "prob")[, "Bad"]

# AUC, Accuracy, F1 for final model
auc_up_final <- roc(test$Class, prob_up_final)$auc
acc_up_final <- mean(pred_up_final == test$Class)
f1_up_final <- F1_Score(pred_up_final, test$Class, positive = "Bad")

cat("Final Model - Accuracy:", round(acc_up_final, 3), "\n")
cat("Final Model - AUC:", round(auc_up_final, 3), "\n")
cat("Final Model - F1 Score (Bad):", round(f1_up_final, 3), "\n")

# Confusion Matrix
conf_matrix <- confusionMatrix(pred_up_final, test$Class, positive = "Bad")
print(conf_matrix)
