##### Load Libraries #####
library(caret)
library(psych)
library(dplyr)
library(ggplot2)
library(corrplot)
library(cluster)
library(gridExtra)
library(randomForest)
library(rattle)
library(rpart)
library(rpart.plot)
library(klaR)
library(pROC)

##### Load CSV #####
red <- as.data.frame(read.csv("winequality-red.csv"))
# Check out the variables
str(red)

##### EDA #####
# Descriptive statistics
summary(red)

# Distribution of variables
multi.hist(red, global=FALSE)

# Count how much missing values
sum(is.na(red))  # No missing values

##### Categorize Quality #####
red.cat <- (red$quality < 6)
rate <- NULL
# Loop to categorize the quality
for (i in red.cat) {
  if (i == TRUE) {
    # BAD: quality < 6 (1 to 5)
    rate <- c(rate, "bad")
  }
  else {
    # GOOD: quality > 6 (6 to 10)
    rate <- c(rate, "good")
  }
}
red$rate <- rate
# Factor
red$rate <- as.factor(red$rate)

##### Linear Regression EDA - Simple Analysis (Without CV) #####
# LM model
lm.all <- lm(quality ~ ., data=red)
summary(lm.all)

# Remove insignificant features
red.adj <- dplyr::select(red, c('volatile.acidity', 'chlorides', 'free.sulfur.dioxide',
                         'total.sulfur.dioxide', 'pH', 'sulphates', 'alcohol',
                         'quality'))
lm.all2 <- lm(quality ~ ., data=red.adj)
summary(lm.all2)

## Compare models
# Compare AIC
AIC(lm.all)  # Selected
AIC(lm.all2)
# Compare R squared
summary(lm.all)$r.squared # Selected
summary(lm.all2)$r.squared

##### Training & Test Sets #####
## Quality as the dependent variable
set.seed(1)
# Create index
train.index <- sample(nrow(red), nrow(red)*0.8) # 80% is training data
red.train <- red[-13][train.index,] # Training set
red.test <- red[-13][-train.index,] # Test set

## Rate as the dependent variable
set.seed(1)
# Create index
red.train.tar <- red[-12][train.index,] # Training set
red.test.tar <- red[-12][-train.index,] # Test set

##### Diagnostics #####
# Identify possible skewed variables
skewValues <- apply(red.train, 2, skew)
skewSE <- sqrt(6/nrow(red.train)) # Standard error of skewness
# Anything over 2 SEs in skew is potentially problematic
abs(skewValues)/skewSE > 2

## Identify correlated predictors
cor.red <- cor(red[-13]) # Remove rate column
cor.red
# Visualize and cluster by high correlation
corrplot(cor.red, order="hclust")

##### Linear Regression - All Variables #####
## K-fold CV (k = 10)
ctrl <- trainControl(method="cv", number=10)
preproc <- c("center", "scale")

# Train the data for all variables
set.seed(1)
lm.red <- train(quality ~ ., data=red.train, method = "lm", trControl=ctrl,
                preProcess=preproc)
lm.red$results
summary(lm.red)

# Diagnostic plots
par(mfrow=c(1,1))
plot(red.train$quality ~ predict(lm.red), xlab="Predict", ylab="Actual", main="Actual vs. Predicted")
par(mfrow=c(1,1))
plot(resid(lm.red) ~ predict(lm.red), xlab="Predict", ylab="Residuals", main="Predicted Residuals vs. Predicted Values")

# Identify important variables
varImp(lm.red)

## Test on test set
red.pred.test <- predict(lm.red, red.test[-12]) # Remove quality feature for prediction
# Measure test performance
results.all <- postResample(red.pred.test, red.test$quality)
results.all

##### Linear Regression - Advanced Analysis (With CV) #####
# Filter table
red.train.imp <- dplyr::select(red.train, c('volatile.acidity', 'citric.acid',
                                            'chlorides', 'total.sulfur.dioxide',
                                            'pH', 'sulphates', 'alcohol',
                                            'quality'))
red.test.imp <- dplyr::select(red.test, c('volatile.acidity', 'citric.acid',
                                        'chlorides', 'total.sulfur.dioxide',
                                        'pH', 'sulphates', 'alcohol',
                                        'quality'))
# Create model
set.seed(1)
lm.red.imp <- train(quality ~ ., data=red.train.imp, method = "lm",
                    trControl=ctrl, preProcess=preproc)
lm.red.imp$results
summary(lm.red.imp)
# Test on test set
red.pred.test.imp <- predict(lm.red.imp, red.test.imp[-8]) # Remove quality feature for prediction
# Measure test performance
results.imp <- postResample(red.pred.test.imp, red.test.imp$quality)
results.imp

##### K-Means Clustering #####
# Compare cluster value to quality data
clust <- kmeans(red[-13], centers=6, nstart=25) # 6 clusters for 6 quality values
cbind(clust$cluster, red$quality)
par(mfrow=c(1,1))
plot(red$quality, clust$cluster) # Graph not as helpful

# Find optimal k
wss <- numeric(15) # Empty vector to add
for(k in 1:15) {
  clust.temp <- kmeans(red[c(-12,-13)], centers=k, nstart=25) # Remove quality feature
  wss[k] <- sum(clust.temp$withinss) # The smaller wss the better
}
par(mfrow=c(1,1))
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within Sum of Squares") # Elbow at 4

# Clustering with best k value
set.seed(1)
km.red <- kmeans(red[c(-12,-13)], centers=4, nstart=25)

# Set up dataframe
df <- as.data.frame(red[c(-12,-13)]) # Remove quality feature
df$cluster <- factor(km.red$cluster)
centers <- as.data.frame(km.red$centers)

# Based on variable importance
varImp(lm.red) # alcohol, volatile.acidity, sulphates
# Alcohol vs. Volatile Acidity
g1 <- ggplot(data=df, aes(x=alcohol, y=volatile.acidity, color=cluster)) + 
  geom_point() + theme(legend.position="right") +
  geom_point(data=centers, aes(x=alcohol, y=volatile.acidity,
                               color=as.factor(c(1,2,3,4))),
             size=10, alpha=.3, show.legend=FALSE)
# Alcohol vs. Sulphates
g2 <- ggplot(data=df, aes(x=alcohol, y=sulphates, color=cluster)) + 
  geom_point() + geom_point(data=centers, aes(x=alcohol, y=volatile.acidity,
                                              color=as.factor(c(1,2,3,4))),
                            size=10, alpha=.3, show.legend=FALSE)
# Volatile Acidity vs. Sulphates
g3 <- ggplot(data=df, aes(x=volatile.acidity, y=sulphates, color=cluster)) + 
  geom_point() + geom_point(data=centers, aes(x=alcohol, y=volatile.acidity,
                                              color=as.factor(c(1,2,3,4))),
                            size=10, alpha=.3, show.legend=FALSE)

grid.arrange(arrangeGrob(g1 + theme(legend.position="none"),
                         g2 + theme(legend.position="none"),
                         g3 + theme(legend.position="none"),
                         top ="Wine Quality Cluster Analysis (3 Top Important Variables)", ncol=1))

##### Decision Tree #####
# Quality as the dependent variable
fit <- rpart(quality ~ ., method="class", data=red[-13],
                 control=rpart.control(minsplit=1),
                 parms=list(split='information'))
summary(fit)
# Plot the decision tree
rpart.plot(fit)
# Cross validated error
printcp(fit)
plotcp(fit)
# Prune the tree
opt.cp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
fit.pruned <- prune(fit, cp=opt.cp)
rpart.plot(fit.pruned) # Tree wasn't pruned

# Rate as the dependent variable
fit.rate <- rpart(rate ~ ., method="class", data=red[-12],
                  control=rpart.control(minsplit=1),
                  parms=list(split='information'))
summary(fit.rate)
# Plot the decision tree
rpart.plot(fit.rate)
# Cross validated error
printcp(fit.rate)
plotcp(fit.rate)
# Prune the tree
opt.cp.rate <- fit.rate$cptable[which.min(fit.rate$cptable[,"xerror"]),"CP"]
fit.pruned.rate <- prune(fit.rate, cp=opt.cp.rate)
rpart.plot(fit.pruned.rate) # Tree wasn't pruned

##### Random Forest (With CV) - Quality (Numerical) #####
## ntree=5
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.5 <- train(quality ~ ., data=red.train, method = "rf",
                  trControl=ctrl, ntree=5, tuneLength=11,
                  preProcess=preproc)
red.rf.5 # mtry=9
t5 <- red.rf.5$results[8,2] # RMSE

## ntree=10
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.10 <- train(quality ~ ., data=red.train, method = "rf",
                   trControl=ctrl, ntree=10, tuneLength=11,
                   preProcess=preproc)
red.rf.10 # mtry=2
t10 <- red.rf.10$results[1,2] # RMSE

## ntree=50
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.50 <- train(quality ~ ., data=red.train, method = "rf",
                   trControl=ctrl, ntree=50, tuneLength=11,
                   preProcess=preproc)
red.rf.50 # mtry=2
t50 <- red.rf.50$results[1,2] # RMSE

## ntree=100
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.100 <- train(quality ~ ., data=red.train, method = "rf",
                    trControl=ctrl, ntree=100, tuneLength=11,
                    preProcess=preproc)
red.rf.100 # mtry=4
t100 <- red.rf.100$results[3,2] # RMSE

# Default value (ntree=500)
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.500 <- train(quality ~ ., data=red.train, method = "rf",
                    trControl=ctrl, ntree=500, tuneLength=11,
                    preProcess=preproc)
red.rf.500 # mtry=4
t500 <- red.rf.500$results[3,2] # RMSE

## Assess the best RMSE value
ntree <- cbind(c(5, 10, 50, 100, 500), c(t5, t10, t50, t100, t500))
ntree[which.min(ntree[,2])] # ntree=500 has the min. RMSE

# Plot a sample tree from the selected random forest
getTree(red.rf.50$finalModel)

# Plot the error vs. number of trees
par(mfrow=c(1,1))
plot(red.rf.50$finalModel, main='Training Error vs. the Number of Trees')

# Random forest on test set
set.seed(1)
# Predict results
red.rf.test <- predict(red.rf.50, red.test[-12])
# Measure test performance
results.rf <- postResample(red.rf.test, red.test$quality)

##### Random Forest (With CV) - Rate (Categorical) #####
## ntree=5
# Tune length is the max # of parameters = 11
set.seed(1)
rate.rf.5 <- train(rate ~ ., data=red.train.tar, method = "rf",
                   trControl=ctrl, ntree=5, tuneLength=11,
                   preProcess=preproc)
rate.rf.5 # mtry=8
t5.rate <- rate.rf.5$results[7,2] # Accuracy

## ntree=10
# Tune length is the max # of parameters = 11
set.seed(1)
rate.rf.10 <- train(rate ~ ., data=red.train.tar, method = "rf",
                    trControl=ctrl, ntree=10, tuneLength=11,
                    preProcess=preproc)
rate.rf.10 # mtry=5
t10.rate <- rate.rf.10$results[4,2] # Accuracy

## ntree=50
# Tune length is the max # of parameters = 11
set.seed(1)
rate.rf.50 <- train(rate ~ ., data=red.train.tar, method = "rf",
                    trControl=ctrl, ntree=50, tuneLength=11,
                    preProcess=preproc)
rate.rf.50 # mtry=2
t50.rate <- rate.rf.50$results[1,2] # Accuracy

## ntree=100
# Tune length is the max # of parameters = 11
set.seed(1)
rate.rf.100 <- train(rate ~ ., data=red.train.tar, method = "rf",
                     trControl=ctrl, ntree=100, tuneLength=11,
                     preProcess=preproc)
rate.rf.100 # mtry=2
t100.rate <- rate.rf.100$results[1,2] # Accuracy

# Default value (ntree=500)
# Tune length is the max # of parameters = 11
set.seed(1)
rate.rf.500 <- train(rate ~ ., data=red.train.tar, method = "rf",
                     trControl=ctrl, ntree=500, tuneLength=11,
                     preProcess=preproc)
rate.rf.500 # mtry=2
t500.rate <- rate.rf.500$results[1,2] # Accuracy

## Assess the best RMSE value
ntree.rate <- cbind(c(5, 10, 50, 100, 500),
               c(t5.rate, t10.rate, t50.rate, t100.rate, t500.rate))
ntree.rate[which.min(ntree.rate[,2])] # ntree=5 has the max accuracy

# Plot a sample tree from the selected random forest
getTree(rate.rf.5$finalModel)

# Plot the error vs. number of trees
par(mfrow=c(1,1))
plot(rate.rf.5$finalModel, main='Training Error vs. the Number of Trees')

# Random forest on test set
set.seed(1)
# Predict results
rate.rf.test <- predict(rate.rf.5, red.test.tar[-12])
# Measure test performance
results.rf.rate <- postResample(rate.rf.test, red.test.tar$rate)
results.rf.rate

##### SVM (With CV) #####
# Control parameters
ctrl.svm <- trainControl(method="repeatedcv", number=10, repeats=3)

## Linear SVM
set.seed(1)
svm.red <- train(rate ~., data=red.train.tar, method="svmLinear",
                 trControl=ctrl.svm, preProcess=preproc,
                 tuneLength=10)
# Predict results
red.svm.test <- predict(svm.red, red.test.tar[-12])
# Measure test performance
results.svm <- postResample(red.svm.test, red.test.tar$rate)
results.svm

## Radial SVM
set.seed(1)
svm.rad.red <- train(rate ~., data=red.train.tar, method="svmRadial",
                 trControl=ctrl.svm, preProcess=preproc,
                 tuneLength=10)
# Predict results
red.svm.rad.test <- predict(svm.rad.red, red.test.tar[-12])

## Measure test performance
# Accuracy
results.svm.rad <- postResample(red.svm.rad.test, red.test.tar$rate)
results.svm.rad

##### Naive Bayes (With CV) #####
set.seed(1)
nb.red <- train(rate ~., data=red.train.tar, method="nb",
                 trControl=ctrl, preProcess=preproc,
                 tuneLength=10)
# Predict results
red.nb.test <- predict(nb.red, red.test.tar[-12])
# Measure test performance
results.nb <- postResample(red.nb.test, red.test.tar$rate)
results.nb

##### Test Data Evaluation - Quantitative #####
# Model setup
model_test <- list("LM - All Vars"=lm.red, "LM - Sig Vars"=lm.red.imp,
                "Random Forest"=red.rf.50)
# Resample data
red.resamples <- resamples(model_test)
# Plot performances
bwplot(red.resamples, metric="RMSE", main="Quality Model Performance - RMSE")
bwplot(red.resamples, metric="Rsquared",
       main="Quality Model Performance - Rsquared")

##### Test Data Evaluation - Qualitative #####
# Model setup
model_cat_name <- c("Linear SVM", "Radial SVM", "Random Forest",
                    "Naive Bayes")
# Create dataframe
red.cat <- data.frame(
  results.svm, # Linear SVM
  results.svm.rad, # Radial SVM
  results.rf.rate, # Random Forest
  results.nb # Naive Bayes
)
red.cat<- t(red.cat)
red.cat <- data.frame(cbind(red.cat, Model=model_cat_name))

# Plot the accuracy
ggplot(data=red.cat, aes(x=Model, y=Accuracy)) +
  geom_bar(stat="identity", color="black", fill="steelblue") +
  labs(title="Rating Model Performance - Accuracy")

# Plot the kappa
ggplot(data=red.cat, aes(x=Model, y=Kappa)) +
  geom_bar(stat="identity", color="black", fill="orangered4") +
  labs(title="Rating Model Performance - Kappa")
