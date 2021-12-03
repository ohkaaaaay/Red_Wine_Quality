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

##### Load CSV #####
red <- as.data.frame(read.csv("winequality-red.csv"))
# Check out the variables
str(red)

##### Question 1 - Descriptive Statistics #####
summary(red)
# Standard deviation
red %>% summarise_if(is.numeric, sd)

##### Question 2 - Distribution of Variables #####
multi.hist(red, global=FALSE)

##### Question 3 - Find Missing Values or Errors #####
# Count how much missing values
sum(is.na(red))  # No missing values

##### Question 4 - Convert to Numerical Form #####
# All variables are of numerical form
str(red)

##### Question 5 - Build Model #####
# Focus only the linear regression and random foreset section.
# Clustering was also performed but caret wasn't used.

################
### Linear Regression - Simple Analysis (Without CV)
# LM model
lm.all <- lm(quality ~ ., data=red)
summary(lm.all)

# Remove insignificant features
red.adj <- select(red, c('volatile.acidity', 'chlorides', 'free.sulfur.dioxide',
                         'total.sulfur.dioxide', 'pH', 'sulphates', 'alcohol',
                         'quality'))
lm.all2 <- lm(quality ~ ., data=red.adj)
summary(lm.all2)

## Compare models
# Compare AIC
AIC(lm.all)
AIC(lm.all2) # Selected
# Compare R squared
summary(lm.all)$r.squared # Selected
summary(lm.all2)$r.squared

## Training & test sets
set.seed(1)
# Create index
train.index <- sample(nrow(red), nrow(red)*0.8) # 80% is training data
red.train <- red[train.index,] # Training set
red.test <- red[-train.index,] # Test set

################
### Diagnostics
# Identify possible skewed variables
skewValues <- apply(red.train, 2, skew)
skewSE <- sqrt(6/nrow(red.train)) # Standard error of skewness
# Anything over 2 SEs in skew is potentially problematic
abs(skewValues)/skewSE > 2

## Identify correlated predictors
cor.red <- cor(red)
cor.red
# Visualize and cluster by high correlation
corrplot(cor.red, order="hclust")

################
### Linear Regression - Advanced Analysis (With CV) - All Variables
## K-fold CV (k = 10)
ctrl <- trainControl(method="cv", number=10)

# Train the data for all variables
set.seed(1)
lm.red <- train(quality ~ ., data=red.train, method = "lm", trControl=ctrl)
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

################
### Linear Regression - Advanced Analysis (With CV) - Remove Insignificant Variables
# Filter table
red.train.imp <- select(red.train, c('volatile.acidity', 'citric.acid',
                                     'chlorides', 'total.sulfur.dioxide',
                                     'pH', 'sulphates', 'alcohol',
                                     'quality'))
red.test.imp <- select(red.test, c('volatile.acidity', 'citric.acid',
                                     'chlorides', 'total.sulfur.dioxide',
                                     'pH', 'sulphates', 'alcohol',
                                     'quality'))

# Train the data for all variables
set.seed(1)
lm.red.imp <- train(quality ~ ., data=red.train.imp, method = "lm",
                    trControl=ctrl)
lm.red.imp$results
summary(lm.red.imp)

## Test on test set
red.pred.test.imp <- predict(lm.red.imp, red.test[-8]) # Remove quality feature for prediction
# Measure test performance
results.imp <- postResample(red.pred.test.imp, red.test.imp$quality)
results.imp

################
### K-Means Clustering
# # Compare cluster value to quality data (COMPLETE FOR FINAL PROJECT)
# clust <- kmeans(red, centers=6, nstart=25) # 6 clusters for 6 quality values
# cbind(clust$cluster, red$quality)
# par(mfrow=c(1,1))
# plot(red$quality, clust$cluster) # Graph not as helpful

# Find optimal k
wss <- numeric(15) # Empty vector to add
for(k in 1:15) {
  clust.temp <- kmeans(red[-12], centers=k, nstart=25) # Remove quality feature
  wss[k] <- sum(clust.temp$withinss) # The smaller wss the better
}
par(mfrow=c(1,1))
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within Sum of Squares") # Elbow at 4

# Clustering with best k value
set.seed(1)
km.red <- kmeans(red[-12], centers=4, nstart=25)

# Set up dataframe
df <- as.data.frame(red[-12]) # Remove quality feature
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

################
### Random Forest (With CV)
## ntree=5
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.5 <- train(quality ~ ., data=red.train, method = "rf",
                 trControl=ctrl, ntree=5, tuneLength=11)
red.rf.5 # mtry=6
t5 <- red.rf.5$results[5,2] # RMSE

## ntree=10
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.10 <- train(quality ~ ., data=red.train, method = "rf",
                   trControl=ctrl, ntree=10, tuneLength=11)
red.rf.10 # mtry=5
t10 <- red.rf.10$results[4,2] # RMSE

## ntree=50
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.50 <- train(quality ~ ., data=red.train, method = "rf",
                   trControl=ctrl, ntree=50, tuneLength=11)
red.rf.50 # mtry=5
t50 <- red.rf.50$results[4,2] # RMSE

## ntree=100
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.100 <- train(quality ~ ., data=red.train, method = "rf",
                    trControl=ctrl, ntree=100, tuneLength=11)
red.rf.100 # mtry=5
t100 <- red.rf.100$results[4,2] # RMSE

# Default value (ntree=500)
# Tune length is the max # of parameters = 11
set.seed(1)
red.rf.500 <- train(quality ~ ., data=red.train, method = "rf",
                    trControl=ctrl, ntree=500, tuneLength=11)
red.rf.500 # mtry=4
t500 <- red.rf.500$results[3,2] # RMSE

## Assess the best RMSE value
ntree <- cbind(c(5, 10, 50, 100, 500), c(t5, t10, t50, t100, t500))
ntree[which.min(ntree[,2])] # ntree=500 has the min. RMSE

# Plot a sample tree from the selected random forest
getTree(red.rf.500$finalModel)

# Plot the error vs. number of trees
par(mfrow=c(1,1))
plot(red.rf.500$finalModel, main='Training Error vs. the Number of Trees')

# Random forest on test set
set.seed(1)
# Tune length is the max # of parameters = 11
red.rf.test <- predict(red.rf.500, red.test[-12])
# Measure test performance
results.rf <- postResample(red.rf.test, red.test$quality)

################
### Test Data Evaluation
# Model setup
model_test <- list("LM - All Vars"=lm.red, "LM - Sig Vars"=lm.red.imp,
                "Random Forest"=red.rf.500)
# Resample data
red.resamples <- resamples(model_test)
# Plot performances
bwplot(red.resamples, metric="RMSE")
bwplot(red.resamples, metric="Rsquared")