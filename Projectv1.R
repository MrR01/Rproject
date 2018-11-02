ds1a <- data.table::fread("C:/../../Reviews.csv")

#str function used to look at the object to find out what's the Structure in this object. object is in ds1a
#output -- ProductId, UserId, ProfileName, Summary and Text are factors. Rest are int  
str(ds1a)

#check if a df exist...
#exists("train")

#Quick Peek
summary(ds1a)
dim(ds1a)
head(ds1a)

## returns TRUE of dataset has missing values
anyNA(ds1a, recursive = FALSE)

##gives you the number of missing value per each columns. Returns TRUE of dataset has missing values
#output -- ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary and Text have missing values   
colSums(is.na(ds1a))
#or
#where the value of `0` will be `TRUE` and all other values `>0` FALSE
!!colSums(is.na(ds1a))

#Check which rows has missing values
ds1a[rowSums(is.na(ds1a)) > 0,]

#remove the rows with missing data
ds1a[complete.cases(ds1a),]
dim(ds1a)


#remove multiple columns
ds1a$Id <- NULL
ds1a$ProfileName <- NULL
ds1a$ProductId <- NULL
ds1a$UserId <- NULL
ds1a$Time <- NULL
ds1a$Summary <- NULL
ds1a$Text <- NULL
ds1a$HelpfulnessNumerator <-NULL

#check the remaining columns
colnames(ds1a)

#install.packages("dplyr")
library(dplyr)
#Return rows with matching conditions
ds <- filter(ds1a, HelpfulnessDenominator >3, Score <=3)
ds
str(ds)

# Random sampling
samplesize = 0.60 * nrow(ds)
set.seed(80)
index = sample(seq_len(nrow(ds)), size = samplesize)

# Create training and test set
train = ds[ index, ]
test = ds[ -index, ]

#binary classification i.e 0 or 1, yes or no etc
#Regression involves estimating or predicting a response. 
#Classification is identifying group membership
#From above notes Regression is the reason I have choosen from the data I have seen

# prepare training scheme
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)


#--------------------------------------------------------------------------------------
##Ensembles of Trees - Linear Regression and Stochastic Gradient Boosting
#Linear Regression
#Opted for Linear Regression because RF giving warning msg "In randomForest.default(x, y, mtry = 
#param$mtry, ...) : The response has five or fewer unique values.  Are you sure you want to do 
#regression?
#So therefore RF is suited for more than five distinct values. 
#-----------------

#K-fold Cross Validation for linear regression
#---------------------------------------------
library(lattice)
library(ggplot2)
library(caret)
library(tidyverse)
library(DAAG)

set.seed(123)
# Building the Linear Regression model
linearMod <- train(Score~., data = train, method = "lm",
               trControl = train.control)

print(linearMod)
summary(linearMod)


RMSE_linearMod <- linearMod$results$RMSE
pcv <- predict(linearMod, test)
errorcv <- (pcv- test$Score)
RMSE_NewDatacv <- sqrt(mean(errorcv^2))

# predict the test
lmPred <- linearMod %>% predict(test)
head(lmPred)

#Compute model accuracy rate
mean(lmPred  == test$Score)

# Compute the prediction error RMSE. The lower the RMSE, the better the model.
RMSE(lmPred, test$Score)


#The RMSE is calculated, which assesses accuracy of the model. 
#After that, the model is applied on the test data to obtain the RMSE.
error <- (lmPred- test$Score)
error



Method1 <- c("K-fold Cross Validation")
ModelRMSE1 <- c(RMSE_NewDatacv)
table <- data.frame(Method1, ModelRMSE1, RMSE_NewDatacv)
library(kableExtra)
kable(table) %>% kable_styling(c("striped", "bordered")) %>%column_spec(2:3, border_left = T)
#RMSE for the train test set and K-fold Cross Validation are the same




confusionMatrix(lmPred, test$Score) 



#SVM (Support Vector Machine)
#----------------------------
library(lattice)
library(ggplot2)
library(caret)
library(e1071)
library(tidyverse)

set.seed(123)

svmModel <- train(Score~., data=train, method="svmRadial", trControl=train.control)
head(svmModel)
summary(svmModel)



#plot
plot(Score ~ HelpfulnessDenominator, data = train)
car::scatterplot(Score ~ HelpfulnessDenominator, data = train)

plot(train$Score, train$HelpfulnessDenominator, main="Scatterplot Example", 
     xlab="Score ", ylab="Unhelpful Comments", pch=19) 

abline(lm(train$Score~ train$HelpfulnessDenominator), col="red") # regression line (y~x) 
lines(lowess(train$HelpfulnessDenominator,train$Score), col="blue") # lowess line (x,y) 

#Use the predictions on the data
svmPred <- svmModel %>% predict(test)
head(svmPred)

# Compute model accuracy rate
mean(svmPred == test$Score)


#svm tuning
svm_tune <- tune.svm(Score~., data = train, gamma = 10^(-5:-1), cost = 10^(-3:1))
summary(svm_tune)
print(svm_tune)
plot(svm_tune)

## Select the best model out of 1100 trained models and compute RMSE
#Find out the best model
BstModel=svm_tune$best.model
#Predict Score using best model
PredYBst=predict(BstModel,train)

#Calculate RMSE of the best model 
RMSEBst=RMSE(PredYBst,train$Score)

##Calculate parameters of the Best SVR model

#Find value of W
W = t(BstModel$coefs) %*% BstModel$SV

#Find value of b
b = BstModel$rho

## Plotting SVM Model and Tuned Model in same plot
plot(train, pch=16)
points(train$HelpfulnessDenominator, predYsvm, col = "blue", pch=3)
points(train$HelpfulnessDenominator, PredYBst, col = "red", pch=4)
points(train$HelpfulnessDenominator, predYsvm, col = "blue", pch=3, type="l")
points(train$HelpfulnessDenominator, PredYBst, col = "red", pch=4, type="l")



#kNN (k-Nearest Neighbour)
#----------------------------------------
library(caret)
library(e1071)
library(tidyverse)

knnModel <- train(Score ~ ., data = train, method = "knn", 
                trControl=train.control, preProcess = c("center","scale"))
knnModel

plot(knnModel)

# Print the best tuning parameter k that
# maximizes model accuracy
knnModel$bestTune

# Make predictions on the test data
knnPred <- knnModel %>% predict(test)
head(knnPred)


# Compute model accuracy rate
mean(knnPred == test$Score)

# Compute the prediction error RMSE. The lower the RMSE, the better the model.
RMSE(knnPred, test$Score)


# Compare algorithms
library(mlbench)
library(caret)
library(ggplot2)

results <- resamples(list(LR=linearMod, SVM=svmModel, knn=knn_model))
summary(results)

dotplot(results)
bwplot(results)

#statistical hypothesis tests
#Student's t-test in R



#Stochastic Gradient Boosting
#----------------------------
library(gbm)
library(ggplot2)
boostModel=gbm(Score ~ . ,data = train,distribution = "gaussian",n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)
summary(boostModel)
plot(boostModel,i="HelpfulnessDenominator") 



cor(train$Score,train$HelpfulnessDenominator)#negetive correlation coeff-r

#Prediction on Test Set
n.trees = seq(from=100 ,to=10000, by=100) #no of trees-a vector of 100 values 


#Generating a Prediction matrix for each Tree
predmatrix<-predict(boostModel,train,n.trees = n.trees)
dim(predmatrix)

#Calculating The Mean squared Test Error
test.error<-with(train,apply( (predmatrix-HelpfulnessDenominator)^2,2,mean))
#contains the Mean squared test error for each of the 100 trees averaged
head(test.error) 

plot(n.trees , test.error , pch=19,col="blue",xlab="Number of Trees",ylab="Test Error", main = "Perfomance of Boosting on Test Set")

#adding the RandomForests Minimum Error line trained on same data and similar parameters
abline(h = min(test.error),col="red") #test.err is the test error of a Random forest fitted on same data
legend("topright",c("Minimum Test error Line for Random Forests"),col="red",lty=1,lwd=1)


#Normality Test 
library(dplyr)
library(magrittr)
library(ggplot2)
library(ggpubr)

#Assess the normality of the data in R
library("ggpubr")
ggdensity(train$Score, 
          main = "Density plot of Score",
          xlab = "Score Value")
#Q-Q plot: Q-Q plot (or quantile-quantile plot) draws the correlation between a given sample and the 
#normal distribution. A 45-degree reference line is also plotted.
ggqqplot(train$Score)

library("car")
qqPlot(train$Score)


#Shapiro-Wilk's method is widely recommended for normality test and it provides better power than K-S. 
#It is based on the correlation between the data and the corresponding normal scores.

# conventional value should of .05 but out is < .00000000000000022.
#As you said that if the P-Value is less than 0.05, we reject the null hypothesis in case of 
#test with 95% confidence or 5% significance. In general, we reject the null hypothesis when the P-value 
#is less then the level of significance of the test.
shapiro.test(train$Score)


#Compute summary statistics by groups:
group_by(train, Score) %>%
  summarise(
    count = n(),
    mean = mean(HelpfulnessDenominator, na.rm = TRUE),
    sd = sd(HelpfulnessDenominator, na.rm = TRUE)
  )


ggboxplot(train, HelpfulnessDenominator = "Score", y = "HelpfulnessDenominator", 
          color = "Score", palette = c("#00AFBB", "#E7B800"),
          order = c("before", "after"),
          ylab = "Score", xlab = "Unhelpful")


# Subset before the score
before <- subset(train,  Score == "before", HelpfulnessDenominator,
                 drop = TRUE)

# subset after the score
after <- subset(train,  Score == "after", HelpfulnessDenominator,
                drop = TRUE)

library(PairedData)
pd <- paired(before, after)
plot(pd, type = "profile") + theme_bw()

# compute the difference
d <- with(train, HelpfulnessDenominator[Score == "before"] - HelpfulnessDenominator[Score == "after"])
#interger showing 0 so therefore the Shapiro-Wilk normality has failed

# Shapiro-Wilk normality test for the differences
shapiro.test(d) # => p-value = 0.6141



#--------------------------Non-Linear methods: Neural Network-------------------------
#Neural Network - part 1
#-----------------------
#install.packages("neuralnet")
#install.packages("MASS")
library(neuralnet)
library(MASS)

## Scale data for neural network
max = apply(ds , 2 , max)
min = apply(ds, 2 , min)
scaled = as.data.frame(scale(ds, center = min, scale = max - min))

train_ <- scaled[index,]
test_ <- scaled[-index,]


#rm(scaleddata, normalize, maxmindf)

  # fit neural network
set.seed(123)
#nnModel = neuralnet(Score ~ HelpfulnessDenominator, data = train_, hidden = 3 , linear.output = T )
#nnModel
#nnModel$result.matrix


n <- names(train_)
f <- as.formula(paste("Score ~", paste(n[!n %in% "Score"], collapse = " + ")))
nnModel <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
nnModel

# plot neural network
plot(nnModel)

## Prediction Score using neural network 
predict_testnn = neuralnet::compute(nnModel, test[,c(1:1)])
predict_testnn = (predict_testnn$net.result * (max(ds$Score) - min(ds$Score))) + min(ds$Score)

plot(test$Score, predict_testnn, col='blue', pch=16, ylab = "predicted rating NN", xlab = "real rating")

abline(0,1)

test.r <- (test_$Score)*(max(ds$Score)-min(ds$Score))+min(ds$Score)
MSE.nn <- sum((test.r - predict_testnn)^2)/nrow(test_)



# Calculate Root Mean Square Error (RMSE)
RMSE.NN = (sum((test$Score - predict_testnn)^2) / nrow(test)) ^ 0.5


plot(test$Score,predict_testnn,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

#Cross validation of neural network model - part 2
# Load libraries
library(boot)
library(plyr)

# Initialize variables
set.seed(50)
k = 100
RMSE.NN = NULL


List = list()

# Fit neural network model within nested for loop
for(j in 10:65){
  for (i in 1:k) {
    index = sample(1:nrow(ds),j )
    
    train = scaled[index,]
    test = scaled[-index,]
    datatest = ds[-index,]
    
    NN = neuralnet(Score ~ HelpfulnessDenominator, train, hidden = 3, linear.output= T)
    predict_testNN = neuralnet::compute(NN,test[,c(1:1)])
    predict_testNN = (predict_testNN$net.result*(max(ds$Score)-min(ds$Score)))+min(ds$Score)
    
    RMSE.NN [i]<- (sum((test$Score - predict_testNN)^2)/nrow(test))^0.5
  }
  List[[j]] = RMSE.NN
}

Matrix.RMSE = do.call(cbind, List)

## Prepare boxplot
boxplot(Matrix.RMSE[,56], ylab = "RMSE", main = "RMSE BoxPlot (length of traning set = 65)")

## Variation of median RMSE 
#install.packages("matrixStats")
library(matrixStats)

med = colMedians(Matrix.RMSE)

X = seq(10,65)

plot (med~X, type = "l", xlab = "length of training set", ylab = "median RMSE", main = "Variation of RMSE with length of training set")


#Naive Bayes
#Naive Bayes is suited for classification, i.e  1 or 0. Alternative would be logistic regression
#-----------
library(e1071)
library(naivebayes)
library(dplyr)
library(mlbench)
library(ggplot2)
library(caret)

set.seed(123)

#Fitting the Naive Bayes model
nbModel=naiveBayes(Score ~., 'nb',data=train)
#What does the model say? Print the model summary
summary(nbModel)


# Predicting the test set results
NBpred <- predict(nbModel, test)
NBpred


#------------------------------------------End of Non-Linear methods--------------------------------------------- 

#------------------------------------------------Linear methods:-------------------------------------------------
#Linear methods: Linear Discriminant Analysis and Ordinal logistic regression.

#Linear Discriminant Analysis
#----------------------------
library(MASS)
#install.packages("magrittr")
library(lattice)
library(ggplot2)
library(caret)
library(dplyr)
ldaModel <- lda(Score ~ ., data=train)
ldaModel

plot(ldaModel)

plot(ldaModel, col = as.integer(train$Score))
plot(ldaModel, dimen = 1, type = "b")

#Prediciton train ~
ldaModelPred <- predict(ldaModel, newdata = train[,c(1,1)])$class
ldaModelPred

# Make predictions for test ~
ldaModelPred <- ldaModel %>% predict(test)
ldaModelPred

# Model accuracy test1a
mean(ldaModelPred$class==test$Score)
sum(ldaModelPred$posterior[ ,1] >=.5)

#Inspect the results:
#Predicted classes
head(ldaModelPred$class, 6)
# Predicted probabilities of class memebership.
head(ldaModelPred$posterior, 6) 
# Linear discriminants
head(ldaModelPred$x, 3)

#Model accuracy:
mean(ldaModelPred$class==test$Score)


lda.data <- cbind(train, predict(ldaModel)$x)
ggplot(lda.data, aes(LD1)) +
  geom_point(aes(color = Score))


#Quadratic discriminant analysis - QDA
#-------------------------------------
QDAmodel <- qda(Score~., data = train)
QDAmodel

# Make predictions
QDApred <- QDAmodel %>% predict(test)

# Model accuracy
mean(QDApred$class == test$Score)

#Ordinal logistic regression
#---------------------------
# Load the library
library(MASS)
library(dplyr)
library(nnet)

olrModel <- multinom(Score ~ ., data=train)

summary(olrModel)

coefs <- coef(olrModel)

#confidence intervals 
# default method gives profiled CIs
ci <- confint(olrModel)

## CIs assuming normality
confint.default(olrModel)

#Convert coefficients into odds ratios
exp(coef(olrModel))

## OR and CI
#exp(cbind(OR = coef(olrModel), ci))

summary(olgModel)$standard.errors

# Calculate z-values
zvalues <- summary(olrModel)$coefficients / summary(olrModel)$standard.errors
# Show z-values
zvalues

#Calculating P value
pnorm(abs(zvalues), lower.tail=FALSE)*2

#Predict on test data
olrModelPred <- predict(olrModel, test)  # predict the classes directly
head(olrModelPred)

# predict the probabilites
predictedScores <- predict(olgModel, test, type="p")  
head(predictedScores)

#misclassification error
mean(as.character(test$Score) != as.character(olrModelPred))  
