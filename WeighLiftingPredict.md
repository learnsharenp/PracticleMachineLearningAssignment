# Practical Machine Learning Assignment
RamKGubhaju  
May 17, 2016  

## Introduction
The dataset of the Weight Lifting Exercises provided from which predictive model should be devise which can predict the kind of weighlifting from the provided data.

## Data Preparation

Load the training and testing data set from the internet if not found locally.


```r
if(!file.exists("pml-training.csv")){
        
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-training.csv")
    }
    
PmlTrain <- read.csv("pml-training.csv")

if(!file.exists("pml-testing.csv")){
        
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")
    }
PmlTest <- read.csv("pml-testing.csv")

library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

The scrub of the training(PmlTrain) has been carried out by two criteria. First is to remove variable that variance is nearly zero, and second is to remove variable that has many NA values.


```r
# remove variables which has nearly zero variance
NearZeroVariance <- nearZeroVar(PmlTrain)
PmlTrain <- PmlTrain[, -NearZeroVariance]


# remove variables Mean of number of NA is .9
MaxNA <- sapply(PmlTrain, function(x) mean(is.na(x))) > 0.9
PmlTrain <- PmlTrain[, MaxNA==F]

# The first five variables (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp) does not make sense in predictive model.
PmlTrain <- PmlTrain[, -(1:5)]
```

Now split the training set into two group in the ration of 70/30 where 70% is used for training and 30% for testing. Setting the seed is must for the reproducibility.


```r
set.seed(12345)
TrainingRatio <- createDataPartition(y=PmlTrain$classe, p=0.7, list=F)
PmlTrain_Train <- PmlTrain[TrainingRatio, ]
PmlTrain_Test <- PmlTrain[-TrainingRatio, ]
```

## Model Building
Use the Random Forest model for training and predict the test set.the use of  2-fold cross-validation is selected for training function for optimal tuning parameters for the model.


```r
# instruct train to use 2-fold CV to select optimal tuning parameters
fitControl <- trainControl(method="cv", number=2, verboseIter=F)

# fit model on ptrain1
fit <- train(classe ~ ., data=PmlTrain_Train, method="rf", trControl=fitControl)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
# print final model to see tuning parameters it chose
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    5 2650    2    1    0 0.0030097818
## C    0    5 2391    0    0 0.0020868114
## D    0    0    7 2245    0 0.0031083481
## E    0    0    0    5 2520 0.0019801980
```

The optimum configuration is that 500 trees and to try 27 variables at each split.

## Model Evaluation and Selection

Now, Predict the  ("classe") in PmlTrain_Test, and evaluate the confusion matrix to compare the predicted versus the actual labels:


```r
# use model to predict classe in validation set (ptrain2)
preds <- predict(fit, newdata=PmlTrain_Test)

# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(PmlTrain_Test$classe, preds)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    5 1133    1    0    0
##          C    0    4 1022    0    0
##          D    0    0    8  956    0
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9943, 0.9977)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9953          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9965   0.9913   0.9958   1.0000
## Specificity            1.0000   0.9987   0.9992   0.9984   0.9992
## Pos Pred Value         1.0000   0.9947   0.9961   0.9917   0.9963
## Neg Pred Value         0.9988   0.9992   0.9981   0.9992   1.0000
## Prevalence             0.2853   0.1932   0.1752   0.1631   0.1832
## Detection Rate         0.2845   0.1925   0.1737   0.1624   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9985   0.9976   0.9952   0.9971   0.9996
```

The predicted accuracy for the out-of-sample error is 0.4% which shows that the model is good enough for the prediction of the type of weigh lifting from the test set.

## Train full traininig dataset using Random Forest model 

For the actual predictive model, train the Random Forest model in full training set.



