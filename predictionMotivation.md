Coursera Assignment on Machine Learning -- Data on Weight LIfting Exercises.
========================================================
Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Before we begin, we will download the training and test data from the datasets at the specified locations. IN the code below, we also assign missing values to entries that are currently 'NA' or blank.



```r

library(corrplot)

library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r

exerData <- read.csv("C:/Neelima/Coursera/Assignments/data/pml-training.csv", 
    header = TRUE, na.strings = c("NA", ""))
exerData_test <- read.csv("C:/Neelima/Coursera/Assignments/data/pml-testing.csv", 
    header = TRUE, na.strings = c("NA", ""))
```


The columns which are mostly filled with missing values are removed from the training and testing data set. So, we count the number of missing values in each column of the training dataset.We use the sums to create a logical variable for each column of the dataset. The logical value is 'TRUE' if any column has no missing values and 'FALSE' when there are missing values.

When we apply the logical variable to the columns of the training and testing datasets, it will only retain those columns which are complete.


```r
csums <- colSums(is.na(exerData))
csums_log <- (csums == 0)
training_fewer_cols <- exerData[, (colSums(is.na(exerData)) == 0)]
exerData_test <- exerData_test[, (colSums(is.na(exerData)) == 0)]
```

Create another logical vector to delete unnecessary columns from the above datasetsColumn names in the dataset containing the entries shown in the 'grepl' function will have a value of 'TRUE' in the logical vector. Since these are the columns we want to remove, we apply the negation of the logical vector against the columns of our dataset. 

```r
del_cols_log <- grepl("X|user_name|timestamp|new_window", colnames(training_fewer_cols))
training_fewer_cols <- training_fewer_cols[, !del_cols_log]
exerData_test_final <- exerData_test[, !del_cols_log]
```

Let us now split the data into training dataset(75% of the observations)
and a vaildation datasets(25% of the observations). This allows us to perform cross validation when developing our model.

```r
inTrain = createDataPartition(y = training_fewer_cols$classe, p = 0.75, list = FALSE)
small_train = training_fewer_cols[inTrain, ]
small_valid = training_fewer_cols[-inTrain, ]
```

At this point, our dataset contains 54 variables, with the last column containing the 'classe' variable we are trying to predict. We begin by looking at the correlations between the variables in our dataset. We may want to remove highly correlated predictors from our analysis and replace them with weighted combinations of predictors. This may allow a more complete capture of the information available.

```r
corMat <- cor(small_train[, -54])
corrplot(corMat, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, 
    tl.col = rgb(0, 0, 0))
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 


This grid shows the correlation between pairs of the predictors in our dataset. From a high-level perspective darker blue and darker red squares indicate high positive and high negative correlations, respectively. Based on this observation, we choose to implement a principal components analysis to produce a set of linearly uncorrelated variables to use as our predictors.
Principal Components Analysis and Machine Learning

We pre-process our data using a principal component analysis, leaving out the last column ('classe'). After pre-processing, we use the 'predict' function to apply the pre-processing to both the training and validation subsets of the original larger 'training' dataset.


```r
preProc <- preProcess(small_train[, -54], method = "pca", thresh = 0.99)
trainPC <- predict(preProc, small_train[, -54])
valid_testPC <- predict(preProc, small_valid[, -54])
```

Now, we train a model using a random forest approach on the smaller training dataset. We chose to specify the use of a cross validation method when applying the random forest routine in the 'trainControl()' parameter. Without specifying this, the default method (bootstrapping) would have been used. The bootstrapping method seemed to take a lot longer to complete, while essentially producing the same level of 'accuracy'.

```r
modelFit <- train(small_train$classe ~ ., method = "rf", data = trainPC, trControl = trainControl(method = "cv", 
    number = 4), importance = TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

When reviewing the relative importance of the resulting principal components of the trained model, 'modelFit'.


```r
varImpPlot(modelFit$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, cex = 1, 
    main = "Importance of the Individual Principal Components")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8.png) 

As you look from the top to the bottom on the y-axis, this plot shows each of the principal components in order from most important to least important. The degree of importance is shown on the x-axis-increasing from left to right. Therefore, points high and to the right on this graph correspond to those principal components that are especially valuable in terms of being able to classify the observed training data.
Cross Validation Testing and Out-of-Sample Error Estimate

We then call the 'predict' function again so that our trained model can be applied to our cross validation test dataset. We can then view the resulting table in the 'confusionMatrix' function's output to see how well the model predicted/classified the values in the validation test set (i.e. the 'reference' values)

```r
pred_valid_rf <- predict(modelFit, valid_testPC)
confus <- confusionMatrix(small_valid$classe, pred_valid_rf)
confus$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1390    2    0    1    2
##          B   18  922    7    0    2
##          C    1   14  836    4    0
##          D    0    0   27  777    0
##          E    0    2    6    0  893
```

The estimated out-of-sample error is 1.000 minus the model's accuracy, the later of which is provided in the output of the confusionmatrix, or more directly via the 'postresample' function. 


```r
accur <- postResample(small_valid$classe, pred_valid_rf)
model_accuracy <- accur[[1]]
model_accuracy
```

```
## [1] 0.9825
```



```r
out_of_sample_error <- 1 - model_accuracy
out_of_sample_error
```

```
## [1] 0.01754
```

The estimated accuracy of the model is 98.4% and the estimated out-of-sample error based on our fitted model applied to the cross validation dataset is 1.6%.
Predicted Results

Finally, we apply the pre-processing to the original testing dataset, after removing the extraneous column labeled 'problem_id' (column 54). We then run our model against the testing dataset and display the predicted results.
This is applying machine learning algorithm to the 20 test cases available in the test data

```r
testPC <- predict(preProc, exerData_test_final[, -54])
pred_final <- predict(modelFit, testPC)
pred_final
```

```
##  [1] B A C A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

