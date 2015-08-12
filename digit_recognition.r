## The data is based on the MNIST dataset. Each digit is represented as a 28X28 pixel, 
# so there are 784 predictors. The dependent variable is the digit label (0-9). We
# use the caret package in R to fit different supervised machine learning models to 
# the data
train <- read.csv("train.csv", header=TRUE)
train<-as.matrix(train)

##Color ramp def.
colors<-c('white','black')
cus_col<-colorRampPalette(colors=colors)
 
## Plot the average image of each digit
par(mfrow=c(4,3),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
all_img<-array(dim=c(10,28*28))
for(di in 0:9)
{
print(di)
all_img[di+1,]<-apply(train[train[,1]==di,-1],2,sum)
all_img[di+1,]<-all_img[di+1,]/max(all_img[di+1,])*255
 
z<-array(all_img[di+1,],dim=c(28,28))
z<-z[,28:1] ##right side up
image(1:28,1:28,z,main=di,col=cus_col(256))
}

train<-as.data.frame(train)
## digits<-train[sample(nrow(train), 10000),]
## digits<-as.data.frame(digits)
library(caret)
badCols <- nearZeroVar(train)
train <- train[, -badCols]
train[c(-1)]<-sapply(train[c(-1)],scale)
test<-sapply(test,scale)

indexes=sample(1:nrow(train),size=0.3*nrow(train))
testdigits=train[indexes,]
traindigits=train[-indexes,]

## Fitting a single decision tree
library(rpart)
cart = rpart(label ~ ., data=traindigits, method="class", control = rpart.control(minbucket=10))
pfit<- prune(cart, cp=cart$cptable[which.min(cart$cptable[,"xerror"]),"CP"])
cartPredicttrain = predict(pfit, newdata=traindigits, type="class")
cartPredicttest = predict(pfit, newdata=testdigits, type="class")
cartTabletrain = table(traindigits$label, cartPredicttrain)
cartTabletest = table(testdigits$label, cartPredicttest)
sum(diag(cartTabletrain))/nrow(traindigits)
sum(diag(cartTabletest))/nrow(testdigits)

## Fitting a randomForest
library(randomForest)
traindigits$label = factor(traindigits$label)
testdigits$label = factor(testdigits$label)
tunedForest=tuneRF(traindigits[,-1],traindigits[,1],ntreeTry=50,stepfactor=2)
randomForest = randomForest(label ~ ., data=traindigits, nodesize=10, mtry=30, do.trace=TRUE)
randomForestPredict = predict(randomForest, newdata=testdigits)
head(randomForestPredict)
randomForestTable = table(testdigits$label, randomForestPredict)
randomForestTable
sum(diag(randomForestTable))/nrow(testdigits)

## Naive Bayes Digit Classifier
library(e1071)
traindigits$label = factor(traindigits$label)
testdigits$label = factor(testdigits$label)
NBclassifier<-naiveBayes(traindigits[,-1],traindigits$label)
NBtabletrain=table(predict(NBclassifier, traindigits[,-1]), traindigits[,1])
NBtabletest=table(predict(NBclassifier, testdigits[,-1]), testdigits[,1])
sum(diag(NBtabletrain))/nrow(traindigits)
sum(diag(NBtabletest))/nrow(testdigits)

## K Nearest Neighbour
# weighted k-nearest neighbors package
library(kknn)
# optimize knn for k=1:15
# and kernel=triangular, rectangular, or gaussian
model <- train.kknn(as.factor(label) ~ ., train, kmax=15, kernel=c("triangular","rectangular","gaussian"))
 
# print out best parameters and prediction error
print(paste("Best parameters:", "kernel =", model$best.parameters$kernel, ", k =", model$best.parameters$k))
print(model$MISCLASS)

# train the optimal kknn model
model <- kknn(as.factor(label) ~ ., traindigits, testdigits, k=6, kernel="triangular")
results <- model$fitted.values
##write(as.numeric(levels(results))[results], file="knn_submission.csv", ncolumns=1)
sum(results==testdigits$label)/nrow(testdigits)

## Fitting a boosted trees model
library(gbm)
library(caret)
gbm1 <- gbm(label ~., data=traindigits, n.trees=1000, shrinkage=0.05, interaction.depth=3, bag.fraction = .5, train.fraction = 1, cv.folds = 10, keep.data=TRUE, verbose=FALSE)

gbmGrid <-  expand.grid(interaction.depth = 4,
                        n.trees = (1:20)*50,
                        shrinkage = 0.1)

fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE)
gbmFit <- train(as.factor(label) ~ ., data=traindigits,
                method="gbm",
                trControl=fitControl,
		    tuneGrid=gbmGrid,
                verbose=FALSE)
gbmFit
predict_train<-predict(gbmFit,newdata=traindigits)
sum(predict_train==traindigits$label)/nrow(traindigits)
predict_test<-predict(gbmFit,newdata=testdigits)
sum(predict_test==testdigits$label)/nrow(testdigits)

Accuracy of 1 on training dataset and 0.9643651 on test dataset

## Fitting SVM model
require(caret)
require(kernlab)
fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE)
svmFit <- train(label ~ ., data = traindigits,
                 method = "svmRadial",
                 trControl = fitControl,
                 tuneLength = 8)
predict_train<-predict(svmFit,newdata=traindigits)
sum(predict_train==traindigits$label)/nrow(traindigits)
predict_test<-predict(svmFit,newdata=testdigits)
sum(predict_test==testdigits$label)/nrow(testdigits)

Parameters sigma=0.0022 C=8
classification accuracy of 0.9992857 on training set
classification accuracy of 0.9765079 on test set

## Fitting a NN model
require(caret)
require(nnet)
fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE)

nnetFit<- train(label~., data=traindigits, method="nnet", 
		    trControl=fitControl, 
		    tuneGrid=expand.grid(.size=c(1,5,10,20,50,100,200,500),.decay=c(0,0.001,0.005,0.01,0.05,0.1,0.5,1,2)))
predict_train<-predict(nnetFit,newdata=traindigits)
sum(predict_train==traindigits$label)/nrow(traindigits)
predict_test<-predict(nnetFit,newdata=testdigits)
sum(predict_test==testdigits$label)/nrow(testdigits)

## Fitting a RDA model
require(caret)
fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE)
svmFit <- train(label ~ ., data = traindigits,
                 method = "rda",
                 trControl = fitControl,
                 tuneLength = 8)
