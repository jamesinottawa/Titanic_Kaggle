#This script loads the Kaggle Titanic dataset and conducts
#a machine learning exercise on it and writes a Kaggle submission
# 
# Author : James T.E. Chapman
# Creation Date : 19 Feb 2017


library(tidyverse)
library(stringr)
library(caret)
library(caretEnsemble)

## The usual
rm(list=ls())

# These are the constants that run the analysis

rdn.seed <- 8101026
train.file <- "./train.csv"
test.file  <- "./test.csv"
base.submission.file <- "titanic_submission.csv"

#A list of titles from training and test sets
titles <-c( "Mr", "Mrs","Miss","Master",
            "Don","Rev","Dr","Mme",
            "Ms","Major","Lady",
            "Sir","Mlle",
            "Col","Capt",
            "the","Jonkheer",
            "Dona","RM","RF")

# Rare titles and royal titles for consolidation
rare.title.male <- c("Capt","Col","Don","Dr","Jonkheer",
                     "Major","Rev","Sir","RF","RM")
rare.title.female.miss <- c("Mme","Ms")
rare.title.female.mrs <- c("Lady","Mlle","the","Dona")
royal.titles <- titles[c(5,11,12,16,17,18)]

# Now load and setup the data
set.seed(rdn.seed)
rm(rdn.seed)
train <- read_csv(train.file)
rm(train.file)
test <- read_csv(test.file)
rm(test.file)


#consolidate both data sets to perform some substitutions
train$test <- FALSE
test$test <-TRUE
test$Survived <- -1

full.data <- rbind(train,test)
rm(train,test)

#Now create the submission file
pass.id <- full.data %>% filter(test) %>% select(PassengerId)
submission <- data.frame(PassengerId = pass.id)

#data cleaning and preprocessing

#split the name column
tmp <- str_split(full.data$Name,",",simplify = TRUE)
#extract last names
full.data$lastName <- tmp[,1]
#extract titles
full.data$title <- str_split(tmp[,2],boundary("word"),simplify = TRUE)[,1]
rm(tmp)
#make a marker for royal (i.e. upper class) titles
full.data <- full.data %>% mutate(royal = title %in% royal.titles)

#clear up the rare titles and turn them into one of Miss,Mrs, Mr, Master
full.data$title[full.data$title %in% rare.title.female.miss] <- "Miss"
full.data$title[full.data$title %in% rare.title.female.mrs] <- "Mrs"
full.data$title[full.data$title %in% rare.title.male] <- "Mr"


#Now conduct a feature engineering and preprocessing of the data
#as a function to run on training and testing data seperatly 
#
extractFeatures <- function(data) {
  #variables to extract
  features <- c("Pclass",
                "Age",
                "Sex",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked",
                "title")
  fea <- data[,features]
  
  #create factors out of catagorical variables
  fea$Pclass <- factor(fea$Pclass,ordered = TRUE)
  fea$Sex      <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  fea$title <- factor(fea$title,levels=titles)
  fea$title <- droplevels(fea$title)

  #make a family size variable
  fea$family <- fea$SibSp+fea$Parch
  fea$family <- cut(fea$family,breaks=c(0,1,2,3,4,10),include.lowest = TRUE,right=FALSE)

  fea$SibSp <- fea$Parch <- fea$Embarked  <- NULL
  return(fea)
}

x.train <- full.data %>% filter(!test) %>% extractFeatures()
x.train$Survived <- as.factor(full.data$Survived[!full.data$test])
levels(x.train$Survived) <- c("dead","alive")
x.test <- full.data %>% filter(test) %>% select(-Survived) %>% extractFeatures()

#Now do some preProcessing
pre.x.train <- preProcess(x.train,method=c("bagImpute"))
x.train <- predict(pre.x.train,x.train)
pre.x.test <-  preProcess(x.test,method=c("bagImpute"))
x.test <- predict(pre.x.test,x.test)

#Now do the analysis
rf.model <- train(Survived~.,
                  data=x.train,
                  method="rf",
                  trControl=trainControl(
                    savePredictions="final",
                    classProbs=TRUE
                  ))
#make an RF submission file
submission$Survived <- predict(rf.model,x.test)
levels(submission$Survived) <- c("0","1")
#create a submission name
submission.file <- paste0("rf",base.submission.file)
write.csv(submission, file = submission.file, row.names=FALSE)

gbm.model <- train(Survived~.,
                   data=x.train,
                   method="gbm",
                   trControl=trainControl(
                     savePredictions="final",
                     classProbs=TRUE
                   ))
#and a GBM submission file
submission$Survived <- predict(gbm.model,x.test)
levels(submission$Survived) <- c("0","1")
#create a submission name
submission.file <- paste0("gbm",base.submission.file)
write.csv(submission, file = submission.file, row.names=FALSE)




