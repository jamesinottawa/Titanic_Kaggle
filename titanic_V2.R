#This script loads the Kaggle Titanic dataset and conducts
#a machine learning exercise on it and writes a Kaggle submission
#
# Author : James T.E. Chapman
# Creation Date : 19 Feb 2017

#library(tidyverse)
library(dplyr)
library(readr)
library(stringr)
library(caret)
library(caretEnsemble)
library(doParallel)

registerDoParallel(cores=4)

## The usual
rm(list = ls())

# These are the constants that run the analysis

rdn.seed <- 8101026
train.file <- "./train.csv"
test.file  <- "./test.csv"
base.submission.file <- "titanic_submission.csv"

#A list of titles from training and test sets
titles <- c(
  "Mr",
  "Mrs",
  "Miss",
  "Master",
  "Don",
  "Rev",
  "Dr",
  "Mme",
  "Ms",
  "Major",
  "Lady",
  "Sir",
  "Mlle",
  "Col",
  "Capt",
  "the",
  "Jonkheer",
  "Dona",
  "RM",
  "RF"
)

# Rare titles and royal titles for consolidation
rare.title.male <- c("Capt",
                     "Col",
                     "Don",
                     "Dr",
                     "Jonkheer",
                     "Major",
                     "Rev",
                     "Sir",
                     "RF",
                     "RM")
rare.title.female.miss <- c("Mme", "Ms")
rare.title.female.mrs <- c("Lady", "Mlle", "the", "Dona")
royal.titles <- titles[c(5, 11, 12, 16, 17, 18)]

#Variables to focus on in the analysis
analysis.vars <- c(
  "test",
  "Pclass",
  "Age",
  "Sex",
  "Parch",
  "SibSp",
  "Fare",
  "Embarked",
  "title",
  "Survived",
  "royal"
)


# Now load and setup the data
set.seed(rdn.seed)
rm(rdn.seed)
train <- read_csv(train.file)
rm(train.file)
test <- read_csv(test.file)
rm(test.file)


#consolidate both data sets to perform some substitutions
train$test <- FALSE
test$test <- TRUE
test$Survived <- -1

full.data <- rbind(train, test)
rm(train, test)

#Now create the submission file
pass.id <- full.data %>% filter(test) %>% select(PassengerId)
submission <- data.frame(PassengerId = pass.id)

#data cleaning and preprocessing

#split the name column
tmp <- str_split(full.data$Name, ",", simplify = TRUE)
#extract last names
full.data$lastName <- tmp[, 1]
#extract titles
full.data$title <-  str_split(tmp[, 2], boundary("word"), simplify = TRUE)[, 1]
rm(tmp)
#make a marker for royal (i.e. upper class) titles
full.data <- full.data %>% mutate(royal = title %in% royal.titles)

#clear up the rare titles and turn them into one of Miss,Mrs, Mr, Master
full.data$title[full.data$title %in% rare.title.female.miss] <-
  "Miss"
full.data$title[full.data$title %in% rare.title.female.mrs] <- "Mrs"
full.data$title[full.data$title %in% rare.title.male] <- "Mr"

cleaned.data <- full.data[, analysis.vars]
cleaned.data <- cleaned.data %>%
  mutate(Pclass = factor(Pclass, ordered = TRUE)) %>%
  mutate(Sex = factor(Sex)) %>%
  mutate(Embarked = factor(Embarked)) %>%
  mutate(title = factor(title)) %>%
  mutate(family = SibSp + Parch+1) %>%
  mutate(family = cut(
    family,
    breaks = c(1, 2, 5, 11),
    include.lowest = TRUE,
    right = FALSE
  ))

#Repartition the data for analysis
x.train <- cleaned.data %>% filter(!test) %>% select(-test) %>%
  mutate(Survived = factor(Survived))
#Note need to fix this imputation to something better
x.train$Embarked[is.na(x.train$Embarked)] <- "S"
levels(x.train$Survived) <- c("dead", "alive")
x.test <- cleaned.data %>% filter(test) %>% select(-Survived, -test)

#Now do some preProcessing
pre.x.train <- preProcess(x.train, method = c("bagImpute"))
x.train <- predict(pre.x.train, x.train)
pre.x.test <-  preProcess(x.test, method = c("bagImpute"))
x.test <- predict(pre.x.test, x.test)

rf.x.train <- x.train %>% select(-royal)
#Now do the analysis
rf.model <- train(
  Survived ~ .,
  data = rf.x.train,
  method = "cforest",
  trControl = trainControl(
    method = "cv",
    number = 100,
    savePredictions = "final",
    classProbs = TRUE
  )
)

#make an RF submission file
submission$Survived <- predict(rf.model, x.test)
levels(submission$Survived) <- c("0", "1")
#create a submission name
submission.file <- paste0("rf_", base.submission.file)
write.csv(submission, file = submission.file, row.names = FALSE)

svm.model <- train(Survived~., data=rf.x.train,method="svmRadial",
                   preProcess=c("center","scale"),
                   trControl=trainControl(
                     method="LOOCV",
                     savePredictions = "final",
                     classProbs = TRUE
                   ))
submission$Survived <- predict(svm.model, x.test)
levels(submission$Survived) <- c("0", "1")
#create a submission name
submission.file <- paste0("svm_", base.submission.file)
write.csv(submission, file = submission.file, row.names = FALSE)


gbm.model <- train(
  Survived ~ .,
  data = x.train,
  method = "gbm",
  trControl = trainControl(
    method = "cv",
    number = 100,
    savePredictions = "final",
    classProbs = TRUE)
)
#and a GBM submission file
submission$Survived <- predict(gbm.model, x.test)
levels(submission$Survived) <- c("0", "1")
#create a submission name
submission.file <- paste0("gbm_", base.submission.file)
write.csv(submission, file = submission.file, row.names = FALSE)
