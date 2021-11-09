library(data.table)
library(tidyverse)
library(inspectdf)
library(text2vec)
library(caTools)
library(glmnet)
library(stringr)

#Importing dataset
df<-fread("emails.csv")

#Data understanding----
df %>% dim()

df %>% colnames()

df %>% glimpse()

df %>% inspect_na()

for (i in 1:nrow(df))
{
  df$id[i]=i;
}
# Data preprocessing 
df$id<-df$id %>% as.character()   


df<-df[!duplicated(df$text), ]
df %>% dim()

df<-select(df,id,  everything());

for (i in 1:nrow(df))
{
  string<-df$text[i]
  temp <- tolower(string)    #lowercase all words 
  temp <- stringr::str_replace_all(temp,"[^a-zA-Z\\s]", " ")   #remove all nonletter characters 
  temp <- stringr::str_replace_all(temp,"[\\s]+", " ")        #remove extra spaces, just one space 
  temp <- stringr::str_replace_all(temp,"subject ", "")       #remove all "subject" words 
  # to get  rid of trailing "" if necessary
  indexes <- which(temp == "")
  if(length(indexes) > 0){
    temp <- temp[-indexes]
  } 
  temp<- gsub("[^\x01-\x7F]", "", temp) #to get rid of all non-ASCII characters
  df$text[i]<-temp
}

# Splitting data into train and test set----
set.seed(123)
split <- df$spam %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

#Tokenizing----
train %>% colnames()

it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$id,
         progressbar = F) 

#Creating vocabulary---- 
vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>%   
  head(100) %>% 
  tail(10) 

#Vectorizing 
vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()

identical(rownames(dtm_train), train$id)

# Modeling: Normal Nfold GLm ----
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

#AUC score for train set 
glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]

#AUC score for test set 
glmnet:::auc(test$spam, preds) %>% round(2)
#No significant difference between the AUC score of train and test,therefore no overfitting problem


#Creating new vocabulary----
# Pruning some words with defining stopwords while creating vocabulary for removing them----
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 

#Creating  DTM for Training and Testing with new pruned vocabulary----
vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

#AUC for training
glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]

#AUC score for test 
glmnet:::auc(test$spam, preds) %>% round(2)
#No significant difference between the AUC score of train and test,therefore no overfitting problem


