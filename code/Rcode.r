# Importing the required libraries
library(dplyr)
library(caret)
library(gains)
library(pROC)
library(ROCR)
library(ROSE)
library(e1071)
#data_raw<-read.csv("card_spent.csv")

#for(i in 1:nrow(data_raw)){
  #if ((data_raw[i,2])>=0 & (data_raw[i,2]<200)){
    #data_raw[i,"Customer_cat"]<-"low_spend_cust"
  #else if((data_raw[i,2])>=200 & (data_raw[i,2]<1000)){
    #data_raw[i,"Customer_cat"]<-"medium_spend_cust"
#}else{
  #data_raw[i,"Customer_cat"]<-"high_spend_cust"
#}
    #}

#data<-data_raw[c(-2)]

#write.csv(data, file = "card_data.csv",row.names=FALSE)

#Importing the dataset

data_raw<-read.csv("card_data.csv")

data<-data_raw[-1]#Variable customer id not required further
summary(data)
str(data)

#changing the multi label categorical variables to factors from default numeric variable
data$townsize<-as.factor(paste(data$townsize))
data$jobcat<-as.factor(paste(data$jobcat))
data$retire<-as.factor(paste(data$retire))
data$hometype<-as.factor(paste(data$hometype))
data$addresscat<-as.factor(paste(data$addresscat))
data$cartype<-as.factor(paste(data$cartype))
data$carvalue<-as.factor(paste(data$carvalue))
data$carbought<-as.factor(paste(data$carbought))
data$card2<-as.factor(paste(data$card2))
data$card2type<-as.factor(paste(data$card2type))
data$card2benefit<-as.factor(paste(data$card2benefit))
data$bfast<-as.factor(paste(data$bfast))
data$internet<-as.factor(paste(data$internet))
data$Customer_cat<-as.factor(paste(data$Customer_cat))

#names(data)
#summary(data)

# finiding out the factor variable column numbers, so to exclude them from mice imputation and other imputation processes as required

fac_col<-as.integer(0)
facnames<-names(Filter(is.factor, data))
k=1
for(i in facnames){
while (k<=15){
fac_col[k]<-grep(i, colnames(data))
k=k+1
break
}
}

#imputing the missing integer values using pmm method

library(mice)
mice_impute<- mice((data[,-c(fac_col)]),m=1,maxit=50,meth='pmm',seed=500)
total_data<-complete(mice_impute,1)
total_data_final<-cbind(total_data,data[,c(fac_col)])#joing back the categorical variables
dim(total_data_final)
summary(total_data_final)
str(total_data_final)
total_data_final$Customer_cat<-as.factor(total_data_final$Customer_cat)


#Dividing the dataset randomly into 80 % train and 20 % test data #
set.seed(100)
train_Index<-sample(x=nrow(total_data_final),size=0.8*nrow(total_data_final),replace=FALSE)

train<-total_data_final[train_Index,]
test<-total_data_final[-train_Index,]

prop.table(table(train$Customer_cat))*100
# We can see that the data is highly imbalance 2 % high_spend_cust,30 % low_spend_cust while 68 % majority is medium_spent_cust
# let's do oversampling using SMOTE
set.seed(301)
library(DMwR)
train_aftersmote<-SMOTE(Customer_cat~ .,train,perc.over=3000,perc.under=200)
prop.table(table(Data_smote$Customer_cat))*100

#..............................with smoted train data.................#
 # fitting the base toolbox model in the smoted data
model_base_withsmote<-svm(Customer_cat~.,data=train_aftersmote)
summary(model_base_withsmote)

# Scoring the test data using the base_model
pred_ts_bs_smote<-predict(model_base_withsmote,test)
confusionMatrix(pred_ts_bs_smote,test$Customer_cat )
#..............................with original train data(train data before applying smote).................#

# fitting the base toolbox model in the smoted data
model_base_original<-svm(Customer_cat~.,data=train)

# Scoring the test data using the base_model
pred_ts_bs_original<-predict(model_base,test)
confusionMatrix(pred_ts_bs_original,test$Customer_cat)

# Thus we can see that the model trained on smoted test data performs considerably  well compared to
#unsmoted trained data  both in terms of sensitivy of the minority classes (and also bit better in term of overall accuracy)
#Thus we we go ahead tuning the model while fitting it to the smoted train dataset and not on the unsmoted train dataset

#............................................linear kernel.........................................#

set.seed(182)

#Let's do some gridsearch for parameter tuning of linear function, but here We won't be keeping the cost more than 2, in order so that the outliers may not effect my decision boundary creation extensively and  thus lead to overfitting

tuned_model1<-tune(svm,Customer_cat~.,data=train_aftersmote,kernel='linear',ranges=list(cost=c(0.05,0.1,0.5,1,2))) 


summary(tuned_model1)#linear kernal
tuned_model1$best.parameters #cost 2
the_lin_model<-tuned_model1$best.model # the model trained on the complete training data using the best parameter combination

pred_ts_t1<-predict(the_lin_model,test)
confusionMatrix(pred_ts_t1,test$Customer_cat)#linear kernal

#It's giving a much better performance  in terms of sensitivity in predicting minority classes 


#............................................radial kernel with tuning the gamma and cost parameters.........................................#

#Let's do gridsearch for parameter tuning of radial functions, but here We won't be keeping the cost more than 2, in order so that the outliers may not effect my decision boundary creation extensively and thus lead to overfitting
#Similarly we won't be  keeping too low lower than 0.001 as it would to lead to over fitting

set.seed(182)
tuned_model2<-tune(svm,Customer_cat~.,data=train_aftersmote,kernel='radial',ranges=list(cost=c(0.01,0.05,0.1,0.5,1,2),gamma=c(0.0001,0.001,0.01,.05,0.1,0.01,1,2)))
summary(tuned_model2)#radial kernal(GaussianKernel) 
tuned_model1$best.parameters# cost=2,gamma=0.01
the_rad_model<-tuned_model2$best.model # the model trained on the complete training data using the best parameter combination

pred_ts_t2<-predict(the_rad_model,test)#radial(gaussian kernal)
confusionMatrix(pred_ts_t2,test$Customer_cat)
# So we can see that the tuned model with radial(Gaussian ) kernal is giving a poor sensitivity in predicting high_spend customers

?svm()
#........................polynominal kernal.....................#

#Let's do gridsearch for parameter tuning of polynomial functions, but here We won't be keeping the cost more than 2, in order so that the outliers may not effect my decision boundary creation extensively and  thus will lead to over fitting
#Similarly we won't be  keeping polynomial higher order degree more than 4 as it would lead to overfitting

set.seed(182)
tuned_model3<-tune(svm,Customer_cat.,data=train_aftersmote,kernel='polynomial',ranges=list(cost=c(1,2),degree=c(2,3,4)))

summary(tuned_model3)
tuned_model3$best.parameters#Cost=2,degree=2
the_pol_model<-tuned_model3$best.model 
pred_tr_t3<-predict(the_pol_model,train)#polynomial kernal
confusionMatrix(pred_tr_t3,train$Customer_cat) 

# So we can see that the tuned model with polynomial kernal is giving a poor sensitivity in predicting high_spend customers for the test set

#.............Thus the final proposed method is built on hyperparameters which are a linear kernal with a Cost of 2 .....................#
