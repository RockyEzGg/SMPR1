# install.packages("caret")
# install.packages("MASS")
# install.packages("klaR")
# require("caret")
require(lattice)
require(ggplot2)
require(klaR)
require(MASS)
require(e1071)

data("iris")
model <- NaiveBayes(Species ~ ., data = iris)
# class(model)
# summary(model)
# print(model)

preds <- predict(model, newdata = iris[,-5]) #type="raw"

conf_matrix <- table(preds$class, iris[,5])

plot(model)

####################################################################

# tbl_list <- sapply(iris[-5], table, iris[ , 5])
# tbl_list <- lapply(tbl_list, t)
# 
# cond_probs <- sapply(tbl_list, function(x) { 
#   apply(x, 1, function(x) { 
#     x / sum(x) }) })
# 
# cond_probs <- lapply(cond_probs, t)
# 
# print(cond_probs)

####################################################################

# y1 <- cond_probs$Petal.Width["setosa",]
# y2 <- cond_probs$Petal.Width["versicolor",]
# y3 <- cond_probs$Petal.Width["virginica",]
# x <- as.double(colnames(cond_probs$Petal.Width))
# 
# plot(x,y, type="n", main="Naive Bayes Plot", xlab="Petal.Width", ylab="Density")
# lines(x,y1,col="red",lwd=3)
# lines(x,y2,col="green3",lwd=3)
# lines(x,y3,col="blue",lwd=3)