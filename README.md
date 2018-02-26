# Similarity-based classifiers. Nearest neighbor classifiers

> **Similarity-based classifiers** estimate the class label of a test sample based on the similarities between
the test sample and a set of labeled training samples, and the pairwise similarities between the
training samples. 

Similarity functions `ρ(x;y)`
may be asymmetric and fail to satisfy the other mathematical properties required for metrics or inner
products.
A popular approach to similarity-based classification is to treat the given dissimilarities as distances in some Euclidean space.

> **Nearest neighbor classifiers**
take training sample, object *u* that needs to be classified and optionally some parameters for weight funcion as input and output label (class) predicted for *u*. Algorithm sorts trainig sample by distance (similarity) to classified object in ascending order (so objects from trainig set with less *i* are closer to *u*). Object label is found as an argument that maximizes the sum of weight functions:

![nn](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2015-30-24.png)

As calculations are delayed until *u* is known nearest neighbor classifier is considered a *lazy learning* method. Varying the weight function different similarity-based classifiers can be obtained.

Some of them are observed in this research:

 1. k nearest neighbor algorithm
   - k weighted nearest neighbor algorithm
 2. Parzen window algorithm
   - Parzen window algorithm with variable window width
 3. Potential funcion algorithm

Algorithms are tested on standard **R** *iris* data set (Fisher's Iris data set). Plots are presented in petal.length, petal.width coordinate space.

Data compression methods are addressed by the example of STOLP algorithm.
At the end presented algorithms are compared in terms of generalization performance.

## k nearest neighbor algorithm
Parameter k is introduced and weight function looks like this:
`ω(i, u) = [i <= k]`. It means it returns 1 if *i* is less or equal to *k* and 0 otherwise.
In other words *u* gets the label of majority among its k nearest neigbors:

![kNN](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2012-21-22.png?raw=true)

Listing of `kNNClassifier` funciton written in R:

```R
# nearest neighbor classifers main params:
# trainSet - data frame with rows representing training objects
# each column represents some numerical feature, last column is label factor
# u - object to classify
# returns c(label, margin)
kNNClassifier <- function(trainSet, u, metric = dst1, k){
  rowsNum <- dim(trainSet)[1]; varsNum <- dim(trainSet)[2]-1
  labels = levels(trainSet[rowsNum, varsNum+1])
  orderedDist <- sortByDist(trainSet, u, rowsNum, varsNum, metric)
  kNNeighbors <- orderedDist[1:k,1]
  lst <- orderedDist[k,2]
  countTable <- table(kNNeighbors)
  maxLabelIdx = which.max(countTable)
  return(c(labels[maxLabelIdx], abs(max(countTable)-max(countTable[-maxLabelIdx]))))
}
```


Parameter ***k*** is found for particular test sample through empirical risk minimization by using **LOO** cross-validation. 
> **LOO** - (leave one out) is a procedure of empirical evaluation of classification algorithm quality. Sample is being divided into a test set consisting of a single element and a training set comprising all other elements for every element in this sample. It outputs the sum of errors (wrong classification cases) divided by total number of tests (number of elements in the sample).

Here is **R** implementation:
```R
LOO <- function(classifier, classifierArgs, dataSet, paramName, paramRange){
  rows <- dim(dataSet)[1]
  cols <- dim(dataSet)[2]-1
  risks <- vector('numeric', length(paramRange))
  minErr <- rows
  bestParam = 0; j=1; paramIdx = 1
  for(param in paramRange){
    err = 0
    for(i in 1:rows){
      currentArgs = c(list(dataSet[-i, ], dataSet[i, 1:cols]),classifierArgs)
      currentArgs[paramName] = param
      class <- do.call(classifier, currentArgs)[1]
      if(class != dataSet[i,cols+1])
        err <- err+1
    }
    if(err < minErr){
      minErr = err
      bestParam = param
      paramIdx = j
    }
    risks[j] = err/rows
    j = j+1
    message(paste("current ", paramName, " = ", sep=''), param, " out of ", paramRange[length(paramRange)])
  }
  message(paste("best ", paramName, ": ", sep=''), bestParam)
  message("errors made: ", minErr)
  message("empirical risk: ", minErr/rows)
  plot(paramRange, risks, xlab=paramName, ylab='LOO', col='red', type='l')
  points(bestParam, risks[paramIdx], bg='red', col='red', pch=21)
  text(bestParam, risks[paramIdx], paste(paramName,"=",bestParam), pos=3)
  return(bestParam)
}
```

Here is optimal *k* for iris data set (only sepal length and width considered) found by LOO:

![LOOkNN2](https://github.com/toxazol/machineLearning/blob/master/img/LOOkNN2noMod.png?raw=true)

Classification of arbitrary objects map (k=6):

![kNNk6](https://github.com/toxazol/machineLearning/blob/master/img/kNNk6.png?raw=true)

And that's optimal *k* all features considered:

![LOOkNN4](https://github.com/toxazol/machineLearning/blob/master/img/LOOkNN4.png?raw=true)

## k **weighted** nearest neighbor algorithm
![kwNN](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2012-56-36.png?raw=true)

> *ω(i)* is a monotonically decreasing sequence of real-number weights, specifying contribution of *i-th* neighbor in *u* classification.

Such sequence can be obtained as geometric sequence with scale factor *q<1*. Where optimal *q* is found by minimizing empirical risk with LOO.

![kWNNLOO2](https://github.com/toxazol/machineLearning/blob/master/img/kWNNLOO2.png?raw=true)
![kWNNq06k6](https://github.com/toxazol/machineLearning/blob/master/img/kWNNq06k6.png?raw=true)


## Parzen window algorithm
Let' s define *ω(i, u)* as a function of distance rather than neighbor rank.

![parzenw](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-27-53.png?raw=true) 

where *K(z)* is  nonincreasing on [0, ∞)  kernel function. Then our metric classifier will look like this:

![parzen](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-31-51.png?raw=true)

This is how kernel functions graphs look like:
![kernels](https://github.com/toxazol/machineLearning/blob/master/img/main-qimg-ece54bb2db23a4f823e3fdb6058761e8.png?raw=true)

Here are some of them implemented in *R*:
```R
ker1 <- (function(r) max(0.75*(1-r*r),0)) # epanechnikov
ker2 <- (function(r) max(0,9375*(1-r*r),0)) # quartic
ker3 <- (function(r) max(1-abs(r),0)) # triangle
ker4 <- (function(r) ((2*pi)^(-0.5))*exp(-0.5*r*r)) # gaussian
ker5 <- (function(r) ifelse(abs(r)<=1, 0.5, 0)) # uniform
```

Parameter *h* is called "window width" and is similar to number of neighbors *k* in kNN.
Here optimal *h* is found using LOO (epanechnikov kernel, sepal.width & sepal.length only): 

![LOOker1parzen2](https://github.com/toxazol/machineLearning/blob/master/img/LOOker1parzen2.png?raw=true)

all four features:

![LOOker1parzen4](https://github.com/toxazol/machineLearning/blob/master/img/LOOker1parzen4.png?raw=true)

triangle kernel, h=0.4:

![h04ker3parzen2](https://github.com/toxazol/machineLearning/blob/master/img/h04ker3parzen2.png?raw=true)

gaussian kernel, h=0.1:

![h01ker4parzen2](https://github.com/toxazol/machineLearning/blob/master/img/h01ker4parzen2.png?raw=true)

Parzen window algorithm can be modified to suit case-based reasoning better.
It's what we call **parzen window algorithm with variable window width**.
Let *h* be equal to the distance to *k+1* nearest neighbor.
Here is comparison of parzen window classifier (uniform kernel) without and with variable window width modification applied:
![parzenKer5](https://github.com/toxazol/machineLearning/blob/master/img/parzenKer5.png?raw=true)
![parzenKer5Var](https://github.com/toxazol/machineLearning/blob/master/img/parzenKer5Var.png?raw=true)

## Potential function algorithm
It's just a slight modification of parzen window algorithm:

![potential](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-40-38.png)

Now window width *h* depends on training object *x*. *γ* represents
importance of each such object.

All these *2l* params can be obtained/adjusted with the following
procedure:

```R
potentialParamsFinder <- function(dataSet){
  n = dim(dataSet)[1]; m = dim(dataSet)[2]-1
  params = cbind(rep(1,n), rep(0,n))
  for(i in 1:n){
    repeat{
      res = potentialClassifier(dataSet, dataSet[i, 1:m], params)[1]
      if(res == dataSet[i,m+1]) break
      params[i,2] = params[i,2] + 1
    }
  }
  return(params)
}
```

Though this process on a sufficiently big sample can take considerable
amount of time. Here is the result of found and applyied params:

![potential](https://github.com/toxazol/machineLearning/blob/master/img/potential.png)

## STOLP algorithm
This algorithm implements data set compression by finding regular (etalon)
objects (stolps) and removing objects which do not or harmfully affect classification from sample.
To explain how this algorithm works idea of *margin* has to be introduced.
> **Margin** of an object (*M(x)*) shows us, how deeply current object lies within its class.
![margin](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%from%2017-12-16%15-07-53.png)

Here is **R** implementation of STOLP algorithm:
```R
STOLP <- function(set, threshold, classifier, argsList){
  plot(iris[,3:4], bg=colors2[iris$Species], col=colors2[iris$Species])
  rowsNum <- dim(set)[1]
  varsNum <- dim(set)[2]-1
  toDelete = numeric()
  for(i in 1:rowsNum){
    currentArgs = c(list(set, set[i, 1:varsNum]),argsList)
    res = do.call(classifier,currentArgs)
    if(res[1] != set[i, varsNum+1]){
      toDelete <- c(toDelete, i)
    }
  }
  points(set[toDelete,], pch=21, bg='grey', col='grey') # debug
  set = set[-toDelete, ]; rowsNum = rowsNum - length(toDelete)
  labels = levels(set[rowsNum,varsNum+1])
  maxRes = rep(0, length(labels)); names(maxRes)<-labels
  maxLabel = rep(0, length(labels)); names(maxLabel)<-labels
  for(i in 1:rowsNum){
    currentArgs = c(list(set, set[i, 1:varsNum]),argsList)
    res = do.call(classifier,currentArgs)
    if(res[2] > maxRes[res[1]]){
      maxRes[res[1]] = res[2]
      maxLabel[res[1]] = i
    }
  }
  regular = set[maxLabel, ]
  points(regular, pch=21, bg=colors2[regular$Species], col=colors2[regular$Species])
  repeat{
    errCount = 0L; toAdd = 0L; maxAbsMargin = -1
    for(i in 1:rowsNum){
      currentArgs = c(list(regular, set[i, 1:varsNum]),argsList)
      res = do.call(classifier,currentArgs)
      if(res[1] != set[i, varsNum+1]){
        errCount = errCount + 1
        if(as.double(res[2]) > maxAbsMargin)
          toAdd <- i
          maxAbsMargin <- as.double(res[2])
      }
    }
    if(errCount <= threshold)
      return(regular)
    newRegular = set[toAdd,]
    regular = rbind(regular,newRegular)
    points(newRegular, pch=21, bg=colors2[newRegular$Species], col=colors2[newRegular$Species])
  }
}
```

Here are STOLP compression results for kNN, kWNN, parzen windowss algorithms respectively:

![stolpKnn](https://github.com/toxazol/machineLearning/blob/master/img/stolpKnn.png)
![stolpKwnn](https://github.com/toxazol/machineLearning/blob/master/img/stolpKwnn.png)
![stolpParzen](https://github.com/toxazol/machineLearning/blob/master/img/stolpParzen.png)


## Conclusion

Here is a summary of results obtained by LOO for different algorithms on *iris* data set:

| algorithm                                        | best parameter | errors | empirical risk |
| -------------------------------------------------|--------------- |--------|----------------|
| kNN (4 features)                                 |      best k: 19|       3|            0.02|
| kNN (2 features)                                 |       best k: 6|       5|          0.0(3)|
|parzen window (epanechnikov kernel, 4 features)   |    best h: 0.8 |       5|          0.0(3)|
|parzen window (gaussian kernel)                   |    best h: 0.02|       5|          0.0(3)|
|parzen window /w variable window (uniform kernel) |       best k: 1|       5|          0.0(3)|
|parzen window /w variable window (gaussian kernel)|       best k: 1|       6|            0.04|
|parzen window (first 3 kernels, 2 features)       |     best h: 0.4|       6|            0.04|
| kWNN                                             |     best q: 0.6|       6|            0.04|

# Linear classification algorithms
___

## Adaline
___
ADALINE stands for **Ada**ptive **Lin**ear **E**lement. It was developed by Professor Bernard Widrow and his graduate student Ted Hoff at Stanford University in 1960. It is based on the McCulloch-Pitts model and consists of a weight, a bias and a summation function.

Operation: ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_1.svg)

Its adaptation is defined through a cost function (error metric) of the residual ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_2.svg) where ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_3.svg) is the desired input. With the MSE error metric ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_4.svg) adapted weight and bias become: ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_5.svg) and ![adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/adaline_6.svg)

The Adaline has practical applications in the controls area. A single neuron with tap delayed inputs (the number of inputs is bounded by the lowest frequency present and the Nyquist rate) can be used to determine the higher order transfer function of a physical system via the bi-linear z-transform. This is done as the Adaline is, functionally, an adaptive FIR filter. Like the single-layer perceptron, ADALINE has a counterpart in statistical modelling, in this case least squares regression.

![Adaline](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/ADALINE.png)
```R
## Квадратичная функция потерь
lossQuad <- function(x)
{
  return ((x-1)^2)
}
## Стохастический градиент для ADALINE
sg.ADALINE <- function(xl, eta = 1, lambda = 1/6)
{
	l <- dim(xl)[1]
	n <- dim(xl)[2] - 1
	w <- c(1/2, 1/2, 1/2)
	iterCount <- 0
	## initialize Q
	Q <- 0
	for (i in 1:l)
	{
	  ## calculate the scalar product <w,x>
	  wx <- sum(w * xl[i, 1:n])
	  ## calculate a margin
	  margin <- wx * xl[i, n + 1]
	  Q <- Q + lossQuad(margin)
	}
	repeat
	{
	  ## calculate the margins for all objects of the training sample
	  margins <- array(dim = l)
	 
	  for (i in 1:l)
	  {
		xi <- xl[i, 1:n]
		yi <- xl[i, n + 1]
		margins[i] <- crossprod(w, xi) * yi
	  }
	  ## select the error objects
	  errorIndexes <- which(margins <= 0)
	  if (length(errorIndexes) > 0)
	  {
		# select the random index from the errors
		i <- sample(errorIndexes, 1)
		iterCount <- iterCount + 1
		xi <- xl[i, 1:n]
		yi <- xl[i, n + 1]
		## calculate the scalar product <w,xi>
		wx <- sum(w * xi)
		## make a gradient step
		margin <- wx * yi
		## calculate an error
		ex <- lossQuad(margin)
		eta <- 1 / sqrt(sum(xi * xi))
		w <- w - eta * (wx - yi) * xi
		## Calculate a new Q
		Qprev <- Q
		Q <- (1 - lambda) * Q + lambda * ex
	  }
	  else
	  {
		break
	  }
	}
	return (w)
}
```

## Hebb
___
From the point of view of artificial neurons and artificial neural networks, Hebb's principle can be described as a method of determining how to alter the weights between model neurons. The weight between two neurons increases if the two neurons activate simultaneously, and reduces if they activate separately. Nodes that tend to be either both positive or both negative at the same time have strong positive weights, while those that tend to be opposite have strong negative weights.

The following is a formulaic description of Hebbian learning:

![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_1.svg)

Another formulaic description is:

![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_2.svg)

Hebb's Rule is often generalized as ![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_3.svg) ![hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/hebb_4.svg)

![Hebb](https://github.com/naemnamenmea/SMCS/blob/master/Linear/images/Hebbs_rule.png)
```R
## Функция потерь для правила Хэбба
lossPerceptron <- function(x)
{
	return (max(-x, 0))
}
## Стохастический градиент с правилом Хебба
sg.Hebb <- function(xl, eta = 0.1, lambda = 1/6)
{
	l <- dim(xl)[1]
	n <- dim(xl)[2] - 1
	w <- c(1/2, 1/2, 1/2)
	iterCount <- 0
	## initialize Q
	Q <- 0
	for (i in 1:l)
	{
		## calculate the scalar product <w,x>
		wx <- sum(w * xl[i, 1:n])
		## calculate a margin
		margin <- wx * xl[i, n + 1]
		# Q <- Q + lossQuad(margin)
		Q <- Q + lossPerceptron(margin)
	}
	repeat
	{
		## Поиск ошибочных объектов
		margins <- array(dim = l)
		for (i in 1:l)
		{
			xi <- xl[i, 1:n]
			yi <- xl[i, n + 1]
			margins[i] <- crossprod(w, xi) * yi
		}
		## выбрать ошибочные объекты
		errorIndexes <- which(margins <= 0)
		if (length(errorIndexes) > 0)
		{
			# выбрать случайный ошибочный объект
			i <- sample(errorIndexes, 1)
			iterCount <- iterCount + 1
			xi <- xl[i, 1:n]
			yi <- xl[i, n + 1]
			w <- w + eta * yi * xi
		}
		else
		break;
	}
	return (w)
}
```

## Naive Bayesian Classifier (NBC)
___
Below is the Naive Bayes’ Theorem:

P(A | B) = P(A) * P(B | A) / P(B)

Which can be derived from the general multiplication formula for AND events:

P(A and B) = P(A) * P(B | A)

P(B | A) = P(A and B) / P(A)

P(B | A) = P(B) * P(A | B) / P(A)

If I replace the letters with meaningful words as I have been adopting throughout, the Naive Bayes formula becomes:

P(outcome | evidence) = P(outcome) * P(evidence | outcome) / P(evidence)

It is with this formula that the Naive Bayes classifier calculates conditional probabilities for a class outcome given prior information.

The reason it is termed “naive” is because we assume independence between attributes when in reality they may be dependent in some way.

So let`s try to implement the naive Bayesian classifier and see what we get.
Before you use the source below you need install several packages & load some libraries...

```R
install.packages("caret")
install.packages("MASS")
install.packages("klaR")
require("caret")
require(lattice)
require(ggplot2)
require(klaR)
require(MASS)
require(e1071) #predict
```

Loading data & Creating the model

```R
data("iris")
model <- NaiveBayes(Species ~ ., data = iris)
```

**predict** computes the conditional a-posterior probabilities of a categorical class variable given independent predictor variables using the Bayes rule.

```R
preds <- predict(model, iris[,-5])
```

Here they are

```R
> preds$posterior
              setosa   versicolor    virginica
  [1,]  1.000000e+00 2.981309e-18 2.152373e-25
  [2,]  1.000000e+00 3.169312e-17 6.938030e-25
  [3,]  1.000000e+00 2.367113e-18 7.240956e-26
  ...
[149,] 1.439996e-195 3.384156e-07 9.999997e-01
[150,] 2.771480e-143 5.987903e-02 9.401210e-01
```

The last one, we need to know how many error classified, so we need to compare the result of prediction with the class/iris species.

```R
table(predict(model, iris[,-5])$class, iris[,5])

##             y
##              setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         47         3
##   virginica       0          3        47
```

If you want to plot the features with Naive Bayes, you can use this command:

```R
naive_iris <- NaiveBayes(iris$Species ~ ., data = iris)
plot(naive_iris)
```

![b_iris_PL](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_PL.png)
![b_iris_PW](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_PW.png)
![b_iris_SL](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_SL.png)
![b_iris_SW](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/b_iris_SW.png)

## Plug-In
___
Estimate the parameters of the likelihood functions µˆy and Σy acting on the parts of the training samples xl(y) for each class y in Y. Then the sample of evaluations we substitute in the optimal Bayesian classifier. Get a normal Bayesian classifier, which is also known as a _**a wildcard (plug-in)**_.

In the asymptotic expansion of l(y) -> oo assessment µˆy and Σy acting have several optimal properties: they are _*not*displaced, efficient and solvent**_. However, estimates made on short samples may not be accurate enough.

![plug-in](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/plug-in_quadro.png)
```R
## Получение коэффициентов подстановочного алгоритма
getPlugInDiskriminantCoeffs <- function(mu1, sigma1, mu2, sigma2)
{
  ## Line equation: a*x1^2 + b*x1*x2 + c*x2 + d*x1 + e*x2 + f = 0
  invSigma1 <- solve(sigma1)
  invSigma2 <- solve(sigma2)
  f <- log(abs(det(sigma1))) - log(abs(det(sigma2))) +
    mu1 %*% invSigma1 %*% t(mu1) - mu2 %*% invSigma2 %*%
    t(mu2);
  alpha <- invSigma1 - invSigma2
  a <- alpha[1, 1]
  b <- 2 * alpha[1, 2]
  c <- alpha[2, 2]
  beta <- invSigma1 %*% t(mu1) - invSigma2 %*% t(mu2)
  d <- -2 * beta[1, 1]
  e <- -2 * beta[2, 1]
  return (c("x^2" = a, "xy" = b, "y^2" = c, "x" = d, "y"
            = e, "1" = f))
}
```

## LDF
___
Let the covariance matrices of the classes are the same and equal to Σ.

![sigma](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/LDF_sigma.gif)

In this case, the separating surface is piecewise linear. The wildcard algorithm is:

![a](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/LDF_a.gif)

![LDF](https://github.com/naemnamenmea/SMCS/blob/master/Baessian/images/LDF.png)
```R
## Оценка ковариационной матрицы для ЛДФ
estimateFisherCovarianceMatrix <- function(objects1, objects2, mu1, mu2)
{
	rows1 <- dim(objects1)[1]
	rows2 <- dim(objects2)[1]
	rows <- rows1 + rows2
	cols <- dim(objects1)[2]
	sigma <- matrix(0, cols, cols)
	for (i in 1:rows1)
	{
		sigma <- sigma + (t(objects1[i,] - mu1) %*%
		(objects1[i,] - mu1)) / (rows + 2)
	}
	for (i in 1:rows2)
	{
		sigma <- sigma + (t(objects2[i,] - mu2) %*%
		(objects2[i,] - mu2)) / (rows + 2)
	}
	return (sigma)
}
```
