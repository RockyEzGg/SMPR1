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