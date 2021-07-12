Project2
================
Rashmi Kadam, Dionte Watie
7/6/2021

-   [Introduction](#introduction)
-   [Linear Regresion Model](#linear-regresion-model)
-   [Fitting Random forest model](#fitting-random-forest-model)
-   [Fitted Boosted Tree Model](#fitted-boosted-tree-model)

### Introduction

*Bike Data Analysis for Wednesday*

For this study we will be aiming to predict the number of bike users.
The bike users have been split into two groups that will be the target
variables (response), casual bikers that rent bikes casually and
registered bikers that rent bikes regularly. The predictor variables
that will be in question are:

-   weekday (day of the week)
-   season
-   yr (year)
-   holiday (whether it is a holiday or not)
-   weathersit (weather: rainy, snowy, clear, cloudy)
-   mnth
-   atemp (Feeling temperature)
-   windspeed

The response and predictor variables will be used in various Multiple
Linear Regression Models, Logistics Models, and Tree fits. The models
will then be tested against the testing data set and the results will
determine which model would be best to use for prediction.

``` r
library(tidyverse)
library(corrplot)
library(ggplot2)
library(ggpubr)
library(caret)
library(randomForest)
```

``` r
set.seed(1)

# read Bike data
bikeData <- read_csv("day.csv")
```

    ## 
    ## ── Column specification ─────────────────────────────────────────────────────────────────────────
    ## cols(
    ##   instant = col_double(),
    ##   dteday = col_date(format = ""),
    ##   season = col_double(),
    ##   yr = col_double(),
    ##   mnth = col_double(),
    ##   holiday = col_double(),
    ##   weekday = col_double(),
    ##   workingday = col_double(),
    ##   weathersit = col_double(),
    ##   temp = col_double(),
    ##   atemp = col_double(),
    ##   hum = col_double(),
    ##   windspeed = col_double(),
    ##   casual = col_double(),
    ##   registered = col_double(),
    ##   cnt = col_double()
    ## )

``` r
wnum <- weekday
wnum
```

    ## [1] 3

``` r
# filtering weekday data
bikeDataWD <- bikeData %>% filter(weekday == wnum)

# Correlation graph has been used to select the predictors
Correlation <- cor(select(bikeDataWD, casual, registered, cnt,holiday, mnth, season,  weathersit ,yr, temp, atemp, hum, windspeed))

corrplot(Correlation)
```

![](Wednesday_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
#Selected the predictors and factored the categorical predictors.

bikeDataM <- bikeDataWD %>% select (season, holiday, mnth, weathersit, atemp, windspeed, casual, registered, cnt, yr)

#
bikeDataM$mnth <- as.factor(bikeDataM$mnth)

bikeDataM$season <- factor(bikeDataM$season,
                          levels = c("1", "2","3","4") , 
                          labels = c("Spring","Summer","Fall","Winter"))

bikeDataM$holiday <- factor(bikeDataM$holiday, 
                           levels = c("0", "1") , 
                           labels = c("Working Day","Holiday"))


bikeDataM$weathersit <- factor(bikeDataM$weathersit,
                          levels = c("1", "2","3","4") , 
                          labels = c("Good:Clear/Sunny","Moderate:Cloudy/Mist","Bad: Rain/Snow/Fog",
                                     "Worse: Heavy Rain/Snow/Fog"))

bikeDataM$yr <- factor(bikeDataM$yr,
                      levels = c("0", "1") , 
                      labels = c("2011","2012"))
```

Created train and test data sets

``` r
train <- sample(1:nrow(bikeDataM), size = nrow(bikeDataM)*0.7)
test <- dplyr::setdiff(1:nrow(bikeDataM), train)
bikeDataTrain <- bikeDataM[train, ]
bikeDataTest <- bikeDataM[test, ]


summary(bikeDataTrain)
```

    ##     season          holiday        mnth                         weathersit     atemp       
    ##  Spring:13   Working Day:71   5      : 8   Good:Clear/Sunny          :47   Min.   :0.1193  
    ##  Summer:20   Holiday    : 1   6      : 8   Moderate:Cloudy/Mist      :21   1st Qu.:0.3655  
    ##  Fall  :20                    9      : 8   Bad: Rain/Snow/Fog        : 4   Median :0.5413  
    ##  Winter:19                    10     : 8   Worse: Heavy Rain/Snow/Fog: 0   Mean   :0.5000  
    ##                               7      : 6                                   3rd Qu.:0.6171  
    ##                               8      : 6                                   Max.   :0.7469  
    ##                               (Other):28                                                   
    ##    windspeed          casual         registered        cnt          yr    
    ##  Min.   :0.0622   Min.   :   9.0   Min.   : 432   Min.   : 441   2011:34  
    ##  1st Qu.:0.1348   1st Qu.: 292.5   1st Qu.:3324   1st Qu.:3794   2012:38  
    ##  Median :0.1726   Median : 589.5   Median :4160   Median :4806            
    ##  Mean   :0.1843   Mean   : 590.3   Mean   :4253   Mean   :4843            
    ##  3rd Qu.:0.2177   3rd Qu.: 789.8   3rd Qu.:5424   3rd Qu.:6343            
    ##  Max.   :0.4154   Max.   :2562.0   Max.   :6946   Max.   :8173            
    ## 

``` r
#Side by side bar plots for month and count by year
  
Year <- bikeDataTrain$yr

ggplot(bikeDataTrain, aes(fill=Year, y=cnt, x=mnth)) + 
    geom_bar(position="dodge", stat="identity") + xlab("Months") + ylab('Total Users')
```

![](Wednesday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
#Boxplot of season versus count

ggplot(bikeDataTrain, aes(x = season,y=cnt)) +
  geom_boxplot(fill="steelblue") +ylab('Total Users')
```

![](Wednesday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
#Scatter plots for casual and registered versus actual temperature

mintemp <- -16
maxtemp <- 50

bikeDataTrain$atemp = bikeDataTrain$atemp * (maxtemp - mintemp) + mintemp
bikeDataTrain$atemp
```

    ##  [1] 15.207836  9.582128 27.167564  7.207514 12.249122  2.124986 27.166442 24.625772  2.666186
    ## [10] 24.334514  4.869200 22.584392 32.334242  2.583158 25.375400 21.624950 19.666664 25.625672
    ## [19]  9.748778 23.542778 28.000286 24.333986 19.501136 20.875586 17.913968 20.335178 18.792428
    ## [28] 29.500664 26.292272 25.042364  3.624308 18.791108 19.791272 29.792714 33.208478 21.960428
    ## [37]  3.124292 26.917886 -6.477322 22.791764 29.208878  6.582692 -5.408782 10.999214 10.706900
    ## [46] 31.583822 20.208722 23.376458 -0.868180 14.164508  2.166764 17.207372 19.207700 32.000414
    ## [55] 23.334350  2.478284 -8.123758 15.040922  8.217380 21.959372 19.919114  3.625100  4.540586
    ## [64] 20.499650 18.169322  7.832600 24.333722 33.292100 -1.458022 12.248792 30.792878 31.584350

``` r
  cTemp <- ggplot(bikeDataTrain,aes(x=atemp, y=casual)) + geom_point() + geom_smooth() + ylim(0, 7000) 
  rTemp <- ggplot(bikeDataTrain, aes(x=atemp, y=registered)) + geom_point() + geom_smooth() + ylim(0, 7000) 
  
  ggarrange(cTemp, rTemp, labels = c("Casual Users", "Registered Users"), ncol = 2, nrow = 1)
```

    ## `geom_smooth()` using method = 'loess' and formula 'y ~ x'
    ## `geom_smooth()` using method = 'loess' and formula 'y ~ x'

![](Wednesday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
#Bar plots split by casual and registered users for season and holiday
rSeason <- ggplot(bikeDataTrain, aes(fill = holiday, x = season,y = registered,)) + geom_bar(position= 'dodge',stat = 'identity')

cSeason <- ggplot(bikeDataTrain, aes(fill = holiday, x = season,y = casual,)) + geom_bar(position= 'dodge',stat = 'identity')

ggarrange(cSeason, rSeason, labels= c("Casual Users", "Registered Users"), ncol = 2, nrow = 1)
```

![](Wednesday_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
#Density plot for weathersit by year 
weather <- ggplot(bikeDataTrain, aes(x= weathersit))
weather + geom_density(adjust= 0.5, alpha= 0.5, aes(fill= Year), kernel="gaussian")
```

![](Wednesday_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
#ECDF plot for count by year
cntPlot <- ggplot(bikeDataTrain, aes(x= cnt))
cntPlot + stat_ecdf(geom = 'step', aes(color= Year)) + ylab("ECDF")
```

![](Wednesday_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
#Summary Statistics by Dionte
#variance, stdev, mean, and median of casual users by month
statsCasual <- bikeDataTrain %>% 
  group_by(mnth) %>% 
  summarise(avg = mean(casual), 
            med = median(casual), 
            var = var(casual), 
            stDev = sd(casual))
statsCasual
```

    ## # A tibble: 12 x 5
    ##    mnth     avg   med     var stDev
    ##    <fct>  <dbl> <dbl>   <dbl> <dbl>
    ##  1 1       80.6   92    1059.  32.5
    ##  2 2      137.   141    6816.  82.6
    ##  3 3      495.   321  194604. 441. 
    ##  4 4      604    547   78624  280. 
    ##  5 5      665    704.  45039. 212. 
    ##  6 6      846    820.  53381. 231. 
    ##  7 7     1185.   887  522416. 723. 
    ##  8 8      914.   916.  45354. 213. 
    ##  9 9      655.   717   79283. 282. 
    ## 10 10     520.   489   53461. 231. 
    ## 11 11     293.   316.   6722.  82.0
    ## 12 12     224.   282.  18320. 135.

``` r
#variance, stdev, mean, and median of registered users by month
statsRegistered <- bikeDataTrain %>% 
  group_by(mnth) %>% 
  summarise(avg = mean(registered), 
            med = median(registered), 
            var = var(registered), 
            stDev = sd(registered))
statsRegistered
```

    ## # A tibble: 12 x 5
    ##    mnth    avg   med      var stDev
    ##    <fct> <dbl> <dbl>    <dbl> <dbl>
    ##  1 1     2056  2085   662114   814.
    ##  2 2     2492. 1897  1798460. 1341.
    ##  3 3     2851. 1871  4615492. 2148.
    ##  4 4     4132. 4020  1804690. 1343.
    ##  5 5     4571. 4366  1917225. 1385.
    ##  6 6     4914. 4875  1418011. 1191.
    ##  7 7     5019. 4878. 1718847. 1311.
    ##  8 8     4817. 4276. 1199149. 1095.
    ##  9 9     5013. 5209  3765768. 1941.
    ## 10 10    4741. 4707  4115782. 2029.
    ## 11 11    4323  4262.  533266.  730.
    ## 12 12    3396. 3744. 4058192. 2014.

``` r
#variance, stdev, mean, and median of total bike users by season
statsCnt <- bikeDataTrain %>% 
  group_by(season) %>% 
  summarise(avg = mean(cnt), 
            med = median(cnt), 
            var = var(cnt), 
            stDev = sd(cnt))
statsCnt
```

    ## # A tibble: 4 x 5
    ##   season   avg   med      var stDev
    ##   <fct>  <dbl> <dbl>    <dbl> <dbl>
    ## 1 Spring 2498. 2192  2179554. 1476.
    ## 2 Summer 5067. 5079  2769411. 1664.
    ## 3 Fall   5858. 5422. 2817974. 1679.
    ## 4 Winter 5144. 5260  2703580. 1644.

``` r
#Calculating z statistic
tapply(bikeDataTrain$casual, INDEX = bikeDataTrain$weathersit, FUN = function(x){x -mean(x)/sd(x)})
```

    ## $`Good:Clear/Sunny`
    ##  [1]  411.35876 1381.35876  139.35876 1196.35876  738.35876  216.35876 2560.35876  371.35876
    ##  [9]  645.35876 1048.35876  786.35876  947.35876  368.35876  666.35876  686.35876  665.35876
    ## [17]  793.35876  746.35876 1075.35876 1092.35876  557.35876 1025.35876  659.35876  797.35876
    ## [25]  196.35876  973.35876  767.35876  882.35876  331.35876   23.35876  303.35876  995.35876
    ## [33]  785.35876   80.35876  107.35876  778.35876 1056.35876  674.35876  989.35876  253.35876
    ## [41]  653.35876  186.35876 1171.35876  642.35876  329.35876  830.35876  870.35876
    ## 
    ## $`Moderate:Cloudy/Mist`
    ##  [1] 402.19878 105.19878 324.19878  90.19878 743.19878 764.19878 726.19878 537.19878 534.19878
    ## [10] 308.19878 478.19878  51.19878 253.19878 511.19878 345.19878  93.19878 319.19878 545.19878
    ## [19] 166.19878 618.19878 417.19878
    ## 
    ## $`Bad: Rain/Snow/Fog`
    ## [1] 252.639202 116.639202 215.639202   7.639202
    ## 
    ## $`Worse: Heavy Rain/Snow/Fog`
    ## NULL

``` r
# summary statistics by Rashmi
#min max stdev and mean of feeling temperature by season
statsAtemp <- bikeDataTrain %>%
  group_by(season) %>%
  summarise(
    atemp.min = min(atemp),
    atemp.max = max(atemp),
    atemp.med = median(atemp),
    atemp.stdev = sd(atemp),
    atemp.mean = mean(atemp))
statsAtemp
```

    ## # A tibble: 4 x 6
    ##   season atemp.min atemp.max atemp.med atemp.stdev atemp.mean
    ##   <fct>      <dbl>     <dbl>     <dbl>       <dbl>      <dbl>
    ## 1 Spring     -8.12      20.2      2.48        7.91       2.90
    ## 2 Summer      3.63      32.0     21.2         7.33      20.5 
    ## 3 Fall       19.7       33.3     27.0         4.19      26.8 
    ## 4 Winter      2.12      23.5     12.2         7.57      12.7

``` r
#min max stdev and mean of total bike users per year
statsYear<- bikeDataTrain %>%
  group_by(yr) %>%
  summarise(
    cnt.min = min(cnt),
    cnt.max = max(cnt),
    cnt.med = median(cnt),
    cnt.stdev = sd(cnt),
    cnt.mean = mean(cnt)) 
statsYear
```

    ## # A tibble: 2 x 6
    ##   yr    cnt.min cnt.max cnt.med cnt.stdev cnt.mean
    ##   <fct>   <dbl>   <dbl>   <dbl>     <dbl>    <dbl>
    ## 1 2011     1162    5180   3900.     1234.    3536 
    ## 2 2012      441    8173   6262.     1766.    6013.

``` r
#min max stdev and mean of total bike users per holiday
statsHoliday<- bikeDataTrain %>%
  group_by(holiday) %>%
  summarise(
    cnt.min = min(cnt),
    cnt.max = max(cnt),
    cnt.med = median(cnt),
    cnt.stdev = sd(cnt),
    cnt.mean = mean(cnt)) 
statsHoliday
```

    ## # A tibble: 2 x 6
    ##   holiday     cnt.min cnt.max cnt.med cnt.stdev cnt.mean
    ##   <fct>         <dbl>   <dbl>   <dbl>     <dbl>    <dbl>
    ## 1 Working Day     441    8173    4785     1961.    4807.
    ## 2 Holiday        7403    7403    7403       NA     7403

``` r
#contingency table 

table(bikeDataTrain$holiday, bikeDataTrain$season)
```

    ##              
    ##               Spring Summer Fall Winter
    ##   Working Day     13     20   19     19
    ##   Holiday          0      0    1      0

### Linear Regresion Model

Idea of linear regression model -Linear regression model assumes a
linear relationship between the input variables (x) and the single
output variable(y).The linear equation assigns one scale factor to each
input value or column, called a coefficient and represented by the
capital Greek letter Beta (B). It is a slope term.Regression models
estimate the values of Beta. Betas are chosen by using ordinary least
square method.Ordinary least squares minimize the sum of squared
residuals assuming normality and constant variance on error terms.It is
called linear regression because it is linear in parameters.

``` r
#Fitting multiple regression models
set.seed(1)

bikeDataTrainF <- bikeDataTrain %>% select(cnt,season,weathersit,atemp,yr)
bikeDataTestF <- bikeDataTest %>% select(cnt,season,weathersit,atemp,yr)

lmRM<-lm(log(cnt)~.,data=bikeDataTrainF)
summary(lmRM)
```

    ## 
    ## Call:
    ## lm(formula = log(cnt) ~ ., data = bikeDataTrainF)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.81558 -0.10249  0.01978  0.13244  0.32455 
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                     7.518681   0.065948 114.009  < 2e-16 ***
    ## seasonSummer                    0.276688   0.094259   2.935  0.00462 ** 
    ## seasonFall                      0.308639   0.109089   2.829  0.00622 ** 
    ## seasonWinter                    0.570279   0.078680   7.248 6.82e-10 ***
    ## weathersitModerate:Cloudy/Mist -0.124118   0.055793  -2.225  0.02964 *  
    ## weathersitBad: Rain/Snow/Fog   -0.997421   0.104924  -9.506 7.46e-14 ***
    ## atemp                           0.024167   0.003541   6.825 3.78e-09 ***
    ## yr2012                          0.418598   0.047148   8.878 9.23e-13 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.196 on 64 degrees of freedom
    ## Multiple R-squared:  0.881,  Adjusted R-squared:  0.868 
    ## F-statistic: 67.72 on 7 and 64 DF,  p-value: < 2.2e-16

``` r
lmFit <- train(log(cnt) ~ ., data = bikeDataTrainF , 
         method = "lm", 
         preProcess = c("center", "scale"),
         trControl = trainControl(method = "cv", number = 10))

predlm <- predict(lmFit, newdata = dplyr::select(bikeDataTestF,-cnt))

lmRM <- postResample(predlm, bikeDataTestF$cnt)

lmRMSE <- lmRM["RMSE"]

lmRMSE
```

    ##     RMSE 
    ## 4376.026

``` r
#Fitting Multiple Linear Regression model
#Using BIC to select predictors for the best fit model
set.seed(1)

#bic_selection = step(
#  lm(cnt ~ 1, bikeDataTrain),
#  scope = cnt ~ season + holiday + mnth + weathersit + atemp + windspeed + Year,
#  direction = "both", k = log(nrow(bikeDataTrain))
#)
#Best fit linear regression model
bikeDataTrainF2 <- bikeDataTrain %>% select(cnt, atemp, season,weathersit)
bikeDataTestF2 <- bikeDataTest %>% select(cnt, atemp, season,weathersit)

bestLm <- lm(cnt ~ atemp + season + weathersit, data = bikeDataTrain)
bestLm
```

    ## 
    ## Call:
    ## lm(formula = cnt ~ atemp + season + weathersit, data = bikeDataTrain)
    ## 
    ## Coefficients:
    ##                    (Intercept)                           atemp                    seasonSummer  
    ##                        2887.58                           79.56                          971.18  
    ##                     seasonFall                    seasonWinter  weathersitModerate:Cloudy/Mist  
    ##                        1055.15                         1900.43                         -938.76  
    ##   weathersitBad: Rain/Snow/Fog  
    ##                       -3367.17

``` r
summary(bestLm)
```

    ## 
    ## Call:
    ## lm(formula = cnt ~ atemp + season + weathersit, data = bikeDataTrain)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -2259.34 -1161.25   -39.09  1087.33  2206.86 
    ## 
    ## Coefficients:
    ##                                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                     2887.58     404.05   7.147 9.55e-10 ***
    ## atemp                             79.56      23.51   3.384 0.001214 ** 
    ## seasonSummer                     971.18     625.20   1.553 0.125186    
    ## seasonFall                      1055.15     724.29   1.457 0.149981    
    ## seasonWinter                    1900.43     520.83   3.649 0.000526 ***
    ## weathersitModerate:Cloudy/Mist  -938.76     368.38  -2.548 0.013195 *  
    ## weathersitBad: Rain/Snow/Fog   -3367.17     688.37  -4.892 6.90e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1301 on 65 degrees of freedom
    ## Multiple R-squared:  0.601,  Adjusted R-squared:  0.5641 
    ## F-statistic: 16.32 on 6 and 65 DF,  p-value: 2.343e-11

``` r
plot(bestLm)
```

![](Wednesday_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->![](Wednesday_files/figure-gfm/unnamed-chunk-15-2.png)<!-- -->![](Wednesday_files/figure-gfm/unnamed-chunk-15-3.png)<!-- -->![](Wednesday_files/figure-gfm/unnamed-chunk-15-4.png)<!-- -->

``` r
bestLmFit <- train(cnt~ atemp + season +weathersit, data= bikeDataTrainF2,
                   method = "lm",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "cv", number = 10))
lmFitPred <- predict(bestLmFit, newdata = dplyr::select(bikeDataTestF2,-cnt))

lm2RM <- postResample(lmFitPred, bikeDataTestF2$cnt)

lm2RMSE <- lm2RM["RMSE"]

lm2RMSE
```

    ##     RMSE 
    ## 1681.991

### Fitting Random forest model

Random Forest model is tree based method used to prediction. It is
powerful ensembling machine learning algorithm which extends the idea of
bagging but instead of including every predictor, we are including
subset of predictors. It works by creating bootstrap samples fitting a
tree for each bootstrap sample. Random Forest method avoids correlation
amoung the trees. It uses m subset of predictors.

m = SQRT(p) for classification and m = p/3 for regression.

``` r
set.seed(1)


rfFit <- train(cnt ~ ., 
               method = "rf",
               trControl = trainControl(method = "repeatedcv",
                                        repeats = 3,
                                        number = 10),
               tuneGrid = data.frame(mtry = 1:9),
               data = bikeDataTrainF)
rfFit
```

    ## Random Forest 
    ## 
    ## 72 samples
    ##  4 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 65, 65, 65, 64, 65, 65, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared   MAE      
    ##   1     1410.0506  0.8630892  1163.9390
    ##   2      940.9595  0.8902487   751.1957
    ##   3      785.8390  0.8888826   613.3976
    ##   4      748.2639  0.8834927   582.0463
    ##   5      745.1285  0.8762326   576.2661
    ##   6      753.5182  0.8711256   577.6366
    ##   7      752.6455  0.8703734   574.7440
    ##   8      757.6500  0.8688150   577.6923
    ##   9      753.6981  0.8694819   576.8804
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 5.

``` r
rfPred <- predict(rfFit,  newdata = dplyr::select(bikeDataTestF,-cnt))
rfRM <- postResample(rfPred, bikeDataTestF$cnt)
rfRMSE <- rfRM["RMSE"]

rfRMSE
```

    ##     RMSE 
    ## 1940.515

### Fitted Boosted Tree Model

The Boosted tree fit model is used on the bike data set to create a
model candidate. The training data set was used in the model to find the
highest accuracy rate when using the tuning parameters n.trees,
interaction.depth, shrinkage, and n.minobsinnode. When the highest rate
was chosen given the parameters, it was used for prediction against the
testing data set. Finally, the predictions was tested finding the RMSE,
Rsquared, and MAE values

``` r
set.seed(1)

trCtrl <- trainControl(method = "repeatedcv", number = 10, repeats =3)
set.seed(1)
BoostFit <- train(cnt ~., data = bikeDataTrain,
                     method = "gbm",
                      verbose = FALSE,
                     preProcess = c("center", "scale"),
                     trControl = trCtrl)
BoostFit
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 72 samples
    ##  9 predictor
    ## 
    ## Pre-processing: centered (23), scaled (23) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 65, 65, 65, 64, 65, 65, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared   MAE     
    ##   1                   50      449.8871  0.9581122  329.9190
    ##   1                  100      427.4502  0.9589659  315.4090
    ##   1                  150      425.0463  0.9591916  314.2318
    ##   2                   50      457.9924  0.9546814  345.1025
    ##   2                  100      448.3756  0.9558298  344.1261
    ##   2                  150      440.2525  0.9571148  333.1095
    ##   3                   50      451.9480  0.9566330  341.5727
    ##   3                  100      442.4225  0.9568532  336.0311
    ##   3                  150      443.3783  0.9565802  335.3010
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning
    ##  parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 150, interaction.depth = 1, shrinkage =
    ##  0.1 and n.minobsinnode = 10.

``` r
BoostFitPred <- predict(BoostFit, newdata = dplyr::select(bikeDataTest, -cnt))
BoostFitPred
```

    ##  [1] 1937.014 1877.910 1946.804 1816.874 1877.910 1976.529 1883.345 1871.489 3254.064 3705.855
    ## [11] 5234.323 4606.910 4475.820 3291.642 1980.583 2217.289 2027.872 3502.963 3930.892 4291.738
    ## [21] 2188.817 4620.786 1938.296 4686.783 5868.222 5296.688 4611.468 7072.241 7196.599 7127.705
    ## [31] 7208.220 4775.708

``` r
bfRM <- postResample(BoostFitPred, bikeDataTest$cnt)

bfRMSE <- bfRM["RMSE"]
bfRMSE
```

    ##    RMSE 
    ## 469.274

``` r
cRMSEsTitles <- c("Linear Regression Model","Liner Regression Model 2","Random Forest","Boosted Tree")
  
cRMSEs <- c(lm = lmRMSE, lm2 = lm2RMSE, rf = rfRMSE, boost = bfRMSE)
cRMSEs
```

    ##    lm.RMSE   lm2.RMSE    rf.RMSE boost.RMSE 
    ##   4376.026   1681.991   1940.515    469.274

``` r
bestCRMSE <- cRMSEsTitles[which.min(cRMSEs)]

bestCRMSE
```

    ## [1] "Boosted Tree"

*Best model fit is :Boosted Tree model*
