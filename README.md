



# Project 2 - Bike Sharing Data Analysis

## Purpose:

This project analyses Bike Sharing Data provided by [UCI](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). We have summarized the data and then tried to predict the number of users using predictive models. We are also automating output file generation (provided below) by each day.

## Packages used

Following packages are used in this project:

* library(tidyverse) - Collection of packages to read the file and other frequently used functions
* library(corrplot) - Used to render correlation matrix  
* library(ggplot2) - To render graphs
* library(ggpubr) - This package provides easy to use functions along with ggplot2
* library(caret) - This package provides functions to run models
* library(randomForest) - This package provides functions to run Random forest model


## Analysis reports

* The analysis for [Monday is available here](MondayAnalysis.md).
* The analysis for [Tuesday is available here](TuesdayAnalysis.md).
* The analysis for [Wednesday is available here](WednesdayAnalysis.md).
* The analysis for [Thursday is available here](ThursdayAnalysis.md).
* The analysis for [Friday is available here](FridayAnalysis.md).
* The analysis for [Saturday is available here](SaturdayAnalysis.md).
* The analysis for [Sunday is available here](SundayAnalysis.md).

## Rmakrdown Code [to gererate above reports]


days <- c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday")

weekday <- 0

for (day in days) {

  rmarkdown::render('Project2.Rmd', output_file = paste0(day,'Analysis.PDF'),
                    params = list(
                      day = day,
                      weekday = weekday
                    ))
  weekday = weekday + 1
}




