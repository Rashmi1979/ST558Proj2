
# This file generates Analysis reports for every day.

days <- c("Sunday")#,"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday")

weekday <- 0

for (day in days) {
  print(day)
  rmarkdown::render('Project2.Rmd', output_file = paste0(day,'Analysis.md'),
                    params = list(
                      day = day,
                      weekday = weekday
                    ))
  weekday = weekday + 1
  
}
