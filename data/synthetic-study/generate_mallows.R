require(dplyr)
require(reshape2)
require(ggplot2)
require(PerMallows)

Nalternatives = 200

#Reference Rankings
#Most unfair
a = seq(1, 100, length.out=100)

#Sample Mallows Model

for (disp in c(0, .2, .4, .6, .8, 1))
{
  
  for (rr in c('a')){
    set.seed(1)
    
    pi <- eval(parse(text = rr))
    dispersion <- disp
    Nvoters <- 50
    
    R <- rmm(Nvoters,pi,dispersion,"kendall", "distances")
    R <- R - 1 #go back to zero index
    R <- data.frame(R)
    #if (disp == 10){
    #  disp = 1 #easier for analysis
    #}

    write.table(R, file=paste('profile','disp',disp, '.csv', sep = '_'), row.names = FALSE, col.names = FALSE, sep = ",")
    
  }
}