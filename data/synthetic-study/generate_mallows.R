require(dplyr)
require(reshape2)
require(ggplot2)
require(PerMallows)
require(BayesMallows)



#Reference Rankings
#Most unfair
a = seq(1, 100, length.out=100)
#rho0 <- seq(from = 1, to = 100, by = 1)

#Sample Mallows Model

for (disp in c(0, .02, .04, .06, .08, .1))
{
  
  for (rr in c('a')){
    set.seed(10)
    
    pi <- eval(parse(text = rr))
    dispersion <- disp
    Nvoters <- 50
    #R <- sample_mallows(rho0 = rho0, alpha0 = disp, diagnostic = TRUE,
                        #n_samples = 50, burnin = 1, thinning = 1)
    
    R <- rmm(Nvoters,pi,dispersion,"kendall", "distances")
    R <- R - 1 #go back to zero index
    R <- data.frame(R)
    #if (disp == 10){
    #  disp = 1 #easier for analysis
    #}

    write.table(R, file=paste('profile','disp',disp, '.csv', sep = '_'), row.names = FALSE, col.names = FALSE, sep = ",")
    
  }
}