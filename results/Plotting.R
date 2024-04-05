library(tidyverse)
library(ggnuplot)
library(ggpubr)
library(ggthemes)



#################################### Mallows Plots

syn <- read_csv("synthetic-study/synthetic.csv")%>%
  dplyr::rename(Method = method) 


syn$Method <- factor(syn$Method, levels = c("BORDA","POST-EG","POST-FQ",
                                            "PRE-EG", "PRE-FQ", "RAPF", "EPIRA", "WISE"))
syn <- syn%>%
  mutate(Method=recode(Method,
                       `POST-EG` = "POST-EG*B"))%>%
  mutate(Method=recode(Method,
                       `POST-FQ` = "POST-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-EG` = "PRE-EG*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-FQ` = "PRE-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `WISE` = "WISE*B"))



syn$NDKL <- signif(syn$NDKL, digits = 3)


# Flip so that WISE is plotted last

flip_colors <-c( '#6cb6ff', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#ff7f0e')

flip_shapes <- c(15, 18, 8, 17, 20, 7, 1, 16)

multi_colors<- c( '#ff7f0e','#6cb6ff', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf')
  
multi_shapes <- c(16, 15, 18, 8, 17, 20, 7, 1)

pt_size <- 3 #3
title_size <- 10
linesize <- 1
axistext <- 14
x_stringm <- "Agreement \U03B1 "
y_stringmf <- "NDKL (\U2193)"
y_stringmu <- "ARBO (\U2191)"
y_stringmt <- "Time in s (\U2193)"
y_stringmw <- "WG-RBO (\U2191)"
fill_limits <- c(0,1.4)
textsize <- 3
mallows_breaks <- c(0, .02, 0.04, 0.06, 0.08, 0.1)

make_fairness_plot <-function(dataset, shapes, colors, x_string, x_col, bks) {
  
  data <-  dataset
  
  p <- ggplot(data, aes(color = Method, x  = data[[x_col]], y = NDKL, shape = Method)) +
    geom_point(size = pt_size)+
    geom_line(size = linesize)+
    #theme_gnuplot()+
    xlab(x_string)+
    ylab(y_stringmf)+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
          axis.text.x = element_text(margin = margin(t = 3)),
          axis.text.y = element_text(margin = margin(r = 3)))+
    #ggtitle('Fairness (\U2193)')+
    scale_x_continuous(breaks=bks)+
    scale_shape_manual(values=shapes)+
    scale_color_manual(values=colors)+
    guides(color=guide_legend(nrow=1))+
    guides(shape = guide_legend(nrow = 1))+
    theme(legend.title=element_blank())
  
  return(p)
}

make_accuracy_plot <-function(dataset, shapes, colors, x_string, x_col, bks) {
  
  data <-  dataset
  
  p <- ggplot(data, aes(color = Method, x  = data[[x_col]], y = rbo, shape = Method)) +
    geom_point(size = pt_size)+
    geom_line(size = linesize)+
    #theme_gnuplot()+
    xlab(x_string)+
    ylab(y_stringmu)+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
          axis.text.x = element_text(margin = margin(t = 3)),
          axis.text.y = element_text(margin = margin(r = 3)))+
    #ggtitle('Utility (\U2191)')+
    scale_x_continuous(breaks=bks)+
    scale_shape_manual(values=shapes)+
    scale_color_manual(values=colors)+
    guides(color=guide_legend(nrow=1))+
    guides(shape = guide_legend(nrow = 1))+
    theme(legend.title=element_blank())
  
  return(p)
}

make_timing_plot <-function(dataset, shapes, colors, x_string, x_col, bks) {
  
  data <-  dataset
  
  p <- ggplot(data, aes(color = Method, x  = data[[x_col]], y = times, shape = Method)) +
    geom_point(size = pt_size)+
    geom_line(size = linesize)+
    #theme_gnuplot()+
    xlab(x_string)+
    ylab(y_stringmt)+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
          axis.text.x = element_text(margin = margin(t = 3)),
          axis.text.y = element_text(margin = margin(r = 3)))+
    #ggtitle('Time (\U2193)')+
    scale_x_continuous(breaks=bks)+
    scale_shape_manual(values=shapes)+
    scale_color_manual(values=colors)+
    guides(color=guide_legend(nrow=1))+
    guides(shape = guide_legend(nrow = 1))+
    theme(legend.title=element_blank())
  
  return(p)
}

make_wkt_plot <-function(dataset, shapes, colors, x_string, x_col, bks) {
  
  data <-  dataset
  
  p <- ggplot(data, aes(color = Method, x  = data[[x_col]], y = wig_rbo, shape = Method)) +
    geom_point(size = pt_size)+
    geom_line(size = linesize)+
    #theme_gnuplot()+
    xlab(x_string)+
    ylab(y_stringmw)+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
          axis.text.x = element_text(margin = margin(t = 3)),
          axis.text.y = element_text(margin = margin(r = 3)))+
    #ggtitle('Avg. WG-RBO (\U2191)')+
    scale_x_continuous(breaks=bks)+
    scale_shape_manual(values=shapes)+
    scale_color_manual(values=colors)+
    guides(color=guide_legend(nrow=1))+
    guides(shape = guide_legend(nrow = 1))+
    theme(legend.title=element_blank())
  
  return(p)
}


mallows_fair <- make_fairness_plot(syn, flip_shapes, flip_colors, x_stringm, 'dispersion', mallows_breaks)
mallows_util <- make_accuracy_plot(syn, flip_shapes, flip_colors, x_stringm, 'dispersion', mallows_breaks)
mallows_time <- make_timing_plot(syn, flip_shapes, flip_colors, x_stringm, 'dispersion', mallows_breaks)
mallows_wkt <- make_wkt_plot(syn, flip_shapes, flip_colors, x_stringm, 'dispersion', mallows_breaks)

ml <- syn
ml$Method <- factor(ml$Method, levels = c("WISE*B", "BORDA","POST-EG*B","POST-FQ*B",
                                            "PRE-EG*B", "PRE-FQ*B", "RAPF", "EPIRA"))

mallows_legend <- make_fairness_plot(ml, multi_shapes, multi_colors, x_stringm, 'dispersion', mallows_breaks)



pdfwidth <- 14
pdfheight <- 2.5


fig_mallows <- ggarrange(mallows_fair, mallows_util,mallows_time, mallows_wkt,
                        ncol = 4, nrow = 1, legend.grob = get_legend(mallows_legend), legend = "top")

ggsave(fig_mallows, filename = glue::glue("plots/mallows_analysis.pdf"), device = cairo_pdf,
       width = pdfwidth, height = pdfheight, units = "in")


#################################### Disjoint Plots
#changed meaning of disjoint to overlap so have to update delta to be 1 - prop.
dis <- read_csv("disjoint-study/disjoint.csv")%>%
  dplyr::rename(Method = method) 

dis$overlap <- 1 - dis$disjointness


dis$Method <- factor(dis$Method, levels = c("BORDA","POST-EG","POST-FQ",
                                            "PRE-EG", "PRE-FQ", "RAPF", "EPIRA", "WISE"))
dis <- dis%>%
  mutate(Method=recode(Method,
                       `POST-EG` = "POST-EG*B"))%>%
  mutate(Method=recode(Method,
                       `POST-FQ` = "POST-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-EG` = "PRE-EG*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-FQ` = "PRE-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `WISE` = "WISE*B"))



x_stringd <- 'Overlap Proportion \U03B4'
disjoint_breaks <- c(0, .2, 0.4, 0.6, 0.8, 1.0)
disjoint_fair <- make_fairness_plot(dis, flip_shapes, flip_colors, x_stringd, 'overlap',disjoint_breaks)
disjoint_util <- make_accuracy_plot(dis, flip_shapes, flip_colors, x_stringd, 'overlap',disjoint_breaks)
disjoint_time <- make_timing_plot(dis, flip_shapes, flip_colors, x_stringd, 'overlap',disjoint_breaks)
disjoint_wkt <- make_wkt_plot(dis, flip_shapes, flip_colors, x_stringd, 'overlap',disjoint_breaks)



fig_disjoint <- ggarrange(disjoint_fair, disjoint_util,disjoint_time, disjoint_wkt,
                         ncol = 4, nrow = 1, legend.grob = get_legend(mallows_legend), legend = "top")

ggsave(fig_disjoint, filename = glue::glue("plots/disjoint_analysis.pdf"), device = cairo_pdf,
       width = pdfwidth, height = pdfheight, units = "in")


#################################### Mallows Tuning Parameter Plots

tuning_data <- read_csv("synthetic-study/tuning.csv")%>%
  dplyr::rename(Method = method) 


tuning_data <- tuning_data%>%
  mutate(Method=recode(Method,
                       `POST-EG` = "POST-EG*B"))%>%
  mutate(Method=recode(Method,
                       `POST-FQ` = "POST-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-EG` = "PRE-EG*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-FQ` = "PRE-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `WISE` = "WISE*B"))



make_tuning_plot <-function(Method, x_tstring) {
  
  df <-  tuning_data %>%
    filter(.data$Method == .env$Method)
  
  linesz <- 1
  #coeff <- .65
  coeff <- .2
  fair_string <- "NDKL (\U2193)"
  util_string <- "ARBO (\U2191)"
  p <- ggplot(df, aes(x = tuning)) + 
    geom_line(aes(y = NDKL, color = 'NDKL (\U2193)'),size = linesz, ) + 
    geom_line(aes(y = rbo - coeff, color = 'ARBO (\u2191)'),size = linesz, linetype= 'dashed')+
    labs(title =  glue::glue({Method}))+
    xlab(x_tstring)+
    ylab(fair_string)+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.y.right = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
          axis.text.x = element_text(margin = margin(t = 3)),
          axis.text.y = element_text(margin = margin(r = 3)),
          axis.text.y.right = element_text(margin = margin(r = 3.5)),)+
    guides(color=guide_legend(nrow=1))+
    guides(shape = guide_legend(nrow = 1))+
    scale_color_manual('Metric', values=c('#000000', '#000000'))+
    scale_y_continuous(
      fair_string, limits = c(0.08, .81),
      sec.axis = sec_axis(~ . * 1+ coeff, name = util_string))
  return(p)
}


make_tuning_plot_reverse <-function(Method, x_tstring) {
  
  df <-  tuning_data %>%
    filter(.data$Method == .env$Method)
  
  linesz <- 1
  #coeff <- .65
  coeff <- .2
  fair_string <- "NDKL (\U2193)"
  util_string <- "ARBO (\U2191)"
  p <- ggplot(df, aes(x = tuning)) + 
    geom_line(aes(y = NDKL, color = 'NDKL (\U2193)'),size = linesz) + 
    geom_line(aes(y = rbo - coeff, color = 'ARBO (\u2191)'),size = linesz, linetype= 'dashed')+
    labs(title =  glue::glue({Method}))+
    xlab(x_tstring)+
    ylab(fair_string)+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.y.right = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
          axis.text.x = element_text(margin = margin(t = 3)),
          axis.text.y = element_text(margin = margin(r = 3)),
          axis.text.y.right = element_text(margin = margin(r = 3.5)),)+
    guides(color=guide_legend(nrow=1))+
    guides(shape = guide_legend(nrow = 1))+
    scale_color_manual('Metric', values=c('#000000', '#000000'))+
    scale_y_continuous(
      fair_string, limits = c(0.08, .81),
      sec.axis = sec_axis(~ . * 1+ coeff, name = util_string)) +
    scale_x_reverse()
  return(p)
}

epira_t<-make_tuning_plot('EPIRA', '\U03B3 More Fair \U2192')
borda_t<-make_tuning_plot('BORDA', 'No Tuning Parameter')
RAPF_t<-make_tuning_plot('RAPF', 'No Tuning Parameter')
post_eg_t<-make_tuning_plot('POST-EG*B', '\U03B5 More Fair \U2192')
pre_eg_t<-make_tuning_plot('PRE-EG*B', '\U03B5 More Fair \U2192')
wise_t<-make_tuning_plot('WISE*B', '\U03BB More Fair \U2192')
post_fq_t<-make_tuning_plot_reverse('POST-FQ*B', 'DDP More Fair \U2192')
pre_fq_t<-make_tuning_plot_reverse('PRE-FQ*B', 'DDP More Fair \U2192')

fig_tuning <- ggarrange(wise_t, borda_t, epira_t,post_eg_t, post_fq_t,
                        pre_eg_t, pre_fq_t, RAPF_t, 
                        ncol = 4, nrow = 2, common.legend = TRUE, legend = "top")

pdfwidth <- 13
pdfheight <- 4

ggsave(fig_tuning, filename = glue::glue("plots/tuning.pdf"), device = cairo_pdf,
       width = pdfwidth, height = pdfheight, units = "in")

#################################### Datasets Fairness-Utility Plots

ibm <- read_csv("ibmhr/results_ibmhr.csv")%>%
  dplyr::rename(Method = method)
ibm$Method <- factor(ibm$Method, levels = c("WISE","COMBmnz","POST-EG","POST-FQ",
                                            "PRE-EG", "PRE-FQ", "RAPF", "EPIRA"))

ibm <- ibm%>%
  mutate(Method=recode(Method,
                       `POST-EG` = "POST-EG*C"))%>%
  mutate(Method=recode(Method,
                       `POST-FQ` = "POST-FQ*C"))%>%
  mutate(Method=recode(Method,
                       `PRE-EG` = "PRE-EG*C"))%>%
  mutate(Method=recode(Method,
                       `PRE-FQ` = "PRE-FQ*C"))%>%
  mutate(Method=recode(Method,
                       `WISE` = "WISE*C"))
# %>%
#   mutate(dataset=recode(dataset,
#                        `IBMHR` = "IBM-HR"))

econf <- read_csv("econf/results_econfreedom.csv")%>%
  dplyr::rename(Method = method)
econf$Method <- factor(econf$Method, levels = c("WISE","COMBmnz","POST-EG","POST-FQ",
                                                "PRE-EG", "PRE-FQ", "RAPF")) 

econf <- econf%>%
  mutate(Method=recode(Method,
                       `POST-EG` = "POST-EG*C"))%>%
  mutate(Method=recode(Method,
                       `POST-FQ` = "POST-FQ*C"))%>%
  mutate(Method=recode(Method,
                       `PRE-EG` = "PRE-EG*C"))%>%
  mutate(Method=recode(Method,
                       `PRE-FQ` = "PRE-FQ*C"))%>%
  mutate(Method=recode(Method,
                       `WISE` = "WISE*C"))

adult <- read_csv("adult/results_adult.csv")%>%
  dplyr::rename(Method = method)
adult$Method <- factor(adult$Method, levels = c("WISE","BORDA","POST-EG","POST-FQ",
                                                "PRE-EG", "PRE-FQ", "RAPF", "EPIRA"))
adult <- adult%>%
  mutate(Method=recode(Method,
                       `POST-EG` = "POST-EG*B"))%>%
  mutate(Method=recode(Method,
                       `POST-FQ` = "POST-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-EG` = "PRE-EG*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-FQ` = "PRE-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `WISE` = "WISE*B"))
                
happiness <- read_csv("world-happiness/results_worldhappiness.csv")%>%
  dplyr::rename(Method = method)
happiness$Method <- factor(happiness$Method, levels = c("WISE","BORDA","POST-EG","POST-FQ",
                                                "PRE-EG", "PRE-FQ", "RAPF")) 

happiness <- happiness%>%
  mutate(Method=recode(Method,
                       `POST-EG` = "POST-EG*B"))%>%
  mutate(Method=recode(Method,
                       `POST-FQ` = "POST-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-EG` = "PRE-EG*B"))%>%
  mutate(Method=recode(Method,
                       `PRE-FQ` = "PRE-FQ*B"))%>%
  mutate(Method=recode(Method,
                       `WISE` = "WISE*B"))






make_fairutil_plot <-function(data, dataset, fair_rep, shapes, colors) {
  x_string <- paste('NDKL', glue::glue("{fair_rep}"), '(\U2193)' )
  y_string <- 'ARBO (\U2191)'
  
  p <- ggplot(data, aes(color = Method, x  = NDKL, y = rbo, shape = Method)) +
    geom_point(size = pt_size)+
    #geom_line(size = linesize)+
    #theme_gnuplot()+
    xlab(x_string)+
    ylab(y_string)+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
          axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
          axis.text.x = element_text(margin = margin(t = 3)),
          axis.text.y = element_text(margin = margin(r = 3)))+
    ggtitle(glue::glue("{dataset}"))+
    scale_shape_manual(values=shapes)+
    scale_color_manual(values=colors)+
    guides(color=guide_legend(nrow=2))+
    guides(shape = guide_legend(nrow = 2))+
    theme(legend.title=element_blank())
    return(p)
}

ibm_colors <- c( '#ff7f0e','#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf')
ibm_p <- make_fairutil_plot(ibm, 'IBM-HR', 'prop.', multi_shapes, ibm_colors)

#Need to adjust due to partial
econf_colors<- c( '#ff7f0e','#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2')
econf_shapes <- c(16, 15, 18, 8, 17, 20, 7)
econf_p <- make_fairutil_plot(econf, 'Econ-Freedom', 'eq.', econf_shapes, econf_colors)

#Adjust by methods
adult_colors<- c( '#ff7f0e','#6cb6ff', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf')
adult_p <- make_fairutil_plot(adult, 'Adult', 'eq.', multi_shapes, adult_colors)

#Adjust for partial
#Need to adjust due to partial
happiness_colors<- c( '#ff7f0e','#6cb6ff', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2')
happiness_shapes <- c(16, 15, 18, 8, 17, 20, 7)
happiness_p <- make_fairutil_plot(happiness, 'Happiness', 'prop.', happiness_shapes, happiness_colors)

#Hack shared legend
leg_colors<- c( '#ff7f0e','#6cb6ff', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf')
legend <- ibm %>%
  mutate(Method=recode(Method,
                       `COMBmnz` = "BORDA"))
                       
shared_legend <- make_fairutil_plot(adult, 'Adult', 'eq', multi_shapes, leg_colors)

coombs_fairutil <- ggarrange(ibm_p, econf_p,
                         ncol = 2, nrow = 1, common.legend = TRUE, legend = "top")

borda_fairutil <- ggarrange(happiness_p, adult_p,
                             ncol = 2, nrow = 1, legend.grob = get_legend(shared_legend), legend = "top")

fu_w <- 6
fu_h <- 2.75
ggsave(coombs_fairutil, filename = glue::glue("plots/coombs_fairutil.pdf"), device = cairo_pdf,
       width = fu_w, height = fu_h, units = "in")

ggsave(borda_fairutil, filename = glue::glue("plots/borda_fairutil.pdf"), device = cairo_pdf,
       width = fu_w, height = fu_h, units = "in")

#################################### Datasets AVG WG-RBO Plots


make_wg_plot <-function(data, dataset, fills) {
p <- ggplot(data, aes(x  = Method, y = wig_rbo, fill = Method)) +
  geom_bar(stat="identity")+
  #geom_line(size = linesize)+
  #theme_gnuplot()+
  ylab('WG-RBO (\U2191)')+
  theme_gnuplot()+
  theme(#legend.position = "top",
        #legend.direction = "horizontal",
        axis.title.y = element_text(size = axistext, margin = margin(r = 1)),
        axis.title.x = element_text(size = axistext,margin = margin(t = 1)),
        axis.text.x = element_text(margin = margin(t = 3)),
        axis.text.y = element_text(margin = margin(r = 3)))+
  ggtitle(glue::glue("{dataset}"))+
  scale_fill_manual(values=fills)+
  #guides(fill=guide_legend(nrow=2))+
  theme(axis.text.x = element_text(angle = 45, hjust=1),
        #axis.ticks.x=element_blank()
    )+
  theme(legend.position="none")
return(p)
}

ibm_wg <- make_wg_plot(ibm, 'IBM-HR', ibm_colors)
econf_wg <- make_wg_plot(econf, 'Econ-Freedom', econf_colors)

adult_wg <- make_wg_plot(adult, 'Adult', adult_colors)
happiness_wg <- make_wg_plot(happiness, 'Happiness', happiness_colors)


sharedwg_legend <- make_wg_plot(legend, 'IBM-HR',leg_colors)

coombs_avgwgrbo <- ggarrange(ibm_wg, econf_wg,
                             ncol = 2, nrow = 1)

borda_avgwgrbo <- ggarrange(happiness_wg, adult_wg,
                            ncol = 2, nrow = 1)

wg_w <- 7
wg_h <- 2.5
ggsave(coombs_avgwgrbo, filename = glue::glue("plots/coombs_avgwgrbo.pdf"), device = cairo_pdf,
       width = wg_w, height = wg_h, units = "in")

ggsave(borda_avgwgrbo, filename = glue::glue("plots/borda_avgwgrbo.pdf"), device = cairo_pdf,
       width = wg_w, height = wg_h, units = "in")
