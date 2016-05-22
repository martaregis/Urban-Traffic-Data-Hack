library(ggplot2)
library(xts)

qacf <- function(x, lag.max=NULL) {
    bacf <- acf(x, lag.max=lag.max, plot = FALSE)
    bacfdf <- with(bacf, data.frame(lag, acf))
    upperCI <- qnorm((1 + 0.95)/2)/sqrt(bacf$n.used)
    lowerCI <- -qnorm((1 + 0.95)/2)/sqrt(bacf$n.used)

    q <- ggplot(data = bacfdf, mapping = aes(x = lag, y = acf))
    q <- q + geom_hline(aes(yintercept = 0))
    q <- q + geom_segment(mapping = aes(xend = lag, yend = 0), colour='grey25')
    q <- q + ylab('acf')
    q <- q + geom_hline(yintercept = lowerCI, color="blue", size = 0.2)
    q <- q + geom_hline(yintercept = upperCI, color="blue", size = 0.2)
    q <- q + geom_hline(yintercept = 0, color="red",  size = 0.3)
    return(q)
}

qccf <- function(x, y, lag.max=NULL) {
    bccf <- ccf(x, y, lag.max=lag.max, plot = FALSE)
    bccfdf <- with(bccf, data.frame(lag, acf))
    upperCI <- qnorm((1 + 0.95)/2)/sqrt(bccf$n.used)
    lowerCI <- -qnorm((1 + 0.95)/2)/sqrt(bccf$n.used)

    q <- ggplot(data = bccfdf, mapping = aes(x = lag, y = acf))
    q <- q + geom_hline(aes(yintercept = 0))
    q <- q + geom_segment(mapping = aes(xend = lag, yend = 0), colour='grey25')
    q <- q + ylab('ccf')
    q <- q + geom_hline(yintercept = lowerCI, color="blue", size = 0.2)
    q <- q + geom_hline(yintercept = upperCI, color="blue", size = 0.2)
    q <- q + geom_hline(yintercept = 0, color="red",  size = 0.3)
    return(q)
}

garages = c('Diezerpoort', 'Emmawijk', 'Noordereiland',
       'Dijkstraat', 'Van.Roijensingel', 'Pas.de.Deux', 'Lubeckplein',
       'Hanzelaan')

df <- read.csv('garage.csv')
timestamps = as.POSIXct(df$time_interval, format='%Y-%m-%d %H:%M:%S')

for (g in garages){
    data <- xts(df[,g], order.by=timestamps)
    p <- qacf(as.numeric(data[complete.cases(data)]), lag.max=48*7)
    p <- p + xlab('lag (x30 minutes)')
    p <- p + theme_grey(base_size = 10)
    ggsave(p, filename=paste('./Plots/ACF_', g, '.pdf', sep=''), width=6, height=4)
}

for (i in 1:(length(garages)-1)){
    for (j in (i+1):length(garages)){
        ilab <- garages[i]
        jlab <- garages[j]
        data <- df[,c(ilab,jlab)]
        data <- data[complete.cases(data),]
        p <- qccf(as.numeric(data[,ilab]), as.numeric(data[,jlab]), lag.max=48*7)
        p <- p + xlab('lag (x30 minutes)')
        p <- p + theme_grey(base_size = 10)
        ggsave(p, filename=paste('./Plots/CCF_', ilab, '_', jlab, '.pdf', sep=''), width=6, height=4)
    }
}
