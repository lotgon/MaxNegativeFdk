require(data.table)
fileName<-".\\data\\AUDNZD_1_M1.csv"
d<-fread(fileName)
require(ggplot2)
ggplot(d[Drawdown>250*10^-5 & Drawdown<5000*10^-5], aes(x=Drawdown*100000)) + facet_grid(Tp ~ .) + geom_density()

ggplot(d, aes(x=Drawdown, y=BarDuration)) + geom_point() + facet_grid(Tp ~ .)
