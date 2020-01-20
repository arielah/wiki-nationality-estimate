library(data.table)
names<-fread("third_full_pass.tsv",header=F)

ours<-fread("Penn/Fall_2019/CIS_519/annotated_names_ethnicolr.tsv")
theirs<-fread("Penn/Fall_2019/CIS_519/wiki_name_race.csv")
ourstats<-as.data.frame(table(ours$ethnicity));setnames(ourstats,c("Group","OurFreq"))
theirstats<-as.data.frame(table(theirs$race));setnames(theirstats,c("Group","TheirFreq"))

library(dplyr)
stats<-left_join(ourstats,theirstats)
plot(stats$TheirFreq,stats$OurFreq)
cor(stats$TheirFreq,stats$OurFreq)
lm(stats$OurFreq~stats$TheirFreq)

