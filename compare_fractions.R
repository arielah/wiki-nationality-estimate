library(data.table)

countries_grouped<-fread("country_list_ethnicolr",header=F); setnames(countries_grouped,c("Name","Region","Eth"))
#populations<-fread("country_population");setnames(populations,c("Name","Population"))
#countries_grouped<-left_join(countries_grouped,populations)                  

aggregates<-inner_join(countries_grouped,populations)
ugh<-right_join(countries_grouped,populations)

country<-fread("country_population")
country$Ethnicolr<-as.factor(country$Ethnicolr)

truth<-aggregate(country$Population,by=list(Category=country$Ethnicolr), FUN=sum);setnames(truth,c("Group","TrueFreq"))


ours<-fread("annotated_names_ethnicolr.tsv")
theirs<-fread("wiki_name_race.csv")
ourstats<-as.data.frame(table(ours$ethnicity));setnames(ourstats,c("Group","OurFreq"))
theirstats<-as.data.frame(table(theirs$race));setnames(theirstats,c("Group","TheirFreq"))

library(dplyr)
stats<-left_join(ourstats,theirstats)
stats<-left_join(stats,truth)

stats<-subset(stats,stats$TrueFreq>100)
stats[is.na(stats)]<-0

plot(stats$TheirFreq,stats$TrueFreq)
plot(stats$OurFreq,stats$TrueFreq)
cor(stats$TheirFreq,stats$TrueFreq)
cor(stats$OurFreq,stats$TrueFreq)

stats$OurFraction<-stats$OurFreq/stats$TrueFreq
stats$TheirFraction<-stats$TheirFreq/stats$TrueFreq

samename<-inner_join(country,countries_grouped)
different<-subset(samename,samename$Region1!=samename$Region2)


raw_names <- fread("third_full_pass.tsv")
head(raw_names)
countries<-fread("nationality_to_country_full.txt")

setnames(raw_names,"V3","Nationality")
raw_names<-inner_join(raw_names,countries,by="Nationality")
ours<-as.data.frame(table(raw_names$Country));setnames(ours,c("Country","Representation"))

regions<-fread("country_to_region_full.txt")
regions<-subset(regions, select=c("Country","Population"))

regions<-left_join(regions,ours)
regions[is.na(regions)]<-0
cor(regions$Population,regions$Representation)
plot(log(regions$Population),regions$Representation)
abline(lm(regions$Representation~regions$Population))
text(log(regions$Population), regions$Representation, labels=regions$Country, cex= 0.7)
regions<-subset(regions,regions$Population<1.5e8 & regions$Representation<15000)
