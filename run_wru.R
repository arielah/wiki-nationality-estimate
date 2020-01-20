library(data.table)
library(wru)

names<-fread("annotated_names.tsv")
head(names)

names$ethnicity<-gsub("\"","",names$ethnicity)
names$surname<-ifelse(names$name_last!="",names$name_last,names$name_first)

predictions<-predict_race(voter.file=names,surname.only = T)

not_garbage<-subset(predictions,predictions$pred.asi!=0.0797 | predictions$pred.whi!=0.6665 | predictions$pred.bla!= 0.0853)
nrow(predictions)-nrow(not_garbage)
head(not_garbage)
table(not_garbage$ethnicity)

garbage<-subset(predictions,predictions$pred.asi==0.0797 & predictions$pred.whi==0.6665 & predictions$pred.bla== 0.0853)
head(garbage)
table(garbage$ethnicity)

pcts<-as.data.frame(table(garbage$ethnicity)); setnames(pcts,c("Group","Garbage"))
good<-as.data.frame(table(not_garbage$ethnicity));setnames(good,c("Group","NotGarbage"))
library(dplyr)
pcts<-left_join(pcts,good)

pcts$Recognized<-pcts$NotGarbage/(pcts$NotGarbage+pcts$Garbage)
pcts<-pcts[order(pcts$Recognized),]    #African and Muslim names are far less recognized than other groups, Europe is surprisingly low,
#Caribbean and Pacific are surprisingly high
pcts 

quit()

head(garbage[garbage$ethnicity=="Caribbean"]) #It doesn't know what to do with the name López????
head(not_garbage[not_garbage$ethnicity=="Caribbean"])

head(not_garbage[not_garbage$ethnicity=="Africa"])
colMeans(not_garbage[not_garbage$ethnicity=="Africa",6:10])

head(not_garbage[not_garbage$ethnicity=="Hispanic"])
colMeans(not_garbage[not_garbage$ethnicity=="Hispanic",6:10])

head(not_garbage[not_garbage$ethnicity=="Asia"])
colMeans(not_garbage[not_garbage$ethnicity=="Asia",6:10])

head(not_garbage[not_garbage$ethnicity=="Pacific"])
colMeans(not_garbage[not_garbage$ethnicity=="Pacific",6:10])



names<-fread("ISMB_Keynotes.txt")
head(names)
setnames(names,"last_name","surname")

#names$ethnicity<-gsub("\"","",names$ethnicity)
names$surname<-ifelse(names$name_last!="",names$name_last,names$name_first)

predictions<-predict_race(voter.file=names,surname.only = T)
