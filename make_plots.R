library(data.table)
authors<-fread("authors_TFIDF_distribution.csv")
keynotes<-fread("keynotes_TFIDF_distribution.csv")

authors$V1<-NULL
keynotes$V1<-NULL

a<-as.data.frame(colMeans(authors))
a$Group<-rownames(a)
k<-as.data.frame(colMeans(keynotes))
k$Group<-rownames(k)

library(dplyr)

thing<-full_join(a,k)
thing
setnames(thing,c("PubMed Authors","Group","ISMB Keynote Speakers"))

library(reshape2)
thingy<-melt(thing,id.vars = 'Group')
setnames(thingy,"variable","Category")
library(ggplot2)

png(filename="Naive_bayes_plot.png", width=1200, height=650)
ggplot(thingy, aes(x=Group, y=value, fill=Category)) + geom_bar(stat='identity',position=position_dodge(preserve='single')) + 
  ggtitle("Probabilistic Estimate of Demographics, Naive Bayes TF-IDF, 4-chars") + xlab("Nationality") + ylab("Proportion") +
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=20,face="bold"),
	plot.title = element_text(size=30),
	legend.text= element_text(size=20))
dev.off()


thing <-fread("LSTM_senior.txt")
thing<-subset(thing,select=c("Group","Probabilistic PubMed LastAuthor","Probabilistic ISMB"))
thing<-setnames(thing,c("Group","PubMed","ISMB"))
thing$PubMed<-as.numeric(sub("%", "", thing$PubMed))/100
thing$ISMB<-as.numeric(sub("%", "", thing$ISMB))/100
thingy<-melt(thing,id.vars='Group')

setnames(thingy,"variable","Category")

png(filename="LSTM_senior_plot.png", width=1200,height=650)
ggplot(thingy, aes(x=Group, y=value, fill=Category)) + geom_bar(stat='identity',position=position_dodge(preserve='single')) + 
  ggtitle("Distribution of ISMB Keynotes vs PubMed Senior Authors, LSTM, 3-chars") + xlab("Nationality") + ylab("Proportion") +
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=20,face="bold"),
	plot.title = element_text(size=30),
	legend.text= element_text(size=20))
dev.off()


