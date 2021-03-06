source("C:\\Users\\13824\\OneDrive\\data\\frequently_used_graph.R")

data=read.csv("C:\\Users\\13824\\OneDrive\\data\\Wind_Spatio_Temporal_Dataset1\\Wind Spatio-Temporal Dataset1(2009, Ave).csv")
coordinate=read.csv("C:\\Users\\13824\\OneDrive\\data\\Wind_Spatio_Temporal_Dataset1\\Wind Spatio-Temporal Dataset1(Coordinates).csv")
coordinate=apply(coordinate,2,as.numeric)
head(data)
selection=sample(c(1:dim(data)[2]),25)
data=data[,selection]
coordinate=coordinate[selection,]

write.csv(data[,c(1,2,3,7,8,10,16,17,19)],'test_data.csv')
write.csv(coordinate[c(1,2,3,7,8,10,16,17,19),],"test_data(coordinate).csv")

data=read.csv("test_data.csv")
data=apply(data[,2:dim(data)[2]],2,as.numeric)

scatter_point(data[1:100,])
box_plot(data[1:720,])
cor_heat_map(data)



#library(ggplot2)
#X=rep(1:dim(speed_data)[1],dim(speed_data)[2]-1)
#Y=c(as.matrix(speed_data[,2:dim(speed_data)[2]]))
#group=rep(colnames(speed_data)[2:length(colnames(speed_data))],each=dim(speed_data)[1])
#data=cbind(X,Y,group)
#data=as.data.frame(data)
#data[,1]=as.numeric(data[,1])
#data[,2]=as.numeric(data[,2])
#str(data)
#data=data[-which(data[,'group']=='Mean_Speed_30_Turbine35'),]
#ggplot(data[8403:10402,],aes(x=X,y=Y,colour=group))+geom_point()
#ggplot(data,aes(x=group,y=Y))+geom_boxplot(aes(group=group))
#
#data_for_heatMap=speed_data[,3:dim(speed_data)[2]]
#data_for_heatMap=apply(data_for_heatMap,2,as.numeric)
#cor_mat=cor(data_for_heatMap)
#library(reshape2)
#cor_mat<-melt(cor_mat)
#head(cor_mat)
#
#ggplot(data = cor_mat, aes(x=Var1, y=Var2, fill=value)) +
#  geom_tile()


