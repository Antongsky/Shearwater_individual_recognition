library(dplyr)
data <- read.csv("C:/Users/26739/Desktop/embeddings_v0.csv")
data$sex <- sub(".*([MF]).*$", "\\1", data$individual_id)
# Only select clips with length >= 2.6s
data_com <- data%>%
  filter(end >= 2.6)
# Have a look which individual has not enough samples
df <- data_com %>%
  group_by(individual_id)%>%
  summarise(n())

#For individuals with samples less than the lower threshold, delete the individual
#For individuals with samples more than the upper threshold, delete extra samples randomly to the upper limit
individuals <- unique(data_com$individual_id)
data_com$select = NA
for (i in individuals){
  sample_n <- length(which(data_com$individual_id == i))
  if (sample_n < 20 ){
    data_com$select[which(data_com$individual_id == i)] <- FALSE
  }
  else if (sample_n >= 20 && sample_n <= 24){
    data_com$select[which(data_com$individual_id == i)] <- TRUE
  }
  if (sample_n > 24){
    data_com$select[which(data_com$individual_id == i)] <- FALSE
    data_com$select[sample(which(data_com$individual_id == i), 24)] <- TRUE
  }
}

data_com <- data_com[-which(data_com$select == FALSE), -7]
data_com_sum <- data_com %>% 
  group_by(individual_id)%>%
  summarise(n = n())
data_com_sum$sex <- sub(".*([MF]).*$", "\\1", data_com_sum$individual_id)
length(which(data_com_sum$sex == "M"))
length(which(data_com_sum$sex == "F"))

write.csv(data_com, "embeddings_v5.csv")
