library(dplyr)
library(ggplot2)
library(hrbrthemes)
library(patchwork)
library(aplot)
library(ggsci)


# Read in CSVs (prediction CSVs by the test mode of the model) of five replicates for the optimized model
pre_1 <- read.csv("pred_1.csv")
test_1 <- read.csv("test_1.csv")
dat_1 <- merge(pre_1, test_1, by = "file_path")
dat_1$true_pred <- dat_1$predicted_individual == dat_1$individual_id
acc1 <- length(which(dat_1$true_pred == TRUE))/length(dat_1$true_pred)

pre_2 <- read.csv("pred_2.csv")
test_2 <- read.csv("test_2.csv")
dat_2 <- merge(pre_2, test_2, by = "file_path")
dat_2$true_pred <- dat_2$predicted_individual == dat_2$individual_id
acc2 <- length(which(dat_2$true_pred == TRUE))/length(dat_2$true_pred)

pre_3 <- read.csv("pred_3.csv")
test_3 <- read.csv("test_3.csv")
dat_3 <- merge(pre_3, test_3, by = "file_path")
dat_3$true_pred <- dat_3$predicted_individual == dat_3$individual_id
acc3 <- length(which(dat_3$true_pred == TRUE))/length(dat_3$true_pred)

pre_4<- read.csv("pred_4.csv")
test_4 <- read.csv("test_4.csv")
dat_4 <- merge(pre_4, test_4, by = "file_path")
dat_4$true_pred <- dat_4$predicted_individual == dat_4$individual_id
acc4 <- length(which(dat_4$true_pred == TRUE))/length(dat_4$true_pred)

pre_5 <- read.csv("pred_5.csv")
test_5 <- read.csv("test_5.csv")
dat_5 <- merge(pre_5, test_5, by = "file_path")
dat_5$true_pred <- dat_5$predicted_individual == dat_5$individual_id
acc5 <- length(which(dat_5$true_pred == TRUE))/length(dat_5$true_pred)

# Mean and sd of prediction accuracies
acc <- c(acc1, acc2, acc3, acc4, acc5)
mean(acc)
sd(acc)

#Analysis of the best one
dat <- dat_1[,c(2,9,10)]
dat$sex <- sub(".*([FM]).*", "\\1", dat$individual_id)
dat_summ <- dat %>% group_by(individual_id) %>% summarise(correct_num = sum(as.integer(true_pred)), 
                                                          false_num = n() - sum(as.integer(true_pred)), 
                                                          samp_num = n())
dat_summ$sex <- sub(".*([FM]).*", "\\1", dat_summ$individual_id)

# GLM model to look at whether sex and sample_number have effect on prediction accuracy
model <- glm(cbind(correct_num, false_num) ~ samp_num + sex, family = "binomial", data = dat_summ)
summary(model)

#Hotplot + bars showing sex
dat <- dat %>%
  arrange(sex)

dat_summ <- dat_summ %>% arrange(sex)
dat_summ$individual_id <- factor(dat_summ$individual_id, levels = dat_summ$individual_id)


dat_mat <- dat %>%
  count(individual_id, predicted_individual) 


dat_mat$individual_id <- factor(dat_mat$individual_id, levels = dat_summ$individual_id)
dat_mat$predicted_individual <- factor(dat_mat$predicted_individual, levels = dat_summ$individual_id)

dat_mat$sex <- sub(".*([FM]).*", "\\1", dat_mat$individual_id)
dat_mat$sex[which(dat_mat$sex == "M")] <- 0
dat_mat$sex[which(dat_mat$sex == "F")] <- 1

p_main <- ggplot(dat_mat,aes(x = individual_id, y = predicted_individual, fill = n))+
  geom_tile()+
  xlab(NULL) +
  ylab(NULL) +
  labs(fill = "Count")+
  scale_x_discrete(expand=c(0,0))+
  scale_y_discrete(expand=c(0,0))+
  scale_fill_gradientn(colours =  c(
    "#F39B7FFF",
    "#DC0000FF",
    "darkred",
    "black"))+
  theme(
    axis.ticks.y = element_blank(),          
    axis.text.y  = element_blank(),  
    axis.title.y = element_blank(),
    axis.ticks.x = element_blank(),          
    axis.text.x  = element_blank(),  
    axis.title.x = element_blank()
  )

p_bottom <- ggplot(dat_summ, aes(x = individual_id, y = 1, fill = sex)) +
  geom_tile() +
  xlab("True ID")+
  scale_fill_npg()+
  theme(
    axis.ticks.x = element_blank(),          
    axis.text.x  = element_blank(),          
    axis.title.x = element_text(size = 10),
    axis.ticks.y = element_blank(),          
    axis.text.y  = element_blank(),  
    axis.title.y = element_blank(),
    legend.position = "none"
  )

p_left <- ggplot(dat_summ, aes(x = 1, y = individual_id, fill = sex)) +
  labs(fill = "Sex")+
  geom_tile() +
  ylab("Predicted ID")+
  scale_fill_npg()+
  theme(
    axis.ticks.y = element_blank(),          
    axis.text.y  = element_blank(),  
    panel.grid   = element_blank(),          
    axis.title.y = element_text(size = 10),
    axis.ticks.x = element_blank(),          
    axis.text.x  = element_blank(),  
    axis.title.x = element_blank()
  )


p_main %>%
  insert_bottom(p_bottom, height = .03)%>%
  insert_left(p_left, width = .03)






















