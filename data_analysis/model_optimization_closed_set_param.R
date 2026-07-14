library(dplyr)
library(ggplot2)
library(ggsci)
library(ggpubr)
library(rstatix)
library(cowplot)
setwd("")

theme_nb <- function(
    base_size = 9.5,
    base_family = ""
) {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Overall text
      text = element_text(family = base_family),
      
      # Panel border/axes: classic keeps panel background clean; we add axis lines explicitly
      axis.line = element_line(linewidth = 0.9, colour = "black"),
      
      # Tick marks
      axis.ticks = element_line(linewidth = 0.8, colour = "black"),
      axis.ticks.length = unit(2.0, "mm"),
      
      # Axis text (tick labels)
      axis.text = element_text(
        size = base_size,
        colour = "black",
        family = base_family
      ),
      
      # Axis titles (label size)
      axis.title = element_text(
        size = base_size * 1.3,
        face = "plain",
        colour = "black",
        family = base_family
      ),
      
      # Remove extra grid lines for a clean “publishing” look
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      
      # Margins (often helps multi-panel figures)
      plot.margin = margin(6, 10, 6, 10)
    )
}

# SupCon temperature-------------------------------------
files <- list.files("C:/Users/26739/Desktop/a", 
                    pattern = "\\.csv$", full.names = TRUE)
names(files) <- basename(files)
df_list <- lapply(files, read.csv)

for (i in names(df_list)){
  
  df_list[[i]]$t <- sub("^t([^_]+)_.*$", "\\1", i)               
  df_list[[i]]$rep <- sub("^.*_([^_]+)\\.csv$", "\\1", i)
}
data <- do.call(rbind, df_list)


# Close-set prediction changing with time
a <- ggplot(data, aes(epoch, val_acc, colour = factor(t)))+
  geom_line(aes(group = t),size = 0.8, alpha = 0.7)+
  labs(x = "Epoch", y = "Prediction Accuracy", colour = "SupCon Temp.")+
  scale_color_npg()+
  theme_nature()

b <- ggplot(data, aes(epoch, supcon_loss, colour = factor(t)))+
  geom_line(aes(group = t),size = 0.8, alpha = 0.7)+
  labs(x = "Epoch", y = "SupCon Loss", colour = "SupCon Temp.")+
  scale_color_npg()+
  theme_nature()

plot_grid(a, b, labels = "AUTO", ncol = 2)

data_p <- data %>% filter(epoch >= 170)
data_summ <- data_p %>% 
  group_by(t, rep) %>%
  summarise(mean_va = mean(val_acc))%>%
  ungroup()

# Close-set prediction changes with supCon temperature

kruskal_test(mean_va ~ t, data =data_summ)

dunn_res <- data_summ %>%
  dunn_test(mean_va ~ t, p.adjust.method = "BH")%>%
  filter(p.adj.signif != "ns")

dunn_res$y.position <- c(0.90, 0.91, 0.92, 0.93, 0.94) 



p1 <-ggplot(data_summ, aes(t, mean_va))+
  geom_boxplot(linewidth = 0.1)+
  geom_point()+
  labs(x = "SupCon Temp. (τ)", y = "Prediction Accuracy")+
  stat_pvalue_manual(
    dunn_res,
    label = "p.adj.signif",
    tip.length = 0.01
  ) +
  stat_compare_means(data = data_summ,aes(t, mean_va),
                     method = "kruskal.test",
                     label = "p.format",
                     label.y = 0.95,size = 4)+
  ylim(0.75,0.95)+
  scale_color_npg()+
  theme_nb()


# noise sd-------------------------------------
files <- list.files("C:/Users/26739/Desktop/新建文件夹 (16)/benchmark_close_set/noise", 
                    pattern = "\\.csv$", full.names = TRUE)
names(files) <- basename(files)
df_list <- lapply(files, read.csv)

for (i in names(df_list)){
  
  df_list[[i]]$noise <- sub("^.*noise", "", sub("_rep\\d+\\.csv$", "", i))             
  df_list[[i]]$rep <- sub("^.*_rep", "", sub("\\.csv$", "", i))
}
data <- do.call(rbind, df_list)


# Close-set prediction changing with time
ggplot(data, aes(epoch, val_acc, colour = factor(noise)))+
  geom_line(aes(group = noise),size = 0.8, alpha = 0.7)+
  ylim(0.5,0.9)+
  labs(x = "Epoch", y = "Prediction Accuracy", colour = "SupCon Temp.")+
  scale_color_npg()+
  theme_nature()

data_p <- data %>% filter(epoch >= 170)
data_summ <- data_p %>% 
  group_by(noise, rep) %>%
  summarise(mean_va = mean(val_acc))%>%
  ungroup()

# Close-set prediction changes with supCon temperature

kruskal_test(mean_va ~ noise, data =data_summ)


p2 <- ggplot(data_summ, aes(noise, mean_va))+
  geom_boxplot(linewidth = 0.1)+
  geom_point()+
  labs(x = "Noise Std. (σ)", y = "Prediction Accuracy")+
  stat_compare_means(data = data_summ,aes(noise, mean_va),
                     method = "kruskal.test",
                     label = "p.format",
                     label.y = 0.95,size = 4)+
  ylim(0.75,0.95)+
  scale_color_npg()+
  theme_nb()



#dropout ----------------------------------------
files <- list.files("C:/Users/26739/Desktop/新建文件夹 (16)/benchmark_close_set/dropout", 
                    pattern = "\\.csv$", full.names = TRUE)
names(files) <- basename(files)
df_list <- lapply(files, read.csv)

for (i in names(df_list)){
  
  df_list[[i]]$dropout <- sub("^.*dropout", "", sub("_rep\\d+\\.csv$", "", i))             
  df_list[[i]]$rep <- sub("^.*_rep", "", sub("\\.csv$", "", i))
}
data <- do.call(rbind, df_list)

ggplot(data, aes(epoch, val_acc, colour = factor(dropout)))+
  geom_line(aes(group = dropout),size = 0.8, alpha = 0.7)+
  ylim(0.5,0.9)+
  labs(x = "Epoch", y = "Prediction Accuracy", colour = "SupCon Temp.")+
  scale_color_npg()+
  theme_nb()

data_p <- data %>% filter(epoch >= 170)
data_summ <- data_p %>% 
  group_by(dropout, rep) %>%
  summarise(mean_va = mean(val_acc))%>%
  ungroup()

# Close-set prediction changes with supCon temperature

kruskal_test(mean_va ~ dropout, data =data_summ)

dunn_res <- data_summ %>%
  dunn_test(mean_va ~ dropout, p.adjust.method = "BH") %>%
  filter(p.adj.signif != "ns")

dunn_res$y.position <- c(0.90, 0.91, 0.92) 


p3 <- ggplot(data_summ, aes(dropout, mean_va))+
  geom_boxplot(linewidth = 0.1)+
  geom_point()+
  labs(x = "Dropout Ratio", y = "Prediction Accuracy")+
  ylim(0.75,0.95)+
  stat_pvalue_manual(
    dunn_res,
    label = "p.adj.signif",
    tip.length = 0.01
  ) +
  stat_compare_means(data = data_summ,aes(dropout, mean_va),
                     method = "kruskal.test",
                     label = "p.format",
                     label.y = 0.95,size = 4)+
  scale_color_npg()+
  theme_nb()



#Plot (Supplemental 2)--------------------------------------
plot_grid(p1, p2, p3, labels = "AUTO", ncol = 3)