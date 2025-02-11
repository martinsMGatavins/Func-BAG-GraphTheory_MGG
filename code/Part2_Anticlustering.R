library(anticlust)
library(tidyr)
library(dplyr)
library(gtsummary)

data <- read.csv(file="data/phen_scanqa_n998_090524.csv")
df_main <- data[data$cut == FALSE,]
df_sens <- data[data$strict_cut == FALSE,]

filtering_cols <- c("age","pfact_score","cog_score")

# Train/test 0.66/0.33 split
df_main$cluster <- anticlustering(df_main[,filtering_cols],
                                  K=c(587,294),
                                  objective="variance",
                                  categories=df_main[,"gender"])
df_sens$cluster <- anticlustering(df_sens[,filtering_cols],
                                  K=c(539,270),
                                  objective="variance",
                                  categories=df_sens[,"gender"])

df_main %>% 
    select(-c(X,sub,cut,strict_cut,NumVolProc,Incl2BL_20220816,SUB)) %>%
    tbl_summary(by = cluster) %>%
    add_p()

df_sens %>% 
    select(-c(X,sub,cut,strict_cut,NumVolProc,Incl2BL_20220816,SUB)) %>%
    tbl_summary(by = cluster) %>%
    add_p()

df_main %>% select(sub,gender,age,MeanDisOrig,pfact_score,cog_score,cluster) %>%
    write.csv(file="data/phen_scanqa_main_n881.csv")
df_sens %>% select(sub,gender,age,MeanDisOrig,pfact_score,cog_score,cluster) %>%
    write.csv(file="data/phen_scanqa_sens_n809.csv")