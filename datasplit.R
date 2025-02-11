invisible(lapply(c("stringi","stringr","tidyr","dplyr","tidyverse","anticlust"), require, character.only=TRUE))
ROOT <- "~/Desktop/BIDS_WUSTL_2022/second_set"
TEMP <- paste0(ROOT,"/train_test_sets")
graphCSV <- read.csv(paste0(ROOT,"/data/graph25perc-fcon_888x2334_20220721.csv"))
graphCSV <- graphCSV %>% filter(strength1 != 0) %>% rename(sub = sub_method) # removes subjects with disconnected graphs
rehoCSV <- read.csv(paste0(ROOT,"/data/func_reho-1644x333_20220718.csv"))
alffCSV <- read.csv(paste0(ROOT,"/data/func_alff-1635x333_20220718.csv"))
masterCSV <- read.csv(paste0(ROOT,"/data/master_889x20_20220720.csv"))
masterCSV <- masterCSV %>% rename(sex = gender) %>% 
  filter(sub %in% graphCSV$sub)
demographicsCSV <- masterCSV %>% select(c("sub","age","sex"))

# Variance picked as objective because the distribution
anticluster <- anticlustering(demographicsCSV[,2:3],
                              K=c(295,591), # test_size=0.33, train_size=0.66
                              objective="variance",
                              categories=demographicsCSV$sex)

demographicsCSV_test <- demographicsCSV[anticluster==1,]
demographicsCSV_train <- demographicsCSV[anticluster==2,]

## ADD VERIFICATION STEPS - check the means and averages
summary(demographicsCSV_test[,2])
summary(demographicsCSV_train[,2])

# Age to merge
age_test <- demographicsCSV_test %>% select(c("sub","age"))
age_train <- demographicsCSV_train %>% select(c("sub","age"))

# Split by anticluster
graphCSV_test <- graphCSV %>% filter(sub %in% demographicsCSV_test$sub) %>%
  merge(age_test,by="sub")
graphCSV_train <- graphCSV %>% filter(sub %in% demographicsCSV_train$sub) %>%
  merge(age_train,by="sub")
reho_test <- rehoCSV %>% filter(sub %in% demographicsCSV_test$sub) %>%
  merge(age_test,by="sub")
reho_train <- rehoCSV %>% filter(sub %in% demographicsCSV_train$sub) %>%
  merge(age_train,by="sub")
alff_test <- alffCSV %>% filter(sub %in% demographicsCSV_test$sub) %>%
  merge(age_test,by="sub")
alff_train <- alffCSV %>% filter(sub %in% demographicsCSV_train$sub) %>%
  merge(age_train,by="sub")

# Within-category measures
modul_test <- graphCSV_test %>% select(c("sub","age") | starts_with("clustering") | starts_with("part_coef"))
modul_train <- graphCSV_train %>% select(c("sub","age") | starts_with("clustering") | starts_with("part_coef"))
centr_test <- graphCSV_test %>% select(c("sub","age") | starts_with("bcentr") | starts_with("str"))
centr_train <- graphCSV_train %>% select(c("sub","age") | starts_with("bcentr") | starts_with("str"))

# Nodal efficiency
eff_test <- graphCSV_test %>% select(c("sub","age") | starts_with("nodal_e"))
eff_train <- graphCSV_train %>% select(c("sub","age") | starts_with("nodal_e"))

# Modularity measures: individually
clust_test <- modul_test %>% select(c("sub","age") | starts_with("clustering"))
clust_train <- modul_train %>% select(c("sub","age") | starts_with("clustering"))
part_test <- modul_test %>% select(c("sub","age") | starts_with("part"))
part_train <- modul_train %>% select(c("sub","age") | starts_with("part"))

# Centrality measures: individually
bc_test <- centr_test %>% select(c("sub","age") | starts_with("bcentr"))
bc_train <- centr_train %>% select(c("sub","age") | starts_with("bcentr"))
str_test <- centr_test %>% select(c("sub","age") | starts_with("str"))
str_train <- centr_train %>% select(c("sub","age") | starts_with("str"))

# Functional connectivity
intrafc_test <- graphCSV_test %>% select(c("sub","age") | starts_with("intra"))
intrafc_train <- graphCSV_train %>% select(c("sub","age") | starts_with("intra"))
interfc_test <- graphCSV_test %>% select(c("sub","age") | starts_with("inter")) 
interfc_train <- graphCSV_train %>% select(c("sub","age") | starts_with("inter"))

rm("age_test","age_train","demographicsCSV_test","demographicsCSV_train")
dfs_test <- mget(ls(pattern="_test"))
dfs_train <- mget(ls(pattern="_train"))

lapply(1:length(dfs_test), function(i) write.csv(dfs_test[[i]], 
                                                file = paste0(TEMP,"/",names(dfs_test[i]), ".csv"),
                                                row.names=FALSE))
lapply(1:length(dfs_train), function(i) write.csv(dfs_train[[i]], 
                                                 file = paste0(TEMP,"/",names(dfs_train[i]), ".csv"),
                                                 row.names=FALSE))
                                                 
                                                 
