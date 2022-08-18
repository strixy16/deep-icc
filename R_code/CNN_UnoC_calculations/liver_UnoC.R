library("survival")
library("survminer")
library("survAUC")
library("readxl")
library("writexl")

cnn_liver_train_preds <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Liver/2022_05_24_1955_train/train_predictions.csv")
cnn_liver_valid_preds <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Liver/2022_05_24_1955_train/valid_predictions.csv")
cnn_liver_test_preds <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Liver/2022_05_24_1955_train/test_predictions.csv")
cnn_liver_train_labels <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Liver/HCC_MCRC_ICC_HDFS_90_10_train_liver.csv")

HCC_liver <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Liver/2022_05_24_1955_train/liver_HCC_test_predictions.csv")
ICC_liver <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Liver/2022_05_24_1955_train/liver_ICC_test_predictions.csv")
MCRC_liver <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Liver/2022_05_24_1955_train/Liver_MCRC_test_predictions.csv")

train_preds = as.vector(cnn_liver_train_preds$Prediction)
valid_preds = as.vector(cnn_liver_valid_preds$Prediction)
test_preds = as.vector(cnn_liver_test_preds$Prediction)

liverSurv.rsp <- Surv(cnn_liver_train_labels$HDFS_Time, cnn_liver_train_labels$HDFS_Code)
liverSurv.rsp.train <- Surv(cnn_liver_train_preds$Time, cnn_liver_train_preds$Event)
liverSurv.rsp.valid <- Surv(cnn_liver_valid_preds$Time, cnn_liver_valid_preds$Event) 
liverSurv.rsp.test <- Surv(cnn_liver_test_preds$Time, cnn_liver_test_preds$Event)

livertrainCstat <- UnoC(liverSurv.rsp, liverSurv.rsp.train, train_preds)
livervalidCstat <- UnoC(liverSurv.rsp, liverSurv.rsp.valid, valid_preds)
livertestCstat <- UnoC(liverSurv.rsp, liverSurv.rsp.test, test_preds)


liverHCC.rsp <- Surv(HCC_liver$Time, HCC_liver$Event)
liverHCCCstat <- UnoC(liverSurv.rsp, liverHCC.rsp, HCC_liver$Prediction)

liverICC.rsp <- Surv(ICC_liver$Time, ICC_liver$Event)
liverICCCstat <- UnoC(liverSurv.rsp, liverICC.rsp, ICC_liver$Prediction)

liverMCRC.rsp <- Surv(MCRC_liver$Time, MCRC_liver$Event)
liverMCRCCstat <- UnoC(liverSurv.rsp, liverMCRC.rsp, MCRC_liver$Prediction)
