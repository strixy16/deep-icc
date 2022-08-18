library("survival")
library("survminer")
library("survAUC")
library("readxl")
library("writexl")

cnn_tumor_train_preds <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Tumor/2022_05_24_1604_train/train_predictions.csv")
cnn_tumor_valid_preds <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Tumor/2022_05_24_1604_train/valid_predictions.csv")
cnn_tumor_test_preds <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Tumor/2022_05_24_1604_train/test_predictions.csv")

cnn_tumor_train_labels <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Tumor/HCC_MCRC_ICC_HDFS_90_10_train_tumors.csv")

HCC_tumor <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Tumor/2022_05_24_1604_train/tumor_HCC_test_predictions.csv")
ICC_tumor <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Tumor/2022_05_24_1604_train/tumor_ICC_test_predictions.csv")
MCRC_tumor <- read.csv("/Users/katyscott/Desktop/HDFS_Project/Data/Output/Tumor/2022_05_24_1604_train/tumor_MCRC_test_predictions.csv")

train_preds = as.vector(cnn_tumor_train_preds$Prediction)
valid_preds = as.vector(cnn_tumor_valid_preds$Prediction)
test_preds = as.vector(cnn_tumor_test_preds$Prediction)

tumorSurv.rsp <- Surv(cnn_tumor_train_preds$Time, cnn_tumor_train_preds$Event)
tumorSurv.rsp.train <- Surv(cnn_tumor_train_preds$Time, cnn_tumor_train_preds$Event)
tumorSurv.rsp.valid <- Surv(cnn_tumor_valid_preds$Time, cnn_tumor_valid_preds$Event) 
tumorSurv.rsp.test <- Surv(cnn_tumor_test_preds$Time, cnn_tumor_test_preds$Event)

tumortrainCstat <- UnoC(tumorSurv.rsp, tumorSurv.rsp.train, train_preds)
tumorvalidCstat <- UnoC(tumorSurv.rsp, tumorSurv.rsp.valid, valid_preds)
tumortestCstat <- UnoC(tumorSurv.rsp, tumorSurv.rsp.test, test_preds)

tumorHCC.rsp <- Surv(HCC_tumor$Time, HCC_tumor$Event)
tumorHCCCstat <- UnoC(tumorSurv.rsp, tumorHCC.rsp, HCC_tumor$Prediction)

tumorICC.rsp <- Surv(ICC_tumor$Time, ICC_tumor$Event)
tumorICCCstat <- UnoC(tumorSurv.rsp, tumorICC.rsp, ICC_tumor$Prediction)

tumorMCRC.rsp <- Surv(MCRC_tumor$Time, MCRC_tumor$Event)
tumorMCRCCstat <- UnoC(tumorSurv.rsp, tumorMCRC.rsp, MCRC_tumor$Prediction)
