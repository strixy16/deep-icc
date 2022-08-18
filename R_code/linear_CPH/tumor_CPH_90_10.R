library("survival")
library("survminer")
library("survAUC")
library("readxl")
library("writexl")

full_TR_tumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/train_tumor_feats_and_labels.xlsx")
full_TE_tumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/test_tumor_feats_and_labels.xlsx")

TR_tumor <- subset(full_TR_tumor, select = -c(ScoutID, Cancer_Type))
TE_tumor <- subset(full_TE_tumor, select = -c(ScoutID, Cancer_Type))
# Set up CoxPH for tumor data
tumor.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ 
                     Kurtosis+MeanAbsoluteDeviation+
                     Minimum+RobustMeanAbsoluteDeviation+Skewness+Uniformity+Variance+
                     Complexity+Sphericity,
                   x=TRUE, y=TRUE, method="breslow", data=TR_tumor)
# Check proportional hazards
tumor.ph <- cox.zph(tumor.fit)
tumor.ph 

tumortrainpred <- predict(tumor.fit, newdata=TR_tumor)
tumortestpred <- predict(tumor.fit, newdata=TE_tumor)

summary(tumor.fit)
#GHCI(tumortrainpred)
#GHCI(tumortestpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(tumor.fit, TE_tumor), TE_tumor)

tumorSurv.rsp <- Surv(TR_tumor$HDFS_Time, TR_tumor$HDFS_Code)
tumorSurv.rsp.new <- Surv(TE_tumor$HDFS_Time, TE_tumor$HDFS_Code)
tumortrainCstat <- UnoC(tumorSurv.rsp, tumorSurv.rsp, tumortrainpred)
tumortestCstat <- UnoC(tumorSurv.rsp, tumorSurv.rsp.new, tumortestpred)

# Cancer specific testing
HCC_rows <- full_TE_tumor[full_TE_tumor$Cancer_Type == 0,]
ICC_rows <- full_TE_tumor[full_TE_tumor$Cancer_Type == 2,]
MCRC_rows <- full_TE_tumor[full_TE_tumor$Cancer_Type == 1,]

HCC_tumor <- subset(HCC_rows, select = -c(ScoutID, Cancer_Type))
ICC_tumor <- subset(ICC_rows, select = -c(ScoutID, Cancer_Type))
MCRC_tumor <- subset(MCRC_rows, select = -c(ScoutID, Cancer_Type))

tumorHCCpred <- predict(tumor.fit, newdata=HCC_tumor)
tumorHCC.rsp <- Surv(HCC_tumor$HDFS_Time, HCC_tumor$HDFS_Code)
tumorHCCCstat <- UnoC(tumorSurv.rsp, tumorHCC.rsp, tumorHCCpred)

tumorICCpred <- predict(tumor.fit, newdata=ICC_tumor)
tumorICC.rsp <- Surv(ICC_tumor$HDFS_Time, ICC_tumor$HDFS_Code)
tumorICCCstat <- UnoC(tumorSurv.rsp, tumorICC.rsp, tumorICCpred)

tumorMCRCpred <- predict(tumor.fit, newdata=MCRC_tumor)
tumorMCRC.rsp <- Surv(MCRC_tumor$HDFS_Time, MCRC_tumor$HDFS_Code)
tumorMCRCCstat <- UnoC(tumorSurv.rsp, tumorMCRC.rsp, tumorMCRCpred)