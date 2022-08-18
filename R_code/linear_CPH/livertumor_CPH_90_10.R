library("survival")
library("survminer")
library("survAUC")
library("readxl")
library("writexl")

full_TR_livertumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/train_liver_tumor_feats_and_labels.xlsx")
full_TE_livertumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/test_liver_tumor_feats_and_labels.xlsx")

TR_livertumor <- subset(full_TR_livertumor, select = -c(ScoutID, Cancer_Type))
TE_livertumor <- subset(full_TE_livertumor, select = -c(ScoutID, Cancer_Type))

# Set up CoxPH for liver and tumor data
livertumor.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ Range+Variance+
                          Busyness+Complexity+Contrast+Strength+Elongation+
                          Maximum2DDiameterRow+MinorAxisLength,
                        x=TRUE, y=TRUE, method="breslow", data=TR_livertumor)
livertumor.ph <- cox.zph(livertumor.fit)
livertumor.ph

livertumortrainpred <- predict(livertumor.fit, newdata=TR_livertumor)
livertumortestpred <- predict(livertumor.fit, newdata=TE_livertumor)

summary(livertumor.fit)
#GHCI(livertumortrainpred)
#GHCI(livertumortestpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(livertumor.fit, TE_livertumor), TE_livertumor)

livertumorSurv.rsp <- Surv(TR_livertumor$HDFS_Time, TR_livertumor$HDFS_Code)
livertumorSurv.rsp.new <- Surv(TE_livertumor$HDFS_Time, TE_livertumor$HDFS_Code)
livertumortrainCstat <- UnoC(livertumorSurv.rsp, livertumorSurv.rsp, livertumortrainpred)
livertumortestCstat <- UnoC(livertumorSurv.rsp, livertumorSurv.rsp.new, livertumortestpred)

# Cancer specific testing
HCC_rows <- full_TE_livertumor[full_TE_livertumor$Cancer_Type == 0,]
ICC_rows <- full_TE_livertumor[full_TE_livertumor$Cancer_Type == 2,]
MCRC_rows <- full_TE_livertumor[full_TE_livertumor$Cancer_Type == 1,]

HCC_livertumor<- subset(HCC_rows, select = -c(ScoutID, Cancer_Type))
ICC_livertumor<- subset(ICC_rows, select = -c(ScoutID, Cancer_Type))
MCRC_livertumor<- subset(MCRC_rows, select = -c(ScoutID, Cancer_Type))

livertumorHCCpred <- predict(livertumor.fit, newdata=HCC_livertumor)
livertumorHCC.rsp <- Surv(HCC_livertumor$HDFS_Time, HCC_livertumor$HDFS_Code)
livertumorHCCCstat <- UnoC(livertumorSurv.rsp, livertumorHCC.rsp, livertumorHCCpred)

livertumorICCpred <- predict(livertumor.fit, newdata=ICC_livertumor)
livertumorICC.rsp <- Surv(ICC_livertumor$HDFS_Time, ICC_livertumor$HDFS_Code)
livertumorICCCstat <- UnoC(livertumorSurv.rsp, livertumorICC.rsp, livertumorICCpred)

livertumorMCRCpred <- predict(livertumor.fit, newdata=MCRC_livertumor)
livertumorMCRC.rsp <- Surv(MCRC_livertumor$HDFS_Time, MCRC_livertumor$HDFS_Code)
livertumorMCRCCstat <- UnoC(livertumorSurv.rsp, livertumorMCRC.rsp, livertumorMCRCpred)