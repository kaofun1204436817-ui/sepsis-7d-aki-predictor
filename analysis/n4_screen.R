# N4 筛选：扩展池上的 LASSO 与 Boruta（流程与 N2 完全一致，只换变量池）
suppressPackageStartupMessages({library(glmnet); library(Boruta); library(randomForest)})
set.seed(123)
OUT <- Sys.getenv("CN2_OUT", ".")
d <- read.csv(file.path(OUT, "n4_train.csv"), check.names = FALSE, fileEncoding = "UTF-8-BOM")
names(d)[1] <- sub("^\ufeff", "", names(d)[1])
FEATS <- readLines(file.path(OUT, "n4_feature_pool.txt"))
FEATS <- FEATS[nzchar(FEATS)]
miss <- setdiff(FEATS, names(d))
if (length(miss)) { cat("缺列:", paste(miss, collapse = ", "), "\n"); quit(status = 1) }
X <- as.matrix(d[, FEATS]); y <- d$y
cat("训练集:", nrow(X), "例；事件:", sum(y), "；候选变量:", length(FEATS), "\n")

cvfit <- cv.glmnet(X, y, family = "binomial", alpha = 1, standardize = TRUE,
                   type.measure = "auc", nfolds = 10, seed = 123)
co_1se <- as.matrix(coef(cvfit, s = "lambda.1se")); co_min <- as.matrix(coef(cvfit, s = "lambda.min"))
lasso <- data.frame(var = rownames(co_1se), coef_1se = as.numeric(co_1se), coef_min = as.numeric(co_min))
lasso$lasso_1se <- lasso$coef_1se != 0; lasso$lasso_min <- lasso$coef_min != 0
lasso <- lasso[lasso$var != "(Intercept)", ]; lasso <- lasso[order(-abs(lasso$coef_1se)), ]
write.csv(lasso, file.path(OUT, "n4_lasso.csv"), row.names = FALSE, fileEncoding = "UTF-8")
cat("LASSO λ.1se 选中:", sum(lasso$lasso_1se), "；λ.min 选中:", sum(lasso$lasso_min),
    "；CV 最优 AUC:", round(max(cvfit$cvm), 4), "；λ.1se AUC:", round(cvfit$cvm[which(cvfit$lambda == cvfit$lambda.1se)], 4), "\n")

png(file.path(OUT, "n4_lasso_cv.png"), width = 1600, height = 1300, res = 200)
plot(cvfit); title(main = "N4 扩展池 LASSO 10 折交叉验证（AUC）", line = 2.2); dev.off()

bd <- d[, c(FEATS, "y")]; bd$y <- factor(bd$y)
bor <- Boruta(y ~ ., data = bd, doTrace = 0, maxRuns = 100, ntree = 300)
bt <- attStats(bor); bt$var <- rownames(bt)
bo <- data.frame(var = bt$var, decision = bt$decision, medianImp = bt$medianImp,
                 meanImp = bt$meanImp, normHits = bt$normHits)
bo <- bo[order(-bo$medianImp), ]
write.csv(bo, file.path(OUT, "n4_boruta.csv"), row.names = FALSE, fileEncoding = "UTF-8")
cat("Boruta：Confirmed", sum(bo$decision == "Confirmed"), "／Tentative",
    sum(bo$decision == "Tentative"), "／Rejected", sum(bo$decision == "Rejected"), "\n")
cat("Confirmed:", paste(bo$var[bo$decision == "Confirmed"], collapse = ", "), "\n")
png(file.path(OUT, "n4_boruta.png"), width = 1900, height = 1500, res = 200)
par(mar = c(14, 4.5, 3, 1)); plot(bor, las = 2, cex.axis = 0.65, xlab = "")
title(main = "N4 扩展池 Boruta 特征重要性", line = 2.2); dev.off()
cat("SAVED: n4_lasso.csv / n4_boruta.csv / n4_lasso_cv.png / n4_boruta.png\n")
