
library(RWeka)
library(Matrix)
library(quadprog)
library(kernlab)
library(matrixcalc)


# -----------------------------------------------------------------
# Input:
# source - filename of the source dataset
# target - filename of the target dataset
# p1 - ratio of source data samples used for transfer learning (0.5)
# p2 - ratio of target data samples used for transfer learning (0.5)
# -----------------------------------------------------------------
testCase <- function(source, p1, target, p2){
  D1 <- read.csv(source)
  D2 <- read.csv(target)
  output <- c()
  for(i in 1:10){
    ix <- sampleSize(D1[,ncol(D1)], p1)
    ds <- D1[ix,]
    ix <- sampleSize(D2[,ncol(D2)], p2)
    dt <- D2[ix,]
    test <- D2[-ix,]
    output <- rbind(output, transFall(ds, dt, test, 0.3, 0.001))
  }
  output <- as.data.frame(output)
  colnames(output) <- c("label_accuracy", 
                        "classification_base", "classification_tf",
                        "classification_upper")
  
  return(output)
}


# ---------------------------- TransFall --------------------------
# Input:
# ds, dt - source and target datasets WITH label in the last column
# test - a separate dataset on the target for test purpose
# sigma, lambda - parameters used in the framework
#
# Output:
# 1. Labeling accuracy of target data samples via transfer learning
# 2. Base performance of the classifier trained using source dataset
# 3. Performance of the classifier trained through TransFall
# 4. Upperbound performance obtained using labeled target dataset
# ------------------------------------------------------------------

transFall <- function(ds, dt, test, sigma, lambda){
  # Step 1. Vertical transformation
  vt <- ftDistChange(ds, dt)
  dtNew <- ftUpdate(dt, vt)
  
  # Step 2. Horizontal transformation
  L <- length(unique(ds[,ncol(ds)]))
  beta <- distMatch(ds, dt, sigma)
  
  # Step 3. Label Estimation
  alpha <- weightLMS(ds, beta, sigma, lambda)
  dat <- ds[beta > 0,-ncol(ds)]
  ys <- kernelLabel(dat, dt, alpha, sigma)
  y <- apply(ys, 1, function(x) which.max(x))
  ylabel <- as.numeric(y)
  
  # Labels obtained for target training dataset
  acc1 <- round(sum(ylabel == as.numeric(dt[,ncol(dt)]))/nrow(dt),4)
  
  # Step 4. Machine learning model develop
  test <- as.data.frame(test)
  train.base <- as.data.frame(ds)
  train.tf <- as.data.frame(cbind(dt[,-ncol(dt)], ylabel))
  train.upper <- as.data.frame(dt)
  
  colnames(test)[ncol(test)] <- "label"
  colnames(train.base)[ncol(train.base)] <- "label"
  train.base$label <- as.factor(train.base$label)
  colnames(train.tf)[ncol(train.tf)] <- "label"
  train.tf$label <- as.factor(train.tf$label)
  colnames(train.upper)[ncol(train.upper)] <- "label"
  train.upper$label <- as.factor(train.upper$label)
    
  mod.base <- Logistic(label ~., data = train.base)
  mod.tf <- Logistic(label ~., data = train.tf)
  mod.upper <- Logistic(label ~., data = train.upper)
  
  ymod <- as.numeric(predict(mod.base, test[,-ncol(test)]))
  ymod <- cbind(ymod, as.numeric(predict(mod.tf, test[,-ncol(test)])))
  ymod <- cbind(ymod, as.numeric(predict(mod.upper, test[,-ncol(test)])))
  
  acc2 <- apply(ymod, 2, 
                function(x) sum(x == as.numeric(test[,ncol(test)]))/nrow(test))
  acc2 <- round(acc2, 4)
  
  return(c(acc1, acc2))
}


# ------------------ Segmentation Method -----------------------
# Input:
# p - ratio of data samples desired for training purpose
# Output:
# inx - the index of data samples been selected for training
# --------------------------------------------------------------
sampleSize <- function(labels, p){
  U <- sort(unique(labels))
  inx <- c()
  for(i in U){
    t <- which(labels == i)
    if(length(t) > 1){
      if(length(t) * p < 1){
        inx <- c(inx, sample(t, 1))
      }
      else{
        inx <- c(inx, sample(t, floor(length(t) * p), replace = F))
      }
    }
  }
  return(inx)
}


# -----------------------------------------------------------------
# Input:
# ds, dt - source and target datasets WITH label in the last column
# Output:
# beta - sample weight factor for source dataset ds
# -----------------------------------------------------------------
distMatch <- function(ds, dt, sig){
  rbf <- rbfdot(sigma = sig)
  dat <- as.matrix(ds[,-ncol(ds)])
  K <- kernelMatrix(kernel = rbf, dat)
  if(!is.symmetric.matrix(K)){
    t <- nearPD(K)
    K <- t$mat
  }
  else if(!is.positive.definite(K)){
    t <- nearPD(K)
    K <- t$mat
  }
  r <- nrow(ds)/nrow(dt)
  ka <- kernelMult(rbf, x = dat, y = as.matrix(dt[,-ncol(dt)]), z = rep(r, nrow(dt)))
  B <- 2 * nrow(ds)
  
  A.lbs <- diag(1, nrow = nrow(ds), ncol = nrow(ds))
  b.lbs <- rep(0, nrow(ds))
  # A.ubs <- diag(-1, nrow = nrow(ds), ncol = nrow(ds))
  # b.ubs <- rep(-B, nrow(ds))
  A.ge1 <- rbind(rep(1, nrow(ds)))
  b.ge1 <- nrow(ds) - B * sqrt(nrow(ds))
  A.ge2 <- rbind(rep(-1, nrow(ds)))
  b.ge2 <- -B * sqrt(nrow(ds)) - nrow(ds)
  
  AM <- t(rbind(A.lbs, A.ge1, A.ge2))
  bv <- as.matrix(c(b.lbs, b.ge1, b.ge2))
  
  sol <- solve.QP(Dmat = K, dvec = ka, Amat = AM, bvec = bv)
  beta <- sol$solution
  if(is.na(sol$solution[1])){
    print("ERROR in BETA solve")
    beta <- sol$unconstrained.solution
  }
  if(min(beta) < 0){
    ws <- beta - min(beta)
  }
  else{
    ws <- beta
  }
  
  return(ws)
}



weightLMS <- function(ds, ws, sig, lam){
  rbf <- rbfdot(sigma = sig)
  if(sum(ws > 0) < length(unique(ncol(ds)))){
    print("too many 0 in weight vector")
  }
  dat <- as.matrix(ds[ws > 0,-ncol(ds)])
  K <- kernelMatrix(kernel = rbf, dat)
  betaD <- diag(ws[ws > 0], nrow = nrow(dat), ncol = nrow(dat))
  B <- solve(betaD, tol = 1e-20)
  A <- solve((B * lam + K), tol = 1e-20)
  
  Y <- ds[ws > 0, ncol(ds)]
  L <- unique(Y)
  alpha <- matrix(0, nrow = length(Y), ncol = length(L))
  
  for(i in 1:length(L)){
    yi <- Y
    yi[Y == L[i]] <- 1
    yi[Y != L[i]] <- 0
    alpha[,L[i]] <- A %*% as.matrix(yi)
  }
  
  return(alpha)
}



kernelLabel <- function(dat, dt, alpha, sig){
  rbf <- rbfdot(sigma = sig)
  ys <- matrix(0, nrow = nrow(dt), ncol = ncol(alpha))
  for(i in 1:ncol(alpha)){
    t <- kernelMult(rbf, x = as.matrix(dt[,-ncol(dt)]), y = as.matrix(dat), z = alpha[,i])
    ys[,i] <- t
  }
  return(ys)
} 


# ----------- Feature Distribution Change ------------
ftDistChange <- function(ds, dt){
  A <- matrix(0, nrow = (ncol(dt)-1), ncol = 2)
  for(i in 1:(ncol(dt)-1)){
    A[i,2] <- sqrt(sd(ds[,i])/sd(dt[,i]))
    A[i,1] <- mean(ds[,i]) - A[i,2] * mean(dt[,i])
  }
  A
}

ftUpdate <- function(dt, A){
  for(i in 1:(ncol(dt) - 1)){
    dt[,i] <- A[i,1] + A[i,2] * dt[,i]
  }
  return(dt)
}
