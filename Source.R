# Cristina Aguilera, Jesus Antonanzas
# GCED, AA1, Juliol 2019.

# Metriques de validacio utilitzades a tot el projecte:

# del total de persone que truco, retorna percentatje que dirán que si.
# pren com input True positive rate i true negative rate d'un model.
eficacia <- function (tpr, tnr) {
  no.real <- 0.8860833  # proporcions originals del data set
  yes.real <- 1 - no.real
  yes.total.trucades <- tpr*yes.real
  trucades.total <- yes.total.trucades + ((1-tnr)*no.real)
  return((yes.total.trucades/trucades.total))
}

# Given precision and recall, returns F1 score.
f1 <- function (a,b) { 2*(a*b)/(a+b) }

# L'utilitzarem a cada modelatge. Pren com input els valors reals de la variable 
# resposta i els valors predits per qualsevol model, en aquest ordre.
# un exemple d'utilització es:

# pred <- predict (mod, type="class")
# measures(bank[learn,]$deposit, pred)

measures <- function(real, pred) {
  t <- table(truth=real, predicted = pred)
  trueyes <- t[2,2]/sum(t[2,])               # percentatge de "yes" que ens prediu bé.
  trueno <- t[1,1]/sum(t[1,])               # percentatge de "no" que ens prediu bé.
  f1score <- f1(trueyes, t[2,2]/(sum(t[,2])))
  err <- (1-sum(diag(t))/sum(t))
  (c(err, trueyes, trueno, f1score))
}

# ----------------------- pre-proces
library(caret)

bank <- read.csv("bank-additional-full.csv", sep = ";")
dim(bank)

# Age
agecont <- bank$age
bank$age <- cut(bank$age, breaks = c(17, 25, 35, 45, 55, 65, 75, 95), include.lowest = T)
# Job
bank$job[bank$job == "unknown"] = NA
bank$job <- droplevels(bank$job)
# Marital
bank$marital[bank$marital == "unknown"] = NA
bank$marital <- droplevels(bank$marital)
# Education
bank$education[bank$education == "unknown"] = NA
bank$education <- droplevels(bank$education)
# Default
default <- bank$default
bank$default[bank$default == "unknown"] = NA
bank <- bank[,-which(colnames(bank) == "default")]
# Housing
bank$housing[bank$housing == "unknown"] = NA
bank$housing <- droplevels(bank$housing)
# Loan
bank$loan[bank$loan == "unknown"] = NA
bank$loan <- droplevels(bank$loan)
# Duration
duration <- bank$duration
bank <- bank[,-which(colnames(bank) == "duration")]
# Target
colnames(bank)[which(names(bank) == "y")] <- "deposit"

# Barrejar les files per evitar biaixos
set.seed (104)
bank = bank[sample.int(nrow(bank)),]

# És el procès que hem fet servir per crear tots els datasets.
# ----------------------- imputacio
library(missForest)

# dividim les dades en train i test (75% / 25%)
n <- nrow(bank)
set.seed(1234)
learn <- sample(1:n, round(0.75*n))

set.seed(500)
bank.train.imp <- missForest::missForest(bank[learn,], verbose = TRUE)
bank <- bank.train.imp$ximp

set.seed(500)
bank.test.imp <- missForest::missForest(bank[-learn,], verbose = TRUE)
bank.test <- bank.test.imp$ximp

# com dividim train en 2 particions: train i validacio.
set.seed(1234)
learn <- sample(1:nrow(bank), round(0.5*n))

# guardem el resultat
# save(bank.test, file = "bank.test.RData")
# save(bank, learn, file = "bank.train.va.RData")

# Un cop imputat el dataset, anem a aplicar técniques de desbalanceig. 
# Podem fer under o over-sampling (reduir la classe dominant o augmentar la 
# classe menys representativa). Primer probem a aplicar una tecnica coneguda 
# com SMOTE (synthetic minority over-sampling techinque):

# Hem de determinar, primer, quin percentatge de dades "yes" (les poc representades) s'ha de crear. Per aixo mirem el percentatge que tenim al dataset original:
nrow(bank[bank$deposit == "yes",])/nrow(bank)
nrow(bank[bank$deposit == "no",])/nrow(bank)

# Tenim un 11% de casos positius d'un total de 41188 files. 
# Si pujem el percentatge de positius massa alta estarem afegint, potser, massa biaix. 

# =============================== UNDERSAMPLING ================================
# load("bank.train.va.RData")

bank.under <- bank[learn,][bank[learn,]$deposit == "yes",] # Posem tots els yes
n <- nrow(bank[learn,][bank[learn,]$deposit == "no",])

# per fer undersampling amb diferents proporcions, hem de variar el numero de mostres del sample.
# en aquest cas és US 50 50
bank.learn.no <- bank[learn,][bank[learn,]$deposit == "no",][sample(1:n, nrow(bank.under)),]
bank.under <- rbind(bank.under, bank.learn.no)
# barrejem per evitar biaixos de nou:
set.seed (1234)
bank.under = bank.under[sample.int(nrow(bank.under)),]
# i els indexs de "learn" son
learn.under <- seq(1, nrow(bank.under))
# afegim el set de validacio al data set
bank.under <- rbind(bank.under, bank[-learn,])
# i sobreescribim els indexs de train i test per compatibilitat amb la resta del programa
learn <- learn.under
# finalment, utilitzarem aquest dataset:
bank <- bank.under

# Mirem que es quedin balancejades com voliem:
prop.table(table(bank[learn,]$deposit))
prop.table(table(bank[-learn,]$deposit))

# =============================== SMOTE ================================

library(DMwR)
# load("bank.train.va.RData")

bank.smoted <- DMwR::SMOTE(deposit ~., data = bank[learn,], perc.over = 100, perc.under = 22, k = 7, learner = NULL)

learn.smote <- seq(1, nrow(bank.smoted)) # indexs de train del nou set

bank.smoted <- rbind(bank.smoted, bank[-learn,])  # i unim els datasets

learn <- learn.smote   # indexos de train que utilitzem, cambiem el nom perque queda mes clar.

# Sobreescribim el dataset, amb el qual modelarem:

bank.original <- bank
bank <- bank.smoted

prop.table(table(bank[learn,]$deposit))
prop.table(table(bank[-learn,]$deposit))

# guardem els resultats per futur us. Guardem tant els indexs com el dataset
# save(bank, learn, file = "smote60.40.RData")

# ----------------------- Modelatge

# LDA QDA

#Repliquem els resultats pel millor model que hem obtingut amb el dataset corresponent 
# (60-40 undersampling).

# load("us60.40.RData")

# Carregem la llibreria per moder aplicar les funcions de lda i qda.

library(MASS)

# Fem servir LDA com a learner.
(my.lda <- lda (deposit ~ ., data = bank[learn,]))
summary(my.lda)

# A partir del summary, podem veure quines s?n les variables amb m?s influ?ncia 
# en el nostre dataset (aquelles que tinguin un coeficient de la funci? discriminant 
#majors en valor absolut).

# Calculem les mesures d'error de training, TPR i TNR.
mesures(bank[learn,]$deposit, predict(my.lda)$class)

# Realitzem la validaci? del model i calculem els errors de validaci?, TPR i TNR.
my.lda <- lda (deposit ~ ., data = bank[learn,])
(err.LDA <- mesures(bank[-learn,]$deposit,predict(my.lda, newdata=bank[-learn,])$class))

# Construim l'interval de confian?a del 95% al voltant de l'error de validaci?.
(ct <- table(bank[-learn,]$deposit, predict(my.lda, newdata=bank[-learn,])$class))
(pe.hat <- 1-sum(diag(prop.table(ct))))
dev <- sqrt(pe.hat*(1-pe.hat)/dim(bank[-learn,])[1])*1.967
sprintf("(%f,%f)", pe.hat-dev,pe.hat+dev)


# Ara fem servir QDA.
my.qda <- qda(deposit ~ ., data = bank[learn,])

# Calculem l'error i m?triques de training.
mesures(bank[learn,]$deposit,predict(my.qda)$class)

# Calculem l'error de validaci?.
my.qda <- qda(deposit ~ ., data = bank[learn,])
err.QDA <- mesures(bank[-learn,]$deposit, predict(my.qda, newdata=bank[-learn,])$class)

# Interval de confian?a del 95% al voltant de l'error de test
(ct <- table(bank[-learn,]$deposit, predict(my.qda, newdata=bank[-learn,])$class))
(pe.hat <- 1-sum(diag(prop.table(ct))))
dev <- sqrt(pe.hat*(1-pe.hat)/dim(bank[-learn,])[1])*1.967
sprintf("(%f,%f)", pe.hat-dev,pe.hat+dev)


# ----------------------- GLM

# Reproduim el proc?s del millor model aconseguit amb el seg?ent dataset.
# load("us80.20.RData")

# Ajustem un GLM per a les dades de training.
glmmod <- glm(deposit ~ ., data=bank[learn,], family=binomial(link=logit))
summary(glmmod)

# Apliquem la funci? step, que ens simplifiquem el model, eliminant les variables menys 
# importants progessivament fins que obt? el model amb menor AIC.
glm.AIC <- step (glmmod)
summary(glm.AIC)

# A partir del summary podem veure quin AIC ens assoleix el darrer model abans de la
# convergencia de l'algorisme i quines s?n les variables que fa servir, ?s a dir, que s?n
# m?s rellevants per la classificaci?.

# Creem una funci? que ens calculi els errors en els sets de training i validaci? segons els 
# valors predits pel model simplificat que acabem de trobar. El valor de "p" ens indica el 
# llindar que farem servir per classificar els dos grups (el valor per defecte 0.5 suposa 
# que els dos grups s'han de classificar amb igual probabilitat).
n <- nrow(bank)
nlearn <- length(learn)
ntest <- n - nlearn

bank.glm <- function (P=0.5)
{
  ## Classifiquem les prediccions de training
  glm.AICpred <- NULL
  glm.AICpred[glm.AIC$fitted.values<P] <- 0
  glm.AICpred[glm.AIC$fitted.values>=P] <- 1
  
  # Hem de considerar la target com a factor un altre cop
  glm.AICpred <- factor(glm.AICpred, labels=c("no","yes"))
  
  # Vector que retorna l'error i m?triques de training per una probabilitat concreta P
  errorsTR <- vector("numeric",length= 5)
  errorsTR <- mesures(bank[learn,]$deposit,glm.AICpred)
  errorsTR <- c(P,errorsTR)
  
  
  # Calculem la precisi? del model sobre validaci?
  gl1t <- predict(glm.AIC, newdata=bank[-learn,],type="response")
  gl1predt <- NULL
  gl1predt[gl1t<P] <- 0
  gl1predt[gl1t>=P] <- 1
  gl1predt <- factor(gl1predt, labels=c("no","yes"))
  
  # Vector que retorna l'error i m?triques de validaci? per una probabilitat concreta P
  errorsVA <- vector("numeric",length= 5)
  errorsVA <- mesures(bank[-learn,]$deposit,gl1predt)
  errorsVA <- c(P,errorsVA)
  
  # Retornem una matriu on la primera fila ?s l'error de training i la segona l'error de validaci?
  return(rbind(errorsTR,errorsVA))
}

# Creem dues matrius on ens guardarem les mesures de training i validaci? per cadascuna de
# probabilitats P com a threshold.
errorsTR <- matrix (nrow=0,ncol=5)
errorsVA <- matrix (nrow=0,ncol=5)
colnames(errorsTR) <- c("P", "err", "trueyes", "trueno", "f1score")
colnames(errorsVA) <- c("P", "err", "trueyes", "trueno", "f1score")

# Calculem quin ?s el millor hiperpar?metre.
# OBS: Cal ajustar l'interval de probabilitats en funci? del datatset que fem servir, 
# ja que amb algunes probabilitats pot ser que el model ens predigui totes les observacions 
# com de la misma classe y a l'hora de factoritzar dins de la funci? podem tenir problemes.
P <- seq(0.5,0.9,0.05)
for (i in P) {
  errors <- bank.glm(i)
  errorsTR <- rbind(errorsVA,errors[1,])
  errorsVA <- rbind(errorsVA,errors[2,])
}

# Representem visualment els errors de validaci?
plot(NULL,xlim = c(0,dim(errorsVA)[1]),ylim = c(0,1), xaxt='n', ylab="% of errors", xlab="probability threshold")
cl <- c("purple","red","cyan","black")
for (i in 2:5){
  lines(errorsVA[,i],col = cl[i-1],type = 'l')
}
legend("topleft", legend = colnames(errorsVA)[2:5], col=cl, pch=1)
axis(1, at=1:dim(errorsVA)[1], labels=P)


# ----------------------- KNN

# Reproduim el proc?s del millor model aconseguit amb el seg?ent dataset.
# load("us60.40.RData")

# Cas 1: variables sense estandaritzar.

# El primer problema que se'ns presenta a l'hora de realizar el knn ?s que no soporta variables categ?riques 
# (totes han de ser num?riques) perqu? en aquest m?tode es calculen internament dist?ncies euclidianes entre els 
# punts. Per tant, el primer pas ?s binaritzar les variables categ?riques.

# Farem servir la llibreria "mlr" on trobem la funci? "createDummyFeatures" que ens binaritza les variables que 
# nosaltres li passem en el par?metre "cols", que s?n les variables categ?riques que tenim. Utilitzarem el 
# "method":"reference" que ens crea (nlevels)-1 variables noves per cada variable amb nlevels categories. 
# No en crea les nlevels perqu? suposa que si les dues s?n 0, 0, la tercera est? completament determinada i 
# haur? de ser 1, per exemple.
library(mlr)
bank_dummies <- createDummyFeatures(bank, method = "reference", 
                                    cols = c("age","job","marital","education","month",
                                             "day_of_week","poutcome", "housing", "loan", "contact"))
head(bank_dummies)
dim(bank_dummies)

# Ara cal que traiem la variable target del conjunt de dades, ja les dades s?n el propi model.
bank_dummies.data <- subset (bank_dummies, select = -deposit)
dim(bank_dummies.data)

# Iterarem sobre k el modelatje de les dades de training per veure amb quants k ve?ns m?s propers aconseguim el menor 
# error de validaci?.
# Obs: se sol fer fins sqrt(N) per convenci?.

library(class)
N <- nrow(bank_dummies.data[learn,])

# Matriu on guardarem les m?triques del model per cada k
errorsVA <- matrix (nrow=0,ncol=5)
colnames(errorsVA) <- c("k", "err", "trueyes", "trueno", "f1score")

# Iterem sobre k i guardem els resultats (pot trigar)
set.seed(90)
for (i in seq(1,sqrt(N),3)) {
  knn.preds <- class::knn (bank_dummies.data[learn,], bank_dummies.data[-learn,], bank_dummies[learn,]$deposit, k = i)
  errorsVA <- rbind(errorsVA,c(i,mesures(bank_dummies[-learn,]$deposit,knn.preds)))
}
errorsVA

# Representem visualment en una gr?fica les mesures de validaci? obtingudes
neighbours <- seq(1,sqrt(N),3)
plot(NULL,xlim = c(0,length(neighbours)),ylim = c(0.1,1), xaxt='n', ylab="% of errors", xlab="probability threshold")
title("VALIDACI?")
cl <- c("green","red","blue","black")
for (i in 2:5){
  lines(errorsVA[1:nrow(errorsVA),i],col = cl[i-1],type = 'l',lwd=2)
}
legend(legend = colnames(errorsVA)[2:5], fill = cl, title = "Mesures", x = "right", y = "right", density = 50, bty = "n", bg = "gray96")
axis(1, at=1:length(neighbours), labels=neighbours)

# Cas 2: variables estandaritzades.

# Estandaritzem les variables cont?nues del dataset
bankstd <- bank
bankstd$campaign <- scale(bankstd$campaign)
bankstd$pdays <- scale(bankstd$pdays)
bankstd$previous <- scale(bankstd$previous)
bankstd$emp.var.rate <- scale(bankstd$emp.var.rate)
bankstd$cons.price.idx <- scale(bankstd$cons.price.idx)
bankstd$cons.conf.idx <- scale(bankstd$cons.conf.idx)
bankstd$euribor3m <- scale(bankstd$euribor3m)
bankstd$nr.employed <- scale(bankstd$nr.employed)

# Repetim el mateix procediment que abans
library(mlr)
bank_dummies_std <- createDummyFeatures(bankstd, method = "reference", 
                                        cols = c("age","job","marital","education","month",
                                                 "day_of_week","poutcome", "housing", "loan", "contact"))
head(bank_dummies_std)
dim(bank_dummies_std)

bank_dummies.data_std <- subset (bank_dummies_std, select = -deposit)
dim(bank_dummies.data_std)

library(class)
N <- nrow(bank_dummies.data_std[learn,])

errorsVA_std <- matrix (nrow=0,ncol=5)
colnames(errorsVA_std) <- c("k", "err", "trueyes", "trueno", "f1score")

set.seed(90)
for (i in seq(1,sqrt(N),3)) {
  knn.preds <- class::knn (bank_dummies.data_std[learn,], bank_dummies.data_std[-learn,], bank_dummies_std[learn,]$deposit, k = i)
  errorsVA_std <- rbind(errorsVA_std,c(i,mesures(bank_dummies_std[-learn,]$deposit,knn.preds)))
}
errorsVA_std

# ----------------------- Naive Bayes

# Reproduim el proc?s del millor model aconseguit amb el seg?ent dataset.
# load("us80.20.RData")

library(e1071)

# Ajustem les dades de training amb Naive Bayes
nbmod <- naiveBayes(deposit ~ ., data = bank[learn,]) 

# El fem predir pel conjunt de training
pred.tr.nb <- predict (nbmod, bank[learn,-which(colnames(bank) == "deposit")])
# El fem predtir pel conjunt de validaci?
pred.va.nb <- predict (nbmod, bank[-learn,-which(colnames(bank) == "deposit")]) # test

# Recollim les mesures error, TPR i TNR
mesures(bank[learn,]$deposit,pred.tr.nb)
mesures(bank[-learn,]$deposit,pred.va.nb)

# ----------------------- MLP

library(nnet)
library(caret)

# load("us60.40.RData") dona els millors resultats

# estandaritzem les dades per evitar convergencia prematura.
bankstd <- bank
bankstd$campaign <- scale(bankstd$campaign)
bankstd$pdays <- scale(bankstd$pdays)
bankstd$previous <- scale(bankstd$previous)
bankstd$emp.var.rate <- scale(bankstd$emp.var.rate)
bankstd$cons.price.idx <- scale(bankstd$cons.price.idx)
bankstd$cons.conf.idx <- scale(bankstd$cons.conf.idx)
bankstd$euribor3m <- scale(bankstd$euribor3m)
bankstd$nr.employed <- scale(bankstd$nr.employed)

# Busquem el paràmetre de regularització més adient per size = 9 fixa.
decays <- 10^seq(-3,0,by=0.6)
mlp.decays.10 <- matrix(nrow = 0, ncol = 5)
colnames(mlp.decays.10) <- c("trerror","teerror", "tpr", "tfr", "f1")
for (i in 1:length(decays)) {
  set.seed(123)
  mlp.mod <- nnet::nnet(deposit ~., data = bankstd, subset = learn, 
                        size=9, decay = decays[i], maxit=1500, MaxNWts = 2000)
  predte <- predict (mlp.mod, newdata = bankstd[-learn,], type="class")
  pred <- predict (mlp.mod, type="class")
  mlp.decays.10 <- rbind(mlp.decays.10, 
                         c(measures(bankstd[learn,]$deposit, pred)[1],
                           measures(bankstd[-learn,]$deposit, predte)))
}

# Veiem com el decaay 0.06309 és el més adient. Provem amb menys neurones
# pel decays que hem troba:
mlp.decays.10 <- matrix(nrow = 0, ncol = 5)
colnames(mlp.decays.10) <- c("trerror","teerror", "tpr", "tfr", "f1")
for (i in 1:length(decays)) {
  set.seed(1234)
  mlp.mod <- nnet::nnet(deposit ~., data = bankstd, subset = learn, 
                        size=7, decay = decays[i], maxit=1500, MaxNWts = 2000)
  predte <- predict (mlp.mod, newdata = bankstd[-learn,], type="class")
  pred <- predict (mlp.mod, type="class")
  mlp.decays.10 <- rbind(mlp.decays.10, 
                         c(measures(bankstd[learn,]$deposit, pred)[1],
                           measures(bankstd[-learn,]$deposit, predte)))
}

# veiem que amb 7 pesos dona millor encara que amb 9.
# Ara probem diferents seeds per size 7 i mateix parametre de regularitzacio.

set.seed(1)
seeds <- round(runif(10, min = 1, max = 1000))
err <- data.frame("trerror","teerror","tyes", "tno", "f1","seed", stringsAsFactors = F)
for (i in 1:10) {
  set.seed(seeds[i])
  mlp.mod <- nnet::nnet(deposit ~., data = bankstd, subset = learn, size=7, maxit=1500, decay=0.0631, MaxNWts = 2000)
  predte <- predict (mlp.mod, newdata = bankstd[-learn,], type="class")
  pred <- predict (mlp.mod, type="class")
  err[i,] <- c(measures(bankstd[learn,]$deposit, pred)[1],measures(bankstd[-learn,]$deposit, predte), seeds[i])
}
err

# D'on veiem que la configuració de 7 nodes a la capa oculta i decay 0.06309 és la millor.

# ----------------------- SVM

library(kernlab)

# load("smote70.30.RData") millor configuracio

bankstd <- bank
bankstd$campaign <- scale(bankstd$campaign)
bankstd$pdays <- scale(bankstd$pdays)
bankstd$previous <- scale(bankstd$previous)
bankstd$emp.var.rate <- scale(bankstd$emp.var.rate)
bankstd$cons.price.idx <- scale(bankstd$cons.price.idx)
bankstd$cons.conf.idx <- scale(bankstd$cons.conf.idx)
bankstd$euribor3m <- scale(bankstd$euribor3m)
bankstd$nr.employed <- scale(bankstd$nr.employed)

# string specifying kernel, list containing parameters of the kernel (or "automatic" for rbf) and vector of C's to try.
svm.func <- function(krn = "rbfdot", param = "automatic", C.vec = c(1)) {
  info <- matrix(nrow = length(C.vec), ncol = 4)
  colnames(info) <- c("testerr", "tpr", "tfr", "f1.score")
  # al nostre cas fixem les C que provem
  rownames(info) <- c("0.001", "0.01","0.1","1", "10", "100")
  for (i in 1:length(C.vec)) {
    svm.mod <- kernlab::ksvm(deposit~.,data=bankstd[learn,],kernel=krn,kpar=param,C = C.vec[i])
    pred <- predict(svm.mod, newdata=bankstd[-learn,])
    info[i,] <- measures(bankstd[-learn,]$deposit, pred)
  }
  (info)
}


# kernel radial.
# compte, triga bastant
load("smote60.40.RData")
svmerrs.smote60.40 <- svm.func(krn = "rbfdot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote60.40.less.RData")
svmerrs.smote60.40.less <- svm.func(krn = "rbfdot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote60.40.min.RData")
svmerrs.smote60.40.min <- svm.func(krn = "rbfdot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote70.30.less.RData")
svmerrs.smote70.30.less <- svm.func(krn = "rbfdot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote70.30.RData")
svmerrs.smote70.30 <- svm.func(krn = "rbfdot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("us60.40.RData")
svmerrs.us60.40 <- svm.func(krn = "rbfdot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("us70.30.RData")
svmerrs.us70.30 <- svm.func(krn = "rbfdot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))

# kernel lineal.
load("smote60.40.RData")
van.svmerrs.smote60.40 <- svm.func(krn = "vanilladot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote60.40.less.RData")
van.svmerrs.smote60.40.less <- svm.func(krn = "vanilladot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote60.40.min.RData")
van.svmerrs.smote60.40.min <- svm.func(krn = "vanilladot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote70.30.less.RData")
van.svmerrs.smote70.30.less <- svm.func(krn = "vanilladot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("smote70.30.RData")
van.svmerrs.smote70.30 <- svm.func(krn = "vanilladot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("us60.40.RData")
van.svmerrs.us60.40 <- svm.func(krn = "vanilladot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))
load("us70.30.RData")
van.svmerrs.us70.30 <- svm.func(krn = "vanilladot", param = "automatic", C.vec = c(0.001, 0.01, 0.1, 1, 10, 100))

# kernel polinomic de grau 2, offset fix a 1.
load("smote60.40.RData")
poly.svmerrs.smote60.40 <- svm.func(krn = "polydot", param = list(degree=2, offset = 1), C.vec = c(0.01, 0.1, 1,10, 100))
load("smote60.40.less.RData")
poly.svmerrs.smote60.40.less <- svm.func(krn = "polydot", param = list(degree=2, offset = 1), C.vec = c(0.01, 0.1, 1,10, 100))
load("smote60.40.min.RData")
poly.svmerrs.smote60.40.min <- svm.func(krn = "polydot", param = list(degree=2, offset = 1), C.vec = c(0.01, 0.1, 1,10, 100))
load("smote70.30.less.RData")
poly.svmerrs.smote70.30.less <- svm.func(krn = "polydot", param = list(degree=2, offset = 1), C.vec = c(0.01, 0.1, 1,10, 100))
load("us60.40.RData")
poly.svmerrs.us60.40 <- svm.func(krn = "polydot", param = list(degree=2, offset = 1), C.vec = c(0.01, 0.1, 1,10, 100))
load("us70.30.RData")
poly.svmerrs.us70.30 <- svm.func(krn = "polydot", param = list(degree=2, offset = 1), C.vec = c(0.01, 0.1, 1,10, 100))


# d'aquí hem apuntat els resultats i hem buscat el millor amb les mesures adients.

# ----------------------- Random Forest

library(kernlab)  
library(tree)
library(randomForest)

# Given precision and recall, returns F1 score.
f1 <- function (a,b) { 2*(a*b)/(a+b) }

# retorna Error de prediccio, TPR, TNR, F1 respectivament
measures <- function(real, pred) {
  t <- table(truth=real, predicted = pred)
  trueyes <- t[2,2]/sum(t[2,])               
  trueno <- t[1,1]/sum(t[1,])                
  f1score <- f1(trueyes, t[2,2]/(sum(t[,2])))
  err <- (1-sum(diag(t))/sum(t))
  (c(err, trueyes, trueno, f1score))
}


# A la pràctica, hem replicat això per cada set de dades manualment
# i hem anat anotant els resultats.
# S'obtindràn els resultats seleccionats al document pdf, però,
# amb aquest dataset: "us60.40.RData"


# Fem ara Random Forest amb 500 arbres:
rf1 <- randomForest (deposit ~ ., data=bank[learn,], ntree=500, proximity=FALSE)

rf1
pred.tree <- predict (rf1, bank[-learn,], type="class")

measures(bank[-learn,]$deposit, pred.tree)

# Un cop hem fet el pas anterior per tots els conjunts d'entrenament, 
# per als millors sets probem de canviar el numero de variables de cada split, 
# aquest cop amb 200 arbres per model:

(nvar <- seq(2, 15, 1))
# prepare the structure to store the partial results
rf.results <- matrix (nrow=0, ncol = 6)
colnames (rf.results) <- c("nvar", "OOB","teerror", "tpr", "tfr", "f1")

ii <- 1

for (nv in nvar)
{ 
  print(nv)
  set.seed(88)
  model.rf <- randomForest(deposit ~ ., data=bank[learn,], 
                           ntree=200, proximity=FALSE, mtry=nvar[ii])
  predte <- predict (model.rf, newdata = bank[-learn,], type="class")
  
  # get the OOB
  rf.results <- rbind(rf.results, c(nvar[ii],model.rf$err.rate[200,1],
                                    measures(bank[-learn,]$deposit, predte)))
  
  ii <- ii+1
}

rf.results

# Veiem que per US 60-40, el model amb 3 variables és el que diem que és el millor
# pel nostre propòsit











