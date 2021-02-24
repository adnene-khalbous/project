# Comparaison des 10 modèles prédictifs étudiées sur le jeu de données Numbers

# Nettoyage de l'espace de travail
rm(list=ls())

# Chargement des packages
library(MASS)
library(rpart) 
library(randomForest)
library(nnet)
library(e1071)

source("score.r")

nomfile = "numbers.txt"
data = read.table(nomfile, header=T,sep=" ",dec=".")
donnees = data[,2:257]
data$y <- as.factor(data$y)

modele_reduit_rf = x_53 + x_120 + x_136 + x_213 + x_38 + x_76 + x_152 + x_105 + x_121 + x_137 + x_115 + x_101 + x_197 + x_168 + x_89 + x_60 + x_180 + x_196 + x_23 + x_169 + x_73 + x_75 + x_88 + x_131 + x_22 + x_56 + x_204 + x_72 + x_92 + x_24
modele_reduit_Topt = x_105 + x_153 + x_76 + x_104 + x_89 + x_59 + x_152 + x_75 + x_120 + x_169 + x_154 + x_137 + x_73 + x_213 + x_121 + x_168 + x_119 + x_37 + x_88 + x_118 + x_24 + x_117 + x_57 + x_91 + x_212 + x_106 + x_196 + x_197 + x_38 + x_60 + x_136 + x_135 + x_22 + x_101 + x_23 + x_90 + x_74 + x_181 + x_133 + x_198 + x_25 + x_21 + x_102 + x_43 + x_116 + x_123 + x_36 + x_132 + x_122 + x_124 + x_9 + x_53 + x_108 + x_92 + x_8 + x_44 + x_56 + x_77 + x_107 + x_69 + x_87 + x_150 + x_72 + x_61 + x_28 + x_26



nb_boucles = 100
tab_recap_app <- matrix(1, nrow = 10, ncol = nb_boucles)
tab_recap_test <- matrix(1, nrow = 10, ncol = nb_boucles)

for(s in 1:nb_boucles)
{
# Semence du générateur
set.seed(s)
  
# Extraction des Echantillons
indice = sample(1:1000, 800, replace=FALSE)
dataApp = data[indice,]
dataTest = data[-indice,]
donneesApp = dataApp[,2:257]
donneesTest = dataTest[,2:257]
yapp = dataApp[,1]
yapp <- as.factor(yapp)
ytest = dataTest[,1]
ytest <- as.factor(ytest)

##############################
# construction des 10 modeles #
##############################

# Regression Polytomique

model.poly <- multinom(yapp~ . ,data=donneesApp, MaxNWts =10000 )
yapp.poly = predict(model.poly)
score.app.poly = score(yapp,yapp.poly)
tab_recap_app[1,s]=score.app.poly
ytest.poly = predict(model.poly,newdata = donneesTest)
score.test.poly = score(ytest, ytest.poly)
tab_recap_test[1,s]=score.test.poly


# Analyse Discriminante ( LDA )
model.lda = lda(donneesApp,yapp)
yapp.lda = predict(model.lda)$class
score.app.lda = score( yapp, yapp.lda )
tab_recap_app[2,s]=score.app.lda

ytest.lda = predict(model.lda, newdata=donneesTest)$class
score.test.lda = score(ytest, ytest.lda)
tab_recap_test[2,s]=score.test.lda



# Reseau de Neurone
model.nnet<-nnet(yapp ~.,data=donneesApp,skip=TRUE,size=2,decay = 5e-3, MaxNWts =20000)
yapp.nnet = predict(model.nnet, data = donneesApp, type = "class")
score.app.nnet = score(yapp , yapp.nnet)
tab_recap_app[3,s]=score.app.nnet

ytest.nnet = predict(model.nnet, newdata = donneesTest, type = "class")
score.test.nnet = score(ytest , ytest.nnet)
tab_recap_test[3,s]=score.test.nnet



# SVM Lineaire
model.svml <- svm(yapp ~ .,kernel="linear",type="C-classification",data=donneesApp)
#pred_prob <- predict(model.svm, decision.values = TRUE, probability = TRUE)
yapp.svml = predict(model.svml, type = "class")
score.app.svml = score(yapp , yapp.svml)
tab_recap_app[4,s]=score.app.svml

ytest.svml = predict(model.svml, type = "class", newdata = donneesTest)
score.test.svml = score(ytest , ytest.svml)
tab_recap_test[4,s]=score.test.svml


# SVM Non Lineaire
model.svmnl <- svm(yapp ~ .,type="C-classification",data=donneesApp)
#pred_prob <- predict(model.svm, decision.values = TRUE, probability = TRUE)
yapp.svmnl = predict(model.svmnl, type = "class")
score.app.svmnl = score(yapp , yapp.svmnl)
tab_recap_app[10,s]=score.app.svmnl

ytest.svmnl = predict(model.svmnl, type = "class", newdata = donneesTest)
score.test.svmnl = score(ytest , ytest.svmnl)
tab_recap_test[10,s]=score.test.svmnl


# Random Forest
Ntree = 1000 # nombre ntree 
model.rf = randomForest(yapp ~. , data=donneesApp,ntree = Ntree, importance=T)
yapp.rf = predict(model.rf, newdata = donneesApp)
ytest.rf = predict(model.rf, newdata = donneesTest)
score.app.rf = score(yapp , yapp.rf )
tab_recap_app[5,s]=score.app.rf
score.test.rf = score(ytest , ytest.rf )
tab_recap_test[5,s]=score.test.rf



# One VS All
set.seed(1234)
Z = rep(NA, nrow(data))
copie = data
napp = 800
ntest= 200
dataOVA = data
indiceOVA = sample(1:1000, napp, replace=FALSE)
solution = matrix(nrow=10, ncol=257) 

for ( i in 0:9)
{
  
  Z[copie$y==i] = 1
  Z[copie$y!=i] = 0
  dataOVA$y = Z
  
  dataAppOVA = dataOVA[indice,]
  dataTestOVA = dataOVA[-indice,]
  
  
  
  # Régression logistique
  resglm = glm(dataAppOVA$y ~., family = binomial, data=dataAppOVA)
  
  solution[i+1,]=resglm$coefficients
}


# Prediction avec dataApp
yAppOVA = copie$y[indice]
imagesAppOVA = copie[indice,] # On recupére les images dans apprentissage
imagesAppOVA[,1]=1 # 
imagesAppOVA=t(imagesAppOVA)
probaAppOVA = solution %*% imagesAppOVA
ypredAppOVA = matrix( nrow = napp , ncol =1 )
for (w in 1:napp)
{
  ypredAppOVA[w]=which.max(probaAppOVA[,w])-1
}
ypredAppOVA
score.app.ova = score(ypredAppOVA,yapp)
tab_recap_app[6,s]=score.app.ova

# Prediction avec dataTest
yTestOVA = copie$y[-indice]
imagesTestOVA = copie[-indice,]
imagesTestOVA[,1]=1
imagesTestOVA=t(imagesTestOVA)
probaTestOVA = solution %*% imagesTestOVA
ypredTestOVA = matrix( nrow = ntest , ncol =1 )
for (w in 1:ntest)
{
  ypredTestOVA[w]=which.max(probaTestOVA[,w])-1
}
score.test.ova = score(ypredTestOVA , ytest)
tab_recap_test[6,s]=score.test.ova






# Arbres Tmax et Topt

yCart = as.ordered(data$y)
yappCart = as.ordered(yapp)
ytestCart = as.ordered(ytest)
Tree.max = rpart(yappCart ~ ., cp=0, data = donneesApp)
Cp=Tree.max$cptable[which.min(Tree.max$cptable[,4]),1]
Tree.opt = prune(Tree.max, cp = Cp)

y.app.tmax = predict(Tree.max, type = "class")
y.test.tmax = predict(Tree.max, type = "class", newdata=donneesTest)

score.app.tmax = score(y.app.tmax,yapp)
tab_recap_app[7,s]=score.app.tmax
score.test.tmax = score(y.test.tmax, ytest)
tab_recap_test[7,s]=score.test.tmax

y.app.topt = predict(Tree.opt, type = "class", data=donneesApp)
y.test.topt = predict(Tree.opt, type = "class", newdata=donneesTest)
score.app.topt = score(y.app.topt,yapp)
tab_recap_app[8,s]=score.app.topt
score.test.topt = score(y.test.topt, ytest)
tab_recap_test[8,s]=score.test.topt

}


for(s in 1:nb_boucles)
{
  # Semence du générateur
  set.seed(s)
  
  # Extraction des Echantillons
  indice = sample(1:1000, 800, replace=FALSE)
  dataApp = data[indice,]
  dataTest = data[-indice,]
  donneesApp = dataApp[,2:257]
  donneesTest = dataTest[,2:257]
  yapp = dataApp[,1]
  yapp <- as.factor(yapp)
  ytest = dataTest[,1]
  ytest <- as.factor(ytest)
  
  # Regression Polytomique Reduit
  
  model.polyro <- multinom(yapp~ x_105 + x_153 + x_76 + x_104 + x_89 + x_59 + x_152 + x_75 + x_120 + x_169 + x_154 + x_137 + x_73 + x_213 + x_121 + x_168 + x_119 + x_37 + x_88 + x_118 + x_24 + x_117 + x_57 + x_91 + x_212 + x_106 + x_196 + x_197 + x_38 + x_60 + x_136 + x_135 + x_22 + x_101 + x_23 + x_90 + x_74 + x_181 + x_133 + x_198 + x_25 + x_21 + x_102 + x_43 + x_116 + x_123 + x_36 + x_132 + x_122 + x_124 + x_9 + x_53 + x_108 + x_92 + x_8 + x_44 + x_56 + x_77 + x_107 + x_69 + x_87 + x_150 + x_72 + x_61 + x_28 + x_26 ,data=donneesApp, MaxNWts =10000 )
  yapp.polyro = predict(model.polyro)
  score.app.polyro = score(yapp,yapp.polyro)
  tab_recap_app[9,s]=score.app.polyro
  ytest.polyro = predict(model.polyro,newdata = donneesTest)
  score.test.polyro = score(ytest, ytest.polyro)
  tab_recap_test[9,s]=score.test.polyro
  
  
}


moy.app.poly = mean( tab_recap_app[1,] )
moy.app.lda = mean( tab_recap_app[2,] )
moy.app.nnet = mean( tab_recap_app[3,] )
moy.app.svml = mean( tab_recap_app[4,] )
moy.app.rf = mean( tab_recap_app[5,] )
moy.app.ova = mean( tab_recap_app[6,] )
moy.app.tmax = mean( tab_recap_app[7,] )
moy.app.topt = mean( tab_recap_app[8,] )
moy.app.polyro = mean( tab_recap_app[9,] )
moy.app.svmnl = mean( tab_recap_app[10,] )

SC = NULL
SC = cbind(SC, moy.app.poly)
SC = cbind(SC, moy.app.lda)
SC = cbind(SC, moy.app.nnet)
SC = cbind(SC, moy.app.svml)
SC = cbind(SC, moy.app.rf)
SC = cbind(SC, moy.app.ova)
SC = cbind(SC, moy.app.tmax)
SC = cbind(SC, moy.app.topt)
SC = cbind(SC, moy.app.polyro)
SC = cbind(SC, moy.app.svmnl)
colnames(SC) = c("Reg Poly", "LDA", "Reseau Neurone", "SVM Lin", "Random Forest", "OneVsAll","Tmax" ,"Topt","Reg Poly Red Opt", "SVM Non Lin")
cat("Performances sur l'échantillon d'Apprentissage \n")
print(SC)
cat("\n")

moy.test.poly = mean( tab_recap_test[1,] )
moy.test.lda = mean( tab_recap_test[2,] )
moy.test.nnet = mean( tab_recap_test[3,] )
moy.test.svml = mean( tab_recap_test[4,] )
moy.test.rf = mean( tab_recap_test[5,] )
moy.test.ova = mean( tab_recap_test[6,] )
moy.test.tmax = mean( tab_recap_test[7,] )
moy.test.topt = mean( tab_recap_test[8,] )
moy.test.polyro = mean( tab_recap_test[9,] )
moy.test.svmnl = mean( tab_recap_test[10,] )

SC = NULL

SC = cbind(SC, moy.test.poly)
SC = cbind(SC, moy.test.lda)
SC = cbind(SC, moy.test.nnet)
SC = cbind(SC, moy.test.svml)
SC = cbind(SC, moy.test.rf)
SC = cbind(SC, moy.test.ova)
SC = cbind(SC, moy.test.tmax)
SC = cbind(SC, moy.test.topt)
SC = cbind(SC, moy.test.polyro)
SC = cbind(SC, moy.test.svmnl)
colnames(SC) = c("Reg Poly", "LDA", "Reseau Neurone", "SVM Lin", "Random Forest", "OneVsAll","Tmax" ,"Topt","Reg Poly Reduit", "SVM Non Lin")
cat("Performances sur l'echantillon Test \n")
print(SC)
cat("\n")

