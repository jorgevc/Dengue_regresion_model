import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import t
import datos as dpl
import math
from scipy import stats 
#import pandas as pd 
import csv
import sys
#from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import curve_fit
import scipy
data = dpl.data("DATA")
valor = []
valor1 = []
valor2 = []
residuos = []
proba_incidencia= []
pr=[]
gen_residuos = []
dis_residuos = []

#---------------generacion-disposicion de residuos solidos=total de residuos
for i in range(len(data.gen_residuos)):
	residuos.append([data.gen_residuos[i][0],float(data.gen_residuos[i][1])- float(data.dis_residuos[i][1])])
tresiduos = np.array(residuos).T[1]
#--------------generacion de residuos
for i in range(len(data.gen_residuos)):
	gen_residuos.append([data.gen_residuos[i][0],float(data.gen_residuos[i][1])])
gresiduos = np.array(gen_residuos).T[1]
#--------------disposicion de residuos
for i in range(len(data.dis_residuos)):
	dis_residuos.append([data.dis_residuos[i][0],float(data.dis_residuos[i][1])])
dresiduos = np.array(dis_residuos).T[1]
#--------------Probabilidad de incidencia de dengue por entidad
for i in range(len(data.cases)):
	proba_incidencia.append([data.cases[i][0],float(data.cases[i][53])/ float(data.poblacion[i][1])])
p_incidencia = np.array(proba_incidencia).T[1]
# --------------Total de residuos solidos
fig1, ax1 = plt.subplots()
x = np.array(range(len(tresiduos)))
ax1.plot(x, tresiduos, 'o')
ax1.set_xlabel('Estados')
ax1.set_ylabel('residuos (miles de toneladas)',)
fig1.suptitle('Residuos solidos', fontsize=20)
#plt.show()
#----DATOS FILTRADOS------
Y =10000*( p_incidencia.astype(np.float))
X2a = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X2a.append(gresiduos.astype(np.float)[i])
        y_incidencia.append(Y[i])
#--------------Probabilidad de incidencia vs generacion de residuos
fig2a, ax2a = plt.subplots()
ax2a.plot(X2a, y_incidencia, 'go')
ax2a.set_ylabel('# de Casos por cada 10000 personas')
ax2a.set_xlabel('GRS (miles de toneladas)')
#fig2a.suptitle('Generacion de residuos S.  vs PID', fontsize=18)
ax2a.axis([ 0, 2000,0, 24])
#plt.show()  
#------------correlacion
print'correlacion'
corr2a= np.corrcoef(np.array(X2a), np.array(y_incidencia))
print corr2a
#----------intervalo de confianza Generacion de RS sin filtrar
n= len(X2a)
print's_xx'
s_xx=n*(np.var(X2a)) 
print s_xx
print 's_yy'
s_yy=n*(np.var(y_incidencia)) # probabilidad de incidencia
print s_yy
m=np.mean(X2a)
m1=np.mean(y_incidencia)
s=[]
for i in range(len(X2a)):
	s.append((X2a[i]- m)*(y_incidencia[i]-m1))
s_xy=np.sum(s)
print's_xy'
print s_xy
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
r=((s_xy)/(math.sqrt(s_xx*s_yy)))
print r
print'estadistico t'
t=((r*(math.sqrt(n-2)))/(math.sqrt(1-(pow(r,2)))))
print t
print' Normal'
#z=(math.lg((1+r)/(1-r)))/(2*(math.sqrt(1/(n-3))))
#print z
#----DATOS FILTRADOS------
X2d = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X2d.append(dresiduos.astype(np.float)[i])
        y_incidencia.append(Y[i])
#---------------Probabilidad de incidencia vs disposicion de residuos
fig2b, ax2b = plt.subplots()
ax2b.plot(X2d, y_incidencia, 'bo')
ax2b.set_ylabel('# de Casos por cada 10000 personas')
ax2b.set_xlabel('DRS (miles de toneladas)')
#fig2b.suptitle('Disposicion de residuos S. vs PID', fontsize=18)
ax2b.axis([ 0, 2000,0, 25])
#plt.show()
#------------correlacion 

corr2d= np.corrcoef(np.array(X2d), np.array(y_incidencia))
print'correlacion'
print corr2d
#----------intervalo de confianza disposicion de RS
print's_xx'
s_xxd=n*(np.var(X2d)) 
print s_xxd
print 's_xy'


md=np.mean(X2d)
sd=[]
for i in range(len(X2a)):
	sd.append((X2d[i]- md)*(y_incidencia[i]-m1))
s_xyd=np.sum(sd)
print s_xy
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rd=((s_xyd)/(math.sqrt(s_xxd*s_yy)))
print rd
print'estadistico t'
td=((rd*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rd,2)))))
print td
print' Normal'
#zd=(math.lg((1+rd)/(1-rd)))/(2*(math.sqrt(1/(n-3))))
#print zd

#----DATOS FILTRADOS---------
X2 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X2.append(tresiduos.astype(np.float)[i])
        y_incidencia.append(Y[i])

#--------------Probabilidad de incidencia vs total de residuos solidos
fig2, ax2 = plt.subplots()
ax2.plot(X2, y_incidencia, 'go')
ax2.set_ylabel('# de Casos por cada 10000 personas')
ax2.set_xlabel('TRS (miles de toneladas)')
#ax2.axis([ 0, 1400,0, 25])
#plt.show()

#---------correlacion 
corr2= np.corrcoef(np.array(X2), np.array(y_incidencia))
print'correlacion'
print corr2
#----------intervalo de confianza total de RS
s_xxt=n*(np.var(X2)) 
print 's_xx'
print s_xxt

mt=np.mean(X2)
st=[]

for i in range(len(X2d)):
	st.append((y_incidencia[i]-m1)*((X2[i])- mt))
s_xyt=np.sum(st)
print 's_xy'
print s_xyt
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rt=((s_xyt)/(math.sqrt(s_xxt*s_yy)))
print rt
print'estadistico t'
tt=((rt*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rt,2)))))
print tt
print' Normal'
#zt=(math.lg((1+rt)/(1-rt)))/(2*(math.sqrt(1/(n-3))))
#print zt
#------DATOS FILTRADOS--------
p=np.array(data.prom_hab).T[1]
X3 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X3.append(p.astype(np.float)[i])
        y_incidencia.append(Y[i])
#-------Promedio de habitantes por vivienda-------        

fig3, ax3 = plt.subplots()
ax3.plot(X3 ,y_incidencia, 'mo')
ax3.set_ylabel('# de Casos por cada 10000 personas')
ax3.set_xlabel('# PHV')
#fig3.suptitle('Promedio de HV vs PID', fontsize=18)
#ax3.axis([ 3, 5,0, 25])
#plt.show()	
#---------correlacion
corr3= np.corrcoef(np.array(X3), np.array(y_incidencia))
print corr3
#----------intervalo de Promedio de habitantes por vivienda-
s_xxh=n*(np.var(X3)) 
print's_xx'
print s_xxh
mh=np.mean(X3)
sh=[]
for i in range(len(X2d)):
	sh.append((X3[i]- mh)*(y_incidencia[i]-m1))
s_xyh=np.sum(sh)
print's_xy'
print s_xyh
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rh=((s_xyh)/(math.sqrt(s_xxh*s_yy)))
print rh
print'estadistico t'
th=((rh*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rh,2)))))
print th
print' Normal'
#zh=(math.lg((1+rh)/(1-rh)))/(2*(math.sqrt(1/(n-3))))
#print zh
#------DATOS FILTRADOS-----
#------------Normalizando Tomas domiciliares de agua con viviendas
for i in range(len(data.tom_agua)):
	valor.append([data.tom_agua[i][0],float(data.tom_agua[i][1])/float(data.viviendas[i][1])])
N=np.array(valor).T[1]
X4 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X4.append(N.astype(np.float)[i])
        y_incidencia.append(Y[i])

#-------------Tomas domiciliarias de agua vs PID
fig4, ax4 = plt.subplots()
ax4.plot( X4, y_incidencia, 'yo')
ax4.set_ylabel('# de Casos por cada 10000 personas')
ax4.set_xlabel('# de TA')
#fig4.suptitle('Tomas de agua vs PID', fontsize=18)
ax4.axis([ 0, 1.5,0, 25])
#plt.show()	
#---------correlacion
corr4= np.corrcoef(np.array(X4), np.array(y_incidencia))
print corr4
#----------intervalo de Tomas domiciliarias de agua---
s_xxta=n*(np.var(X4)) 
print's_xx'
print s_xxta
mta=np.mean(X4)
sta=[]
for i in range(len(X2d)):
	sta.append((X4[i]- mta)*(y_incidencia[i]-m1))
s_xyta=np.sum(sta)
print 's_xy'
print s_xyta
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
#print ' Coeficiente de c.m.'
rta=((s_xyta)/(math.sqrt(s_xxta*s_yy)))
#print rt
print'estadistico t'
tta=((rta*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rta,2)))))
print tta
print' Normal'

#--- datos filtrados-----
#------------Normalizando localidades con distribucion de agua 
for i in range(len(data.loc_agua)):
	valor1.append([data.loc_agua[i][0],float(data.loc_agua[i][1])/float(data.localidad[i][1])])
l=np.array(valor1).T[1]

X5 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X5.append(l.astype(np.float)[i])
        y_incidencia.append(Y[i])


#------localidades con distribucion de agua  vs PID
fig5, ax5 = plt.subplots()
ax5.plot( X5, y_incidencia, 'bo')
ax5.set_ylabel('# de Casos por cada 10000 personas')
ax5.set_xlabel('# de LA')
#fig5.suptitle('Localidades con agua vs PID', fontsize=18)
ax5.axis([ 0, 1.2,0, 25])
#plt.show()	
#---------correlacion

corr5= np.corrcoef(np.array(X5), np.array(y_incidencia))
print corr5
#----------intervalo de Localidades con agua---
s_xxla=n*(np.var(X5)) 
print 's_xx'
print s_xxla
mla=np.mean(X5)
sla=[]
for i in range(len(X2d)):
	sla.append((X5[i]- mla)*(y_incidencia[i]-m1))
s_xyla=np.sum(sla)
print's_xy'
print s_xyla
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
#print ' Coeficiente de c.m.'
rla=((s_xyla)/(math.sqrt(s_xxla*s_yy)))
print'estadistico t'
tla=((rla*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rla,2)))))
print tla
print' Normal'

#   -------FILTRAR DATOS----

#------------Normalizando localidades con drenaje y alcantarillado con localidades
for i in range(len(data.loc_alcantarillado)):
    valor2.append([data.loc_alcantarillado[i][0],float(data.loc_alcantarillado[i][1])/float(data.localidad[i][1])])
d=np.array(valor2).T[1]
X6 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X6.append(d.astype(np.float)[i])
        y_incidencia.append(Y[i])

#----- Localidades con drenaje y alcantarillado vs PID
fig6, ax6 = plt.subplots()
ax6.plot(X6, y_incidencia, 'ko')
#ax6.plot(d, 10000*Y, 'ko')
ax6.set_ylabel('# de Casos por cada 10000 personas')
ax6.set_xlabel('# de LDA')
#fig6.suptitle(' L.Drenaje y Alcantarillado vs PID', fontsize=18)
ax6.axis([0,1, 0, 25])  #recorre ejes
#plt.show()
#------------------------- correlacion
corr6= np.corrcoef(np.array(X6), np.array(y_incidencia))
print corr6
#----------intervalo de Localidades con drenaje y alcantarillado ---
s_xxda=n*(np.var(X6)) 
print's_xx'
print s_xxda
mda=np.mean(X6)
sda=[]
for i in range(len(X2d)):
	sda.append((X6[i]- mda)*(y_incidencia[i]-m1))
s_xyda=np.sum(sda)
print's_xy'
print s_xyda
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rda=((s_xyda)/(math.sqrt(s_xxda*s_yy)))
print rda
print'estadistico t'
tda=((rda*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rda,2)))))
print tda
print' Normal'

#----------------------normalizacion precipitacion'/////'
precipitacion=np.array(data.precipitation).T[1]

X7 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X7.append(precipitacion.astype(np.float)[i])
        y_incidencia.append(Y[i])

#-- -----------------precipitacion vs PID
fig7, ax7 = plt.subplots()
ax7.plot( X7 ,y_incidencia, 'og')
ax7.set_ylabel('# de Casos por cada 10000 personas')
ax7.set_xlabel('P')
#fig7.suptitle('Precipitacion vs PID', fontsize=18)
#ax7.axis([0, 200,0, 0.00020])
#plt.show()
#---------correlacion
corr7= np.corrcoef(np.array(X7), np.array(y_incidencia))
print corr7
#----------intervalo de Precipitacion---
s_xxp=n*(np.var(X7)) 
print 's_xx'
print s_xxp
mp=np.mean(X7)
sp=[]
for i in range(len(X2d)):
	sp.append((X7[i]- mp)*(y_incidencia[i]-m1))
s_xyp=np.sum(sp)
print's_xy'
print s_xyp
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rp=((s_xyp)/(math.sqrt(s_xxp*s_yy)))
print rp
print'estadistico t'
tp=((rp*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rp,2)))))
print tp

#--- FILTRAR DATOS-----
#----------------------normalizacion temp_minima'/////'
tem_min=np.array(data.temp_min).T[1]
X8 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X8.append(tem_min.astype(np.float)[i])
        y_incidencia.append(Y[i])
#------- temperatura minima vs PID        
fig8, ax8 = plt.subplots()
ax8.plot(X8, y_incidencia, 'oc')
ax8.set_ylabel('# de Casos por cada 10000 personas')
ax8.set_xlabel('TMI')
#fig8.suptitle('Temperatura minima vs PID', fontsize=18)
#ax8.axis([5, 25,0, 0.00020])
#plt.show()
#---------correlacion
corr8= np.corrcoef(np.array(X8), np.array(y_incidencia))
print corr8
#----------intervalo de temperatura minima---
s_xxti=n*(np.var(X8)) 
print 's_xx'
print s_xxti
mti=np.mean(X8)
sti=[]
for i in range(len(X2d)):
	sti.append((X8[i]- mti)*(y_incidencia[i]-m1))
s_xyti=np.sum(sti)
print's_xy'
print s_xyti
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rti=((s_xyti)/(math.sqrt(s_xxti*s_yy)))
print rti
print'estadistico t'
tti=((rti*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rti,2)))))
print tti
print' Normal'


#----------Filtrar datos----
#----------------------normalizacion temp_media'/////'

tem_med=np.array(data.temp_med).T[1]
X9 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X9.append(tem_med.astype(np.float)[i])
        y_incidencia.append(Y[i])

#------temperatura media vs PID
fig9, ax9 = plt.subplots()
ax9.plot(X9, y_incidencia, 'og')
ax9.set_ylabel(' # de Casos por cada 10000 personas')
ax9.set_xlabel('TME')
#fig9.suptitle('Temperatura media vs PID', fontsize=18)
ax9.axis([18, 28,0, 25])
#plt.show()
#---------correlacion

corr9= np.corrcoef(np.array(X9), np.array(y_incidencia))
print corr9
#----------intervalo de temperatura media---
s_xxte=n*(np.var(X9)) 
print 's_xx'
print s_xxte
mte=np.mean(X9)
ste=[]
for i in range(len(X2d)):
	ste.append((X9[i]- mte)*(y_incidencia[i]-m1))
s_xyte=np.sum(ste)
print 's_xy'
print s_xyte
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rte=((s_xyte)/(math.sqrt(s_xxte*s_yy)))
print rte
print'estadistico t'
tte=((rte*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rte,2)))))
print tte


# ----FILTRAR DATOS-----
#----------------------normalizacion temp_maxima'/////'

tem_max=np.array(data.temp_max).T[1]
X10 = []
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        X10.append(tem_max.astype(np.float)[i])
        y_incidencia.append(Y[i])

#------temperatura maxima vs PID
fig10, ax10 = plt.subplots()
ax10.plot(X10, y_incidencia, 'ob')
ax10.set_ylabel('# de Casos por cada 10000 personas')
ax10.set_xlabel('TMA')
#fig10.suptitle('Temperatura maxima vs PID', fontsize=18)
ax10.axis([25, 34,0, 25])
#plt.show()
#---------correlacion
corr10= np.corrcoef(np.array(X10), np.array(y_incidencia))
print corr10
#----------intervalo de temperatura maxima---
s_xxtm=n*(np.var(X10)) 
print 's_xx'
print s_xxtm
mtm=np.mean(X10)
stm=[]
for i in range(len(X2d)):
	stm.append((X10[i]- mtm)*(y_incidencia[i]-m1))
s_xytm=np.sum(stm)
print 's_xy'
print s_xytm
print't tabulado'
print 2.787436 #Distribucion t con nivel de significancia 0.01 Calculado  con : qt(0.005,df=25)
print ' Coeficiente de c.m.'
rtm=((s_xytm)/(math.sqrt(s_xxtm*s_yy)))
print rtm
print'estadistico t'
ttm=((rtm*(math.sqrt(n-2)))/(math.sqrt(1-(pow(rtm,2)))))
print ttm


#print'variable-generacion RS'
#print X2a
#print'variable-disposicion RS'
#print X2d
#print'variable- Total de GRS'
#print X2
#print'variable-PHV'
#print X3
#print'variable-TA'
#print X4
#print'variable-LA'
#print X5
#print'variable-LDA'
#print X6
#print'variables-Pre'
#print X7
#print'variable-TMI'
#print X8
#print'variable-TME'
#print X9
#print'variable-TMA'
#print X10

#-------------Matriz de correlacion------Normalizar datos
X2_GEN= []
X2_DIS= []
X2_TRS= []
X3_PHV= []
X4_TA= []
X5_LA= []
X6_LDA= []
X7_P= []
X8_TMI= []
X9_TME= []
X10_TMA= []

for i in range(len(X2)):
    X2_GEN.append((X2a[i]-min(X2))/(max(X2a)-min(X2a)))
    X2_DIS.append((X2d[i]-min(X2d))/(max(X2d)-min(X2d)))
    X2_TRS.append((X2[i]-min(X2))/(max(X2)-min(X2)))
    X3_PHV.append((X3[i]-min(X3))/(max(X3)-min(X3)))
    X4_TA.append((X4[i]-min(X4))/(max(X4)-min(X4)))
    X5_LA.append((X5[i]-min(X5))/(max(X5)-min(X5)))
    X6_LDA.append((X6[i]-min(X6))/(max(X6)-min(X6)))
    X7_P.append((X7[i]-min(X7))/(max(X7)-min(X7)))
    X8_TMI.append((X8[i]-min(X8))/(max(X8)-min(X8)))
    X9_TME.append((X9[i]-min(X9))/(max(X9)-min(X9)))
    X10_TMA.append((X10[i]-min(X10))/(max(X10)-min(X10)))
    
print X3_PHV

cov_mat = np.cov([X2_DIS,X3_PHV,X4_TA,X5_LA,X6_LDA,X7_P,X8_TMI,X9_TME,X10_TMA])
#cov_mat= np.cov([X2_GEN,X3_PHV,X4_TA,X5_LA,X6_LDA,X7_P,X8_TMI,X9_TME,X10_TMA])
#cov_mat= np.cov([X2_TRS,X3_PHV,X4_TA,X5_LA,X6_LDA,X7_P,X8_TMI,X9_TME,X10_TMA])
#cov_mat= np.cov([X2_GEN,X2_DIS,X3_PHV,X4_TA,X5_LA,X6_LDA,X7_P,X8_TMI,X9_TME,X10_TMA])


#MC=np.corrcoef([X2_DIS,X3_PHV,X4_TA,X5_LA,X6_LDA,X7_P,X8_TMI,X9_TME,X10_TMA])
MC=np.cov([X2d,X3,X4,X5,X6,X7,X8,X9,X10])
#print'CORRELACION'
#print Mc
#print'////////'
print 'COVARIANZA'
print cov_mat
print'//////'

print "MATRIZ Covarianza datos no Normalizados"
print MC
print "//////////////////////////"

#----------------eigenvectores y eigenvaloresd de CORRELACION-------
eig_valc, eig_vecc=np.linalg.eig(MC)
for i in range(len(eig_valc)):
    eigvec_cov = eig_vecc[:,i].reshape(1,9).T
    print('Eigenvalor {} de matriz de cov no norm: {}'.format(i+1, eig_valc[i]))
    print('Eigenvector {} '.format(eig_vecc[i]))
    print('Otra forma de Eigenvenctor {} '.format(eig_vecc[i]/eig_vecc[i][0]))
    print(10 * '-')

#----------------eigenvectores y eigenvalores
#eig_valc, eig_vecc=np.linalg.eig(cov_mat)
#   eigvec_cov = eig_vecc[:,i].reshape(1,9).T
 #   print('Eigenvalor {} de matriz de covarianza: {}'.format(i+1, eig_valc[i]))
  #  print('Eigenvector {} '.format(eig_vecc[i]))
   # print('Otra forma de Eigenvenctor {} '.format(eig_vecc[i]/eig_vecc[i][0]))
    #print(10 * '-')
#-----------------Combinacion lineal 
#----normalizar eigenvectores
eig_vecc_unit = []
for ev in eig_vecc:
    eig_vecc_unit.append(ev/np.linalg.norm(ev))
xx= []
yy=[]
for estado in range(len(X2_DIS)):
     xx.append(eig_vecc_unit[0][0]*X2[estado]+eig_vecc_unit[0][1]*X3[estado]+eig_vecc_unit[0][2]*X4[estado]+eig_vecc_unit[0][3]*X5[estado]+eig_vecc_unit[0][4]*X6[estado]+eig_vecc_unit[0][5]*X7[estado]+eig_vecc_unit[0][6]*X8[estado]+eig_vecc_unit[0][7]*X9[estado]+eig_vecc_unit[0][8]*X10[estado])
     yy.append(eig_vecc_unit[1][0]*X2[estado]+eig_vecc_unit[1][1]*X3[estado]+eig_vecc_unit[1][2]*X4[estado]+eig_vecc_unit[1][3]*X5[estado]+eig_vecc_unit[1][4]*X6[estado]+eig_vecc_unit[1][5]*X7[estado]+eig_vecc_unit[1][6]*X8[estado]+eig_vecc_unit[1][7]*X9[estado]+eig_vecc_unit[1][8]*X10[estado])
    #xx.append(eig_vecc_unit[0][0]*X2DIS[estado]+eig_vecc_unit[0][1]*X3_PHV[estado]+eig_vecc_unit[0][2]*X4_TA[estado]+eig_vecc_unit[0][3]*X5_LA[estado]+eig_vecc_unit[0][4]*X6_LDA[estado]+eig_vecc_unit[0][5]*X7_P[estado]+eig_vecc_unit[0][6]*X8_TMI[estado]+eig_vecc_unit[0][7]*X9_TME[estado]+eig_vecc_unit[0][8]*X10_TMA[estado])
    #yy.append(eig_vecc_unit[1][0]*X2_DIS[estado]+eig_vecc_unit[1][1]*X3_PHV[estado]+eig_vecc_unit[1][2]*X4_TA[estado]+eig_vecc_unit[1][3]*X5_LA[estado]+eig_vecc_unit[1][4]*X6_LDA[estado]+eig_vecc_unit[1][5]*X7_P[estado]+eig_vecc_unit[1][6]*X8_TMI[estado]+eig_vecc_unit[1][7]*X9_TME[estado]+eig_vecc_unit[1][8]*X10_TMA[estado])
    

#--------Grafica combinacion lineal e incidencia
fig11, ax11 = plt.subplots()
ax11.plot(xx, y_incidencia, 'ob')
ax11.set_ylabel('# de Casos por cada 10000 personas')
ax11.set_xlabel('PCP')
fig11.suptitle('Primer componente principal vs Casos de dengue', fontsize=18)
ax11.axis([0, 500,0, 25])
plt.show()

#--------Grafica 2a componente principal e incidencia
fig12, ax12 = plt.subplots()
ax12.plot(yy, y_incidencia,  'og')
ax12.set_ylabel('# de Casos por cada 10000 personas')
ax12.set_xlabel('SCP')
#fig12.suptitle('Segunda componente principal vs Casos de dengue', fontsize=18)
ax12.axis([-15, 25,0, 25])
plt.show()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#--------Grafica residuos-precipitacion-incidencia
fig13=plt.figure(13)
ax=fig13.add_subplot(111,projection='3d')
#ax.scatter(X2_DIS,X7_P,y_incidencia)
ax.scatter(X2d,X8,y_incidencia)
#ax.set_xlabel('Dis. Residuos Solidos')
#ax.set_ylabel('Temperatura Minima')
ax.set_xlabel('DRS')
ax.set_ylabel('TMI')
ax.set_zlabel('# de Casos p/c 10000 personas')
ax.set_title('Correlacion componentes principales')
ax.axis([0,2000,10,25])
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#-----modelo prediccion------superficie de segundo orden
def fn(x, a, b, c,d,e,f):
    return a + b*x[:,0] + c*x[:,1]+ d*x[:,0]*x[:,0] + e*x[:,1]*x[:,1]+ f*x[:,0]*x[:,1]
#cp= scipy.array([X2_DIS, X7_P, y_incidencia]).T
cp= scipy.array([X2d, X8, y_incidencia]).T
y_dengue= scipy.array(y_incidencia)
x0 = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
fitParams, fitCovariances = curve_fit(fn, cp[:,:2], cp[:,2], x0)
print ' coeficientes fit Residuos Precipitacion 2Orden:\n', fitParams
#print 'cp'
cp1=scipy.array([X2d, X8]).T
#print cp1
#exit
fig14=plt.figure(14)
ax=fig14.add_subplot(111,projection='3d')
#ax.scatter(X2_DIS,X7_P,y_incidencia)
ax.scatter(X2d,X8,y_incidencia)
#X = np.arange(0,2000+60,65)
#Y = np.arange(10,20+8,4)
X = np.arange(0,2000+200,200)
Y = np.arange(10,20+8,3)
X, Y = np.meshgrid(X, Y)
Z =  fitParams[0] + fitParams[1]*X + fitParams[2]*Y + fitParams[3]*X*X + fitParams[4]*Y*Y+fitParams[5]*X*Y
ax.plot_wireframe(X,Y,Z,color='c',)
ax.set_xlabel('DRS')
ax.set_ylabel('TMI')
ax.set_zlabel('# de Casos p/c 10000 personas')
#ax.set_title('Modelo prediccion segundo orden ')
#plt.axis([0, max(X2d), 0, max(X8)])
plt.axis([0, 2000, 10, 25])
#ax.azim = 200 
#ax.elev = 45
plt.show()


#---------Normalidad---------------------------
import scipy.stats as stats

y = y_incidencia - fn(cp1,1.03902203e+01,-1.50459320e-03,-1.70122342, 9.61475423e-08, 7.08672648e-02,8.96065492e-05)
plt.figure(101)
plt.plot(y,'go')
plt.show()
y_filt = [i if (i<10 and i>-10) else 0 for i in y]

#plt.plot(xdata,y,'go')
#plt.show()

stats.probplot(y_filt, dist="norm", plot=plt)
plt.plot()
plt.show()
stats.probplot(y,dist="norm", plot=plt)
plt.plot()

plt.title("Normal Q-Q")
#plt.axis([-2.0,2.0,-0.5,0.5])
plt.ylabel('valores ordenados')
plt.show()
print '---------------------------'

print '********************'
#print y_incidencia
print'*********'
#print y
#-----------------------------------------------------
#fig14.savefig('Modelo2Orden.jpg', bbox_inches='tight')

#&&&&&&&&&&&&&&&
#---------------------Calculo del error
datosX= []
for i in range(len(X2_DIS)):
	datosX.append([X2d[i],X8[i]])
datos_X= np.array(datosX)
#print 'datos_X'
SSE = 0.0
for i in range(len(y_incidencia)):
    SSEC = SSE + math.pow(10000.0*(y_incidencia[i] - (fitParams[0] + fitParams[1]*datos_X[i][0] + fitParams[2]*datos_X[i][1] + fitParams[3]*datos_X[i][0]*datos_X[i][0] + fitParams[4]*datos_X[i][1]*datos_X[i][1]+fitParams[5]*datos_X[i][0]*datos_X[i][1])),2.0)
print 'SSEC'
print SSEC
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-----modelo prediccion------superficie primer orden
def fnc(x, g, h, k):
    return g + h*x[:,0] + k*x[:,1]
#cpr= scipy.array([X2_DIS, X7_P, y_incidencia]).T
cpr= scipy.array([X2d, X8, y_incidencia]).T
y_dengue= scipy.array(y_incidencia)
x0r = np.array([1.0,1.0,1.0])
fitParams, fitCovariances = curve_fit(fnc, cpr[:,:2], cpr[:,2], x0r)
print'fit reducido'
print ' coeficientes fit Precip y Residuos 1 orden:\n', fitParams

fig15=plt.figure(15)
ax=fig15.add_subplot(111,projection='3d')
#ax.scatter(X2_DIS,X7_P,y_incidencia)
ax.scatter(X2d,X8,y_incidencia)
#X = np.arange(0,2000,200)
#Y = np.arange(5,25,20)
X = np.arange(0,2000+200,200)
Y = np.arange(10,20+10,4)
X, Y = np.meshgrid(X, Y)
Z =  fitParams[0] + fitParams[1]*X + fitParams[2]*Y
ax.plot_wireframe(X,Y,Z,color='c',)
ax.set_xlabel('DRS')
ax.set_ylabel('TMI')
ax.set_zlabel('# de Casos p/c 10000 personas')
#ax.set_title('Modelo prediccion primer orden')
plt.axis([0, 2000, 0, 25])
plt.show()
#fig15.savefig('Modelo1Orden.jpg', bbox_inches='tight')


#---------Normalidad---------------------------
import scipy.stats as stats
y = y_incidencia - fnc(cp1,-4.61986522,5.27658092e-05,4.32518778e-01)
y_filt = [i if (i<10 and i>-10) else 0 for i in y]
plt.figure(102)
plt.plot(y,'go')
plt.show()
#plt.plot(xdata,y,'go')
#plt.show()

stats.probplot(y_filt, dist="norm", plot=plt)
plt.ylabel('valores ordenados')
plt.plot()
plt.show()
stats.probplot(y,dist="norm", plot=plt)
plt.plot()

plt.title("Normal Q-Q")
#plt.axis([-2.0,2.0,-0.5,0.5])
plt.show()
print '---------------------------'

#---------------------Calculo del error
SSER = 0.0
for i in range(len(y_incidencia)):
    SSER = SSER + math.pow(10000.0*(y_incidencia[i] - (fitParams[0] + fitParams[1]*datos_X[i][0] + fitParams[2]*datos_X[i][1])),2.0)
print 'SSER'
print SSER
#-----Calculo de diferencia de errores
SSEA=SSER-SSEC
print 'SSEA'
print SSEA
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#----Prueba de hipotesis
k=len(x0)-1      # numero de var. ind .del Modelo completo
g=len(x0r)-1     # numero de var. ind .del Modelo reducido
n=len(X2_DIS)        # tamano de la muestra
v1=k-g           # g.ls en el numerador
v2=n-(k+1)       # g.l en el denomina
estadistico_F=(SSEA/v1)/(SSEC/v2)
print 'estadistico Residuos solidos'
print estadistico_F
print v1
print v2
#Para encontrar el valor p de distribucion F
print 'p_valor'
#valor_p =scipy.stats.f.cdf(estadistico_F, v1, v2)
valor_p =scipy.stats.f.cdf(estadistico_F, v1, v2)
print valor_p
print '1-p_valor '
print 1-valor_p # no hay suficiente evidencia para apoyar la afirmacion b1,b2,b3,b4 o b5 difieren de cero