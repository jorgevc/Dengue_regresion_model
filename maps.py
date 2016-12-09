from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
import datos as dpl

#----------------------------prueba---------------------------
data1 = dpl.data("DATA")
proba_incidencia= []

for i in range(len(data1.cases)):
	proba_incidencia.append([data1.cases[i][0],float(data1.cases[i][53])/ float(data1.poblacion[i][1])])

#p_incidencia = np.array(proba_incidencia[:][1])
p_incidencia = np.array(proba_incidencia).T[1]



                                                                     

print '-------------------'
#----------------map--------------------------------------
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import rgb2hex

cmap = plt.cm.autumn

fig     = plt.figure()
ax      = fig.add_subplot(111)

m = Basemap(projection='mill',
            llcrnrlat =14,
            llcrnrlon = -118,
            urcrnrlat = 33,
            urcrnrlon = -86,
            resolution='l')

m.readshapefile('./DATA/mexstates', 'mexstates')
m.readshapefile('./DATA/destdv1gw', 'destdv1gw')
sonoraPolygon = []
sonoraInfo = []
bajaPolygon = []
bajaInfo = []
bajPolygon = []
bajInfo = []
agcPolygon = []
agcInfo = []
naPolygon = []
naInfo = []
caPolygon = []
caInfo = []
chiPolygon = []
chiInfo = []
quinPolygon = []
quinInfo = []
for info, shape in zip(m.destdv1gw_info,m.destdv1gw):
    if(info['ENTIDAD']=='SONORA'):
        sonoraPolygon.append(Polygon(np.array(shape), True))
        sonoraInfo.append(info)
    if(info['ENTIDAD']=='BAJA CALIFORNIA SUR'):
        bajaPolygon.append(Polygon(np.array(shape), True))
        bajaInfo.append(info)
    if(info['ENTIDAD']=='BAJA CALIFORNIA'):
        bajPolygon.append(Polygon(np.array(shape), True))
        bajInfo.append(info)    
    if(info['ENTIDAD']=='AGUASCALIENTES'):
        agcPolygon.append(Polygon(np.array(shape), True))
        agcInfo.append(info)    
    if(info['ENTIDAD']=='NAYARIT'):
        naPolygon.append(Polygon(np.array(shape), True))
        naInfo.append(info)
    if(info['ENTIDAD']=='CAMPECHE'):
        caPolygon.append(Polygon(np.array(shape), True))
        caInfo.append(info)
    if(info['ENTIDAD']=='CHIAPAS'):
        chiPolygon.append(Polygon(np.array(shape), True))
        chiInfo.append(info)    
    if(info['ENTIDAD']=='QUINTANA ROO'):
        quinPolygon.append(Polygon(np.array(shape), True))
        quinInfo.append(info)    
#print quinInfo
#print'----'
#print quinPolygon    

patches = []
infos = []

for info, shape in zip(m.mexstates_info, m.mexstates):
    patches.append( Polygon(np.array(shape), True) )
    infos.append(info['ADMIN_NAME'])


info1=sorted(zip(infos, patches))
#print info1

print info1[2]
print patches[4]

estates = []

estates.append([agcInfo[0], agcPolygon[0]])
#estates.append(info1[0]) #Aguascalientes
#estates.append(info1[1]) #Baja California Norte
estates.append([bajInfo[0], bajPolygon[0]])
#estates.append(info1[11]) #Baja California Sur
estates.append([bajaInfo[0], bajaPolygon[0]])
#estates.append([caInfo[0], caPolygon[0]]) No pinta
estates.append(info1[16]) #Campeche
#estates.append(info1[17]) #Chiapas
estates.append([chiInfo[0], chiPolygon[0]])
estates.append(info1[19]) #Chihuahua
estates.append(info1[20]) #Coahuila
estates.append(info1[21]) #Colima
estates.append(info1[22]) #Distrito Federal
estates.append(info1[23]) #Durango
estates.append(info1[28]) #(Estado) Mexico
estates.append(info1[24]) #Guanajuato
estates.append(info1[25]) #Guerrero
estates.append(info1[26]) #Hidalgo
estates.append(info1[27]) #Jalisco
estates.append(info1[29]) #Michuacan
estates.append(info1[30]) #Morelos
#estates.append(info1[31]) #Nayarit
estates.append([naInfo[0], naPolygon[0]])
estates.append(info1[34]) #Nuevo Leon
estates.append(info1[35]) #Oaxaca
estates.append(info1[36]) #Puebla
estates.append(info1[37]) #Queretaro
#estates.append([quinInfo[0], quinPolygon[0]]) #no pinta
estates.append(info1[38]) #Quintana Roo
estates.append(info1[40]) #San Luis Potosi
estates.append(info1[41]) #Sinaloa
#estates.append(info1[42]) #Sonora
estates.append([sonoraInfo[0], sonoraPolygon[0]])
estates.append(info1[45]) #Tabasco
estates.append(info1[46]) #Tamaulipas
estates.append(info1[47]) #Tlaxcala
estates.append(info1[48]) #Veracruz
estates.append(info1[49]) #Yucatan
estates.append(info1[50]) #Zacatecas
print '-----xxxxx'

ax.add_collection(PatchCollection([estates[3][1]], facecolor= 'r', edgecolor='k', linewidths=1., zorder=2))

#colors[statename] = cmap( 1.-np.sqrt( (pop-vmin) /(vmax-vmin) ))[:3]
#color = rgb2hex(colors[statenames[nshape]])
#-----------filtro-------------------
#print p_incidencia
estates1=[]
Y = p_incidencia.astype(np.float)
y_incidencia = []
for i in range(len(Y)):
    if (Y[i]!=0.0):
        estates1.append(estates[i])#.astype(np.float)[i])
        y_incidencia.append(Y[i])
       # print "estado del mapa: %s , estado datos: %s" % (estates[i][0] , proba_incidencia[i][0])
    
#print estates1
print len(estates1)
print '.......'
#print y_incidencia
print len(y_incidencia)
print'&&&&&&&'
fig1, ax1 = plt.subplots()

for i in range(len(y_incidencia)):
    inc = (y_incidencia[i] - min(y_incidencia))/(max(y_incidencia)-min(y_incidencia))
    colorRGB = cmap(1.0-pow(inc,0.2))
    colorHex = rgb2hex(colorRGB)
    ax1.add_collection(PatchCollection([estates1[i][1]], facecolor= colorHex, linewidths=1.,zorder=(i+2)))
    # zorder=2
    # edgecolor='k'
    #print inc
#-----------end---------------------

#for i in range(len(p_incidencia)):
#    inc = (p_incidencia[i] - min(p_incidencia))/(max(p_incidencia)-min(p_incidencia))
  #  colorRGB = cmap(1.0-pow(inc,0.2))
    #colorHex = rgb2hex(colorRGB)
    #ax.add_collection(PatchCollection([estates[i][1]], facecolor= colorHex, edgecolor='k', linewidths=1., zorder=2))


m.drawcoastlines()
m.drawcountries()
#plt.title('REPUBLICA MEXICANA',fontsize=20)
plt.show()
print '-------------------'

#pru=[['mari','b',4],['adri','c',2],['zara','a',6]]
#pru1=sorted(pru)
#print pru1

