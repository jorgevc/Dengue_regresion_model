import numpy as np
import csv

class data:
	
	def __init__(self, path):
		self.cases = []
		self.total = []
		self.dis_residuos = []
		self.gen_residuos = []
		self.loc_agua = []
		self.loc_alcantarillado = []
		self.prom_hab = []
		self.tom_agua = []
		self.viviendas = []
		self.localidad = []
		self.poblacion = []
		self.temp_min = []
		self.temp_med = []
		self.temp_max = []
		self.precipitation = []
		self.lon = []
		self.lat_lon = []
		
		with open(path + '/incidencia.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next() 
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				for i in range(1, len(tmp_row)):
					row.append(float(tmp_row[i]))
				self.cases.append(row)

		with open(path + '/Dis-residuos.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				row.append(float(tmp_row[len(tmp_row)-1]))
				self.dis_residuos.append(row)
			
		with open(path + '/Gen-residuos.csv', 'rb') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		    spamreader.next()
		    for tmp_row in spamreader:
		        row =  [tmp_row[0]]
		        row.append(float(tmp_row[len(tmp_row)-2]))
		        self.gen_residuos.append(row)
	
		with open(path + '/Localidades-agua.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			spamreader.next()
			for tmp_row in spamreader:
			    row = [tmp_row[0]]
			    if(tmp_row[len(tmp_row)-1] != 'ND'):
			        value = float(tmp_row[len(tmp_row)-1])
			        row.append(value)
			        self.loc_agua.append(row)
			    else:   
			        value ='ND'  # row.append(value)
			        row.append(0)
			        self.loc_agua.append(row)
                  
			
		with open(path + '/Localidades-drenaje-alcantarillado.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				try:
				    row.append(float(tmp_row[len(tmp_row)-1]))
				except ValueError:
				    row.append(0.0)
				self.loc_alcantarillado.append(row)
				
		with open(path + '/Ocupantes.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				value = float(tmp_row[len(tmp_row)-2])/float(tmp_row[1])
				row.append(value)
				self.prom_hab.append(row)
				
		with open(path + '/Tomas-domiciliares-agua.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				try:
					row.append(float(tmp_row[len(tmp_row)-1]))
				except ValueError:
				    row.append(0)
				self.tom_agua.append(row)
				
		with open(path + '/Viviendas-particulares-habitadas.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				row.append(float(tmp_row[4]))
				self.viviendas.append(row)		
		with open(path + '/Localidades.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next() 
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				for i in range(1, len(tmp_row)):
					row.append(float(tmp_row[i]))
				self.localidad.append(row)	
		with open(path + '/Poblacion.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next() 
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				for i in range(1, len(tmp_row)):
					row.append(float(tmp_row[i]))
				self.poblacion.append(row)	
		with open(path + '/Temp-med.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				try:
				    row.append(float(tmp_row[len(tmp_row)-1]))
				except ValueError:
				    row.append(0.0)
				self.temp_med.append(row)
			
		with open(path + '/Temp-max.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') 
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				try:
				    row.append(float(tmp_row[len(tmp_row)-1]))
				except ValueError:
				    row.append(0.0)
				self.temp_max.append(row)
		
		with open(path + '/Temp-min.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				try:
				    row.append(float(tmp_row[len(tmp_row)-1]))
				except ValueError:
				    row.append(0.0)
				self.temp_min.append(row)
			
		with open(path + '/Precipitacion.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				try:
				    row.append(float(tmp_row[len(tmp_row)-1]))
				except ValueError:
				    row.append(0.0)
				self.precipitation.append(row)
		with open(path + '/lon-lat.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next()
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				try:
				    row.append(float(tmp_row[len(tmp_row)-1]))
				except ValueError:
				    row.append(0.0)
				self.lon.append(row)
				
		with open(path + '/lon-lat.csv', 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			spamreader.next() 
			for tmp_row in spamreader:
				row = [tmp_row[0]]
				for i in range(1, len(tmp_row)):
					row.append(float(tmp_row[i]))
				self.lat_lon.append(row)