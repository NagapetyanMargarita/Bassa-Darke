import sqlite3
import os
from http.server import HTTPServer
from http.server import CGIHTTPRequestHandler
from xml.dom import minidom
from math import exp
from random import seed
from random import random

server_address=("localhost", 8000)
http_server = HTTPServer(server_address, CGIHTTPRequestHandler)
http_server.serve_forever()
'''def initialize_network(n_inputs, n_hidden_1, n_hidden_2, n_outputs):
	network = list()
	hidden_layer_1 = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden_1)]
	network.append(hidden_layer_1)
	hidden_layer_2 = [{'weights': [random() for i in range(n_hidden_1 + 1)]} for i in range(n_hidden_2)]
	network.append(hidden_layer_2)
	output_layer = [{'weights': [random() for i in range(n_hidden_2 + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
# Инлуцированное локальное поле
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Функция активации
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Прямое распространение
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Функция активации
def transfer_derivative(output):
	return output * (1.0 - output)

# Обратное распространение
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta']) #ошибка для каждого нейрона в скрытом слое
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output']) #ошибка для каждого нейрона вне скрытого слоя
		for j in range(len(layer)): #конечный подсчет ошибки
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Обновление весов с ошибкой
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):#сеть, набор, скорость, кол-во эпох, кол-во выхода
	for epoch in range(n_epoch):
		sum_error = 0 #Подсчет ошибки
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Предсказание
def predict(network, row):
	outputs = forward_propagate(network, row)
	print(outputs)
	return outputs.index(max(outputs))

# Test training backprop algorithm
seed(1)
dataset, dataset_1, dataset_2, dataset_3, dataset_4 = [], [], [], [],[]
with  open('answers_1.txt', 'r') as f1, open('answers_2.txt', 'r') as f2, open('answers_3.txt', 'r') as f3, open('answers_4.txt', 'r') as f4:
	for line in f1:
		dataset_1.append(list(map(int, line.strip().replace(' ', ''))))
	dataset.append(dataset_1)
	for line in f2:
		dataset_2.append(list(map(int, line.strip().replace(' ', ''))))
	dataset.append(dataset_2)
	for line in f3:
		dataset_3.append(list(map(int, line.strip().replace(' ', ''))))
	dataset.append(dataset_3)
	for line in f4:
		dataset_4.append(list(map(int, line.strip().replace(' ', ''))))
	dataset.append(dataset_4)
print(dataset[1][0])'''

'''n_inputs = len(dataset[0]) - 1 #кол-во входных нейронов
n_outputs = len(set([row[-1] for row in dataset])) #кол-во индивидуальных выходных данных
network = initialize_network(n_inputs, 10, n_outputs) #инициализация сети
train_network(network, dataset, 0.5, 150, n_outputs) #обучение сети
row=[0,0,0,1,0,0,0,0,0,0,0,0,0,1]
prediction = predict(network, row)
Otvet_1=["Инфантильность, ощущение одиночества", "Ответственность, Эмпатия", "Эмпатичность, Коммуникабельность", "Коммуникабельность"]
Otvet_2=["Спокойствие", "Активность", "Усталость", "Целеустремленость"]
Otvet_3=["Уверенность в себе", "Самодисциплина", "Склонность к депрессии", "Саморазвитие"]
Otvet_4=["Упертость, Внутренняя гармония", "Пониженная самооценка", "Активность", "Самозанятость"]
print(Otvet_1[prediction])'''
'''for layer in network:
	print(layer)
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))'''