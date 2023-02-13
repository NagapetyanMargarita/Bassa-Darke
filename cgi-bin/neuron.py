#!/usr/bin/python
import cgi
import os
import sqlite3
from math import exp
from random import seed
from random import random

form = cgi.FieldStorage()
n = []
f_agr = {0: 1, 8: 0, 16: 0, 24: 0, 32: 0, 40: 0, 47: 1, 54: 1, 61: 1, 67: 1}
v_agr = {6: 1, 14: 1, 22: 1, 30: 1, 38: 0, 45: 1, 52: 1, 59: 1, 65: 0, 70: 1, 72: 1, 73: 0, 74: 0}
k_agr = {1: 1, 9: 1, 17: 1, 25: 0, 33: 1, 41: 1, 48: 0, 55: 1, 62: 1}
neg = {3: 1, 11: 1, 19: 1, 27: 1, 35: 0}
razdr = {2: 1, 10: 0, 18: 1, 26: 1, 34: 0, 42: 1, 49: 1, 56: 1, 63: 1, 68: 0, 71: 1}
pod = {5: 1, 13: 1, 21: 1, 29: 1, 37: 1, 44: 1, 51: 1, 58: 1, 64: 0, 69: 0}
obid = {4: 1, 12: 1, 20: 1, 28: 1, 36: 1, 43: 1, 50: 1, 57: 1}
vina = {7: 1, 15: 1, 23: 1, 31: 1, 39: 1, 46: 1, 53: 1, 60: 1, 66: 1}
sum = 0
for i in range(1, 76):
    na = form.getfirst("n_"+str(i), "")
    if (na==""):
        sum += 1
    n.append(na)

if (sum != 0):
    flag = False
else:
    flag = True

print("Content-type: text/html\n\n")
print("""<!DOCTYPE HTML>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Обработка данных форм</title>
        </head>
        <body>
        <h1 style="text-align: center; margin-top: 60px">"""
      )
if flag:
    print("""Данные записаны!""")
else:
    print("""Ошибка! Заполнены не все поля!""")
print("""</h1>
        </body>
        </html>""")
def pods(slov, number):
    for k in slov.keys():
        if number == k:
            s=slov[k]
            return str(s)
def obch_sum (slov, number,i):
    if (i in slov):
        res = pods(slov, i)
        if (res == n[i]):
            number += 1
    return (number)


sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8 = 0, 0, 0, 0, 0, 0, 0, 0
for i in range(len(n)):
   sum1 = obch_sum(f_agr, sum1, i)
   sum2 = obch_sum(v_agr, sum2, i)
   sum3 = obch_sum(k_agr, sum3, i)
   sum4 = obch_sum(neg, sum4, i)
   sum5 = obch_sum(razdr, sum5, i)
   sum6 = obch_sum(pod, sum6, i)
   sum7 = obch_sum(obid, sum7, i)
   sum8 = obch_sum(vina, sum8, i)
sum1 *= 11
sum2 *= 8
sum3 *= 13
sum4 *= 20
sum5 *= 9
sum6 *= 11
sum7 *= 13
sum8 *= 11

#вывод сумм на экран пользователя
print(str(sum1) +" "+ str(sum2) +" " +str(sum3) +" " +str(sum4) +" " +str(sum5) +" " +str(sum6) +" " +str(sum7) + " " +str(sum8))

# данные для записи в файл с целью обучения нс
if (sum1>=0 and sum1<=37):
    res_1 = "0"
elif (sum1>=38 and sum1<=75):
    res_1 = "1"
else:
    res_1 = "2"
if (sum2>=0 and sum2<=35):
    res_2 = "0"
elif (sum2>=36 and sum2<=71):
    res_2 = "1"
else:
    res_2 = "2"
if (sum3>=0 and sum3<=39):
    res_3 = "0"
elif (sum3>=40 and sum3<=79):
    res_3 = "1"
else:
    res_3 = "2"
if (sum4>=0 and sum4<=33):
    res_4 = "0"
elif (sum4>=34 and sum4<=67):
    res_4 = "1"
else:
    res_4 = "2"
if (sum5>=0 and sum5<=33):
    res_5 = "0"
elif (sum5>=34 and sum5<=67):
    res_5 = "1"
else:
    res_5 = "2"
if (sum6>=0 and sum6<=37):
    res_6 = "0"
elif (sum6>=38 and sum6<=75):
    res_6 = "1"
else:
    res_6 = "2"
if (sum7>=0 and sum7<=35):
    res_7 = "0"
elif (sum7>=36 and sum7<=71):
    res_7 = "1"
else:
    res_7 = "2"
if (sum8>=0 and sum8<=33):
    res_8 = "0"
elif (sum8>=34 and sum8<=67):
    res_8 = "1"
else:
    res_8 = "2"
#Запись в файл
with open('fiz_agress.txt', 'a') as f1, open('verb_agress.txt', 'a') as f2, open('kosv_agress.txt', 'a') as f3, open('negat.txt', 'a') as f4, open('razdr.txt', 'a') as f5, open('podozr.txt', 'a') as f6, open('obida.txt', 'a') as f7, open('aut.txt', 'a') as f8:
    f1.write(str(n[0])+str(n[8])+str(n[16])+str(n[24])+str(n[32])+str(n[40])+str(n[47])+str(n[54])+str(n[61])+str(n[67]) + " " +res_1 +'\n')
    f2.write(str(n[6])+str(n[14])+str(n[22])+str(n[30])+str(n[38])+str(n[45])+str(n[52])+str(n[59])+str(n[65])+str(n[70])+str(n[72])+str(n[73])+str(n[74]) + " " + res_2 +'\n')
    f3.write(str(n[1])+str(n[9])+str(n[17])+str(n[25])+str(n[33])+str(n[41])+str(n[48])+str(n[55])+str(n[62]) + " " + res_3 + '\n')
    f4.write(str(n[3])+str(n[11])+str(n[19])+str(n[27])+str(n[35]) + " " + res_4 + '\n')
    f5.write(str(n[2])+str(n[10])+str(n[18])+str(n[26])+str(n[34])+str(n[42])+str(n[49])+str(n[56])+str(n[63])+str(n[68])+str(n[71]) + " " + res_5 + '\n')
    f6.write(str(n[5])+str(n[13])+str(n[21])+str(n[29])+str(n[37])+str(n[44])+str(n[51])+str(n[58])+str(n[64])+str(n[69]) + " " + res_6 + '\n')
    f7.write(str(n[4])+str(n[12])+str(n[20])+str(n[28])+str(n[36])+str(n[43])+str(n[50])+str(n[57]) + " " + res_7 + '\n')
    f8.write(str(n[7])+str(n[15])+str(n[23])+str(n[31])+str(n[39])+str(n[46])+str(n[53])+str(n[60])+str(n[66]) + " " + res_8 + '\n')


# Initialize a network
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
seed(1)'''
'''dataset = [[0, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 1, 0],
	[0, 1, 1, 0],
	[1, 0, 0, 1],
	[1, 0, 0, 1],
	[1, 1, 0, 1],
	[2, 1, 0, 2],
	[2, 0, 0, 2],
	[2, 1, 1, 2],
	[1, 1, 1, 1]]
n_inputs = len(dataset[0]) - 1 #кол-во входных нейронов
n_outputs = len(set([row[-1] for row in dataset])) #кол-во индивидуальных выходных данных
network = initialize_network(n_inputs, 10, n_outputs) #инициализация сети
train_network(network, dataset, 0.5, 170, n_outputs) #обучение сети
row=[1, 0, 1]
prediction = predict(network, row)
Otvet_1=["Слабая выраженность", "Умеренная выраженность", "Сильная выраженность"]

print(Otvet[prediction])'''
'''for layer in network:
	print(layer)
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))'''