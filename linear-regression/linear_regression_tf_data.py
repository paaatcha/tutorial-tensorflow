#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Author: André Pacheco
Email: pacheco.comp@gmail.com

Implementação de uma regressão linear em TensorFlow
Caso encontre algum bug ou sugestão, por favor me envie um email
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Função para carregar o arquivo com os dados. Estou usando pandas apenas para
# simplicar. Não se preocupe caso não esteja habituado com a biblioteca
def carrega_dados (caminho, MODEL_ORDER):
    dados = pd.read_csv(caminho)
    X = dados['Birth-rate'].values
    X = np.reshape(X, (X.shape[0], MODEL_ORDER)) # apenas para ficar com shape [n,1]
    Y = dados['Life-expectancy'].values
    Y = np.reshape(Y, (Y.shape[0], 1)) # apenas para ficar com shape [n,1]
    return X, Y
    
############################## Parametros #####################################
CAMINHO = '../datasets/life-expec.csv'
MODEL_ORDER = 1
MAX_EPOCAS = 10000
TOL = 0.0001
LR = 0.01
BATCH_SIZE = 25
###############################################################################

# Carregando os dados de entrada
Xdata, Ydata = carrega_dados(CAMINHO, MODEL_ORDER)
# Adicionando 1 em cada dimensão de X para incluir b dentro de W
Xdata = np.concatenate ((Xdata, np.ones([Xdata.shape[0],1])), axis = 1)

dataset = tf.data.Dataset.from_tensor_slices((Xdata, Ydata))
dataset = dataset.shuffle(Xdata.shape[0])
dataset = dataset.batch(BATCH_SIZE)

iterator = dataset.make_initializable_iterator()
X, Yreal = iterator.get_next() 


# Acrescentando +1 linha em W. Esse será nosso b
W = tf.Variable(tf.random_uniform([MODEL_ORDER+1,1],-1,1,name='W', dtype=tf.float64)) 
Ypred = tf.matmul(X,W)

# Criando a funcao de custo (MSE)
func_custo = tf.reduce_mean(tf.pow(Ypred-Yreal,2), name="Func_Custo")

# Criando o otimizador
train_op = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(func_custo)

# Operador para inicializar todas as variaveis
init = tf.global_variables_initializer()

# Vou armazenar aqui o valor dos pesos treinados
pesos = None

# Iniciando a sessao
with tf.Session() as sess:
    # Criando arquivo de escrita do TensorBoard
#    writer = tf.summary.FileWriter("./tbdata", sess.graph)
    sess.run(init)
#    
    custo_anterior = 0
    
    try:
        for ep in range(MAX_EPOCAS):
            sess.run(iterator.initializer)
                
            _, custo, pesos = sess.run([train_op, func_custo, W])
            
            if (ep % 100 == 0):
                print "Epoca: ", ep, " | Funcao de custo: ", custo
                
            if (abs(custo_anterior - custo) < TOL):
                print "Os pesos convergiram"
                break
            
            custo_anterior = custo
    except tf.errors.OutOfRangeError:
        pass
        

    print "Pesos = ", pesos

# Plotando a curva da regressao
plt.plot(Xdata[:,0], Xdata.dot(pesos)[:,0], c='k', label='Regressao')    
plt.scatter(Xdata[:,0], Ydata[:,0], c='r', label='Dados')

plt.grid()
plt.legend(loc=0)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regressao linear')
plt.show()


















    


