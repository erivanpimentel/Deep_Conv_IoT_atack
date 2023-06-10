# -->>>> A base de dados Utilizada no modelo está disponivél em: https://www.unb.ca/cic/datasets/iotdataset-2023.html
#função de callbackt
class Mycallback(tf.keras.callbacks.Callback):
    def on_epochs_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.8065:
            print('accuracy: 80%, vamos cancelar o treino! Essa condição já nos satisfaz!')
            self.model.stop_training = True

#vamos criar o modelo de rede Conv1D
# 64 O Números de filtros (Kernel)
# (3) --> o tamanha de cada filtro
# input_shape=(,1) --> tamanho da imagem em scala de cinza
# activation="relu" --> função de ativação
def modeloIoT(previssores_train, classe_train):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64,(3), activation="relu",input_shape=(previssores_train.shape[1],1)),
    #vamoa adicionar a uma camada de normalização (Normaliza a saída entre 0 e 1)
    tf.keras.layers.BatchNormalization(),
    #pool_size=(2,2) --> tamanho da camada de maxPooling2D() (Vai criar uma matriz 2x2 e retornar os maiores valores)
    tf.keras.layers.MaxPooling1D(pool_size=(4)),
    tf.keras.layers.Dropout(0.2),
    #Segunda camada de convolução
    tf.keras.layers.Conv1D(64,(3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=(4)),
    #Vamos transformar o a saída da rede em uma lista para passar a rede densa
    tf.keras.layers.Flatten(),
    #rede Densa
    tf.keras.layers.Dense(units=300, activation='relu', kernel_initializer='normal'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=300, activation='relu', kernel_initializer='normal'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=34, activation='softmax'),
    ])
    #camos compilar o modelo
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    modelo = model.fit(previssores_train, classe_train, batch_size=500, epochs=10,callbaks=[Mycallback()])
    return modelo


def prepocess():
    # Vamos carregar a base de dados
    DATASET_DIRECTORY = '/home/erivan/Documentos/estudo_machine/dados_IoT_2023/IoT/CICIoT2023'
    # endswith('.csv') ----> verifica se termina com ".csv"
    df_sets = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
    # Realizando a ordenação dos dados
    df_sets.sort()
    #Dados de treinamento
    training_sets = df_sets[:int(len(df_sets)*.8)]
    #dados de teste
    test_sets = df_sets[int(len(df_sets)*.8):]
    normalizador_teste = MinMaxScaler(feature_range=(0,1))
    #dados de test
    for test in test_sets:
        path = DATASET_DIRECTORY+'/'+test
        test = pd.read_csv(path)
        classe_test = test['label']
        previssores_test = test.drop('label', axis=1)
    
    labelencoder_teste = LabelEncoder()
    classe_test = labelencoder_teste.fit_transform(classe_test)
    classe_test_dumpy = np_utils.to_categorical(classe_test)
    previssores_test = normalizador_teste.fit_transform(previssores_test)
    
    previssores_test = np.array(previssores_test)

    #dados de treino
    for train in training_sets:
        path = DATASET_DIRECTORY+'/'+train
        train = pd.read_csv(path)
        classe_train = train['label']
        previssores_train = train.drop('label', axis=1)
    
    #normalizar os dados entre 0 e 1
    #train /= 255
    normalizador_train = MinMaxScaler(feature_range=(0,1))
    labelencoder_train = LabelEncoder()
    classe_train = labelencoder_train.fit_transform(classe_train)
    classe_train_dumpy = np_utils.to_categorical(classe_train)
    previssores_train = normalizador_train.fit_transform(previssores_train)
    
    #transformar o dataframe em array numpy
    previssores_train = np.array(previssores_train)
    
    #Transformando no formato indicado para rede 
    X_train = np.reshape(previssores_train,(previssores_train.shape[0],previssores_train.shape[1],1))
    X_test = np.reshape(previssores_test ,( previssores_test.shape[0],previssores_test.shape[1],1))
    
    return X_train, classe_train_dumpy, X_test, classe_test_dumpy


if __name__ == '__main__':
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    X_train, classe_train_dumpy, X_test, classe_test_dumpy = prepocess()
    resultado = modeloIoT(X_train, classe_train_dumpy)
    
    
