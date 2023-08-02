import json
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import random
import heapq
from fractions import Fraction as frac

from azureml.core.model import Model

folder_root = 'https://mcdiaml0013163313257.blob.core.windows.net/azureml-blobstore-7d43f6a0-6b8c-4be7-9f3c-cc968c380029/mcdi2021-proyecto/data/'
    
staff01_4_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff01_4_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff01_4_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff01_4_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff01_4_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)

staff02_4_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff02_4_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff02_4_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff02_4_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)
staff02_4_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_4_4_1_notes_durations.csv'), index_col = 0)

staff01_2_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff01_2_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff01_2_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff01_2_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff01_2_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)

staff02_2_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff02_2_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff02_2_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff02_2_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)
staff02_2_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_4_1_notes_durations.csv'), index_col = 0)

staff01_2_2_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff01_2_2_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff01_2_2_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff01_2_2_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff01_2_2_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)

staff02_2_2_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff02_2_2_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff02_2_2_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff02_2_2_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)
staff02_2_2_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_2_2_1_notes_durations.csv'), index_col = 0)

staff01_3_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff01_3_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff01_3_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff01_3_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff01_3_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)

staff02_3_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff02_3_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff02_3_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff02_3_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)
staff02_3_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_3_4_1_notes_durations.csv'), index_col = 0)

staff01_5_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff01_5_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff01_5_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff01_5_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff01_5_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)

staff02_5_4_1_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff02_5_4_2_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff02_5_4_3_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff02_5_4_4_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)
staff02_5_4_5_notes_durations = pd.read_csv(os.path.join(folder_root, 'staff01_5_4_1_notes_durations.csv'), index_col = 0)

master_n_notes = pd.read_csv(os.path.join(folder_root, 'master_n_notes.csv'), index_col = 0)

def get_n_notes_id(n_notes):
    id = master_n_notes.loc[(master_n_notes.staff01_note01 == n_notes[0]) &
                         (master_n_notes.staff01_note02 == n_notes[1]) &
                         (master_n_notes.staff01_note03 == n_notes[2]) &
                         (master_n_notes.staff01_note04 == n_notes[3]) &
                         (master_n_notes.staff01_note05 == n_notes[4]) &
                         (master_n_notes.staff02_note01 == n_notes[5]) &
                         (master_n_notes.staff02_note02 == n_notes[6]) &
                         (master_n_notes.staff02_note03 == n_notes[7]) &
                         (master_n_notes.staff02_note04 == n_notes[8]) &
                         (master_n_notes.staff02_note05 == n_notes[9])].index[0]
    return id

def define_durations(n_notes, staff = 1, time_signature = '4/4'):
    ## n_notes, lets define n
    n = np.count_nonzero(n_notes)
    n5_durations = [0, 0, 0, 0, 0]
    random_durations = [0, 0, 0, 0, 0]
    
    if time_signature == '4/4':
        staff01_1_notes_durations = staff01_4_4_1_notes_durations
        staff01_2_notes_durations = staff01_4_4_2_notes_durations
        staff01_3_notes_durations = staff01_4_4_3_notes_durations
        staff01_4_notes_durations = staff01_4_4_4_notes_durations
        staff01_5_notes_durations = staff01_4_4_5_notes_durations

        staff02_1_notes_durations = staff02_4_4_1_notes_durations
        staff02_2_notes_durations = staff02_4_4_2_notes_durations
        staff02_3_notes_durations = staff02_4_4_3_notes_durations
        staff02_4_notes_durations = staff02_4_4_4_notes_durations
        staff02_5_notes_durations = staff02_4_4_5_notes_durations
    if time_signature == '2/4':
        staff01_1_notes_durations = staff01_2_4_1_notes_durations
        staff01_2_notes_durations = staff01_2_4_2_notes_durations
        staff01_3_notes_durations = staff01_2_4_3_notes_durations
        staff01_4_notes_durations = staff01_2_4_4_notes_durations
        staff01_5_notes_durations = staff01_2_4_5_notes_durations

        staff02_1_notes_durations = staff02_2_4_1_notes_durations
        staff02_2_notes_durations = staff02_2_4_2_notes_durations
        staff02_3_notes_durations = staff02_2_4_3_notes_durations
        staff02_4_notes_durations = staff02_2_4_4_notes_durations
        staff02_5_notes_durations = staff02_2_4_5_notes_durations
    if time_signature == '2/2':
        staff01_1_notes_durations = staff01_2_2_1_notes_durations
        staff01_2_notes_durations = staff01_2_2_2_notes_durations
        staff01_3_notes_durations = staff01_2_2_3_notes_durations
        staff01_4_notes_durations = staff01_2_2_4_notes_durations
        staff01_5_notes_durations = staff01_2_2_5_notes_durations

        staff02_1_notes_durations = staff02_2_2_1_notes_durations
        staff02_2_notes_durations = staff02_2_2_2_notes_durations
        staff02_3_notes_durations = staff02_2_2_3_notes_durations
        staff02_4_notes_durations = staff02_2_2_4_notes_durations
        staff02_5_notes_durations = staff02_2_2_5_notes_durations
    if time_signature == '3/4':
        staff01_1_notes_durations = staff01_3_4_1_notes_durations
        staff01_2_notes_durations = staff01_3_4_2_notes_durations
        staff01_3_notes_durations = staff01_3_4_3_notes_durations
        staff01_4_notes_durations = staff01_3_4_4_notes_durations
        staff01_5_notes_durations = staff01_3_4_5_notes_durations

        staff02_1_notes_durations = staff02_3_4_1_notes_durations
        staff02_2_notes_durations = staff02_3_4_2_notes_durations
        staff02_3_notes_durations = staff02_3_4_3_notes_durations
        staff02_4_notes_durations = staff02_3_4_4_notes_durations
        staff02_5_notes_durations = staff02_3_4_5_notes_durations
    if time_signature == '5/4':
        staff01_1_notes_durations = staff01_5_4_1_notes_durations
        staff01_2_notes_durations = staff01_5_4_2_notes_durations
        staff01_3_notes_durations = staff01_5_4_3_notes_durations
        staff01_4_notes_durations = staff01_5_4_4_notes_durations
        staff01_5_notes_durations = staff01_5_4_5_notes_durations

        staff02_1_notes_durations = staff02_5_4_1_notes_durations
        staff02_2_notes_durations = staff02_5_4_2_notes_durations
        staff02_3_notes_durations = staff02_5_4_3_notes_durations
        staff02_4_notes_durations = staff02_5_4_4_notes_durations
        staff02_5_notes_durations = staff02_5_4_5_notes_durations
    
    if staff == 1:
        if n == np.int32(1):
            random_durations = random.choices(staff01_1_notes_durations.duration, 
                                              staff01_1_notes_durations.distribution, k=n)
        if n == np.int32(2):
            random_durations = random.choices(staff01_2_notes_durations.duration, 
                                              staff01_2_notes_durations.distribution, k=n)
        if n == np.int32(3):
            random_durations = random.choices(staff01_3_notes_durations.duration, 
                                              staff01_3_notes_durations.distribution, k=n)
        if n == np.int32(4):
            random_durations = random.choices(staff01_4_notes_durations.duration, 
                                              staff01_4_notes_durations.distribution, k=n)
        if n == np.int32(5):
            random_durations = random.choices(staff01_5_notes_durations.duration, 
                                              staff01_5_notes_durations.distribution, k=n)
    if staff == 2:
        if n == np.int32(1):
            random_durations = random.choices(staff02_1_notes_durations.duration, 
                                              staff02_1_notes_durations.distribution, k=n)
        if n == np.int32(2):
            random_durations = random.choices(staff02_2_notes_durations.duration, 
                                              staff02_2_notes_durations.distribution, k=n)
        if n == np.int32(3):
            random_durations = random.choices(staff02_3_notes_durations.duration, 
                                              staff02_3_notes_durations.distribution, k=n)
        if n == np.int32(4):
            random_durations = random.choices(staff02_4_notes_durations.duration, 
                                              staff02_4_notes_durations.distribution, k=n)
        if n == np.int32(5):
            random_durations = random.choices(staff02_5_notes_durations.duration, 
                                              staff02_5_notes_durations.distribution, k=n)
    for i in range(n):
        n5_durations[i] = random_durations[i]
    return n5_durations

def make_music(x):
    top_n = 5

    n_notes_4_4_sequence = pd.DataFrame(columns=['staff01_notes', 'staff01_durations', 
                                            'staff02_notes','staff02_durations'], dtype=object)

    n_notes_2_4_sequence = pd.DataFrame(columns=['staff01_notes', 'staff01_durations', 
                                            'staff02_notes','staff02_durations'], dtype=object)

    n_notes_2_2_sequence = pd.DataFrame(columns=['staff01_notes', 'staff01_durations', 
                                            'staff02_notes','staff02_durations'], dtype=object)

    n_notes_3_4_sequence = pd.DataFrame(columns=['staff01_notes', 'staff01_durations', 
                                            'staff02_notes','staff02_durations'], dtype=object)

    n_notes_5_4_sequence = pd.DataFrame(columns=['staff01_notes', 'staff01_durations', 
                                            'staff02_notes','staff02_durations'], dtype=object)

    main_n_notes_id_sequence = []

    ## Aquí obtenemos un valor aleatorio de una tripleta de secuencias de n-notas:
    ##    - Previo
    ##    - Actual
    ##    - Siguiente
    ## NOTA: Dado que x está definido como una secuencia de un elemento de n-notas,
    ##    sólo se extrae el valor "Previo" no se requiere de una selección de algún elemento de la tripleta
    ##encoded_n_notes = x[np.random.choice(range(len(x)), size=1)[0]]
    
    ## Aquí traemos el id de la secuencia de n-notes inicial
    n_notes_index = get_n_notes_id(x)
    ## Aquí traemos el tensor del acorde inicial (codificado)
    first_encoding = np.zeros((1, master_n_notes.shape[0]), dtype=np.float32)
    first_encoding[0][n_notes_index] = True    
    encoded_n_notes = np.expand_dims(first_encoding[-2:], axis=0)
    
    ##n_notes_index = encoded_n_notes.argmax()
    ## En esta variable vamos a comparar la repetición de secuencias de n-notas que se repitan (2, 3, 4, 5)
    main_n_notes_id_sequence.append(n_notes_index)
    ## Aquí traemos la secuencia de n-notes inicial desde el id
    all_n_notes = master_n_notes.loc[n_notes_index]
    ## Aquí separamos la secuencia completa de n-notes en las 2 manos, representados por cada pentagrama
    staff01_n_notes = all_n_notes[0:5].values
    staff02_n_notes = all_n_notes[5:10].values
    ## Aquí definimos la duración de las notas en cada secuencia de n-notes por pentagrama
    staff01_4_4_durations = define_durations(staff01_n_notes, 1)
    staff02_4_4_durations = define_durations(staff02_n_notes, 2)
    staff01_2_4_durations = define_durations(staff01_n_notes, 1, time_signature = '2/4')
    staff02_2_4_durations = define_durations(staff02_n_notes, 2, time_signature = '2/4')
    staff01_2_2_durations = define_durations(staff01_n_notes, 1, time_signature = '2/2')
    staff02_2_2_durations = define_durations(staff02_n_notes, 2, time_signature = '2/2')
    staff01_3_4_durations = define_durations(staff01_n_notes, 1, time_signature = '3/4')
    staff02_3_4_durations = define_durations(staff02_n_notes, 2, time_signature = '3/4')
    staff01_5_4_durations = define_durations(staff01_n_notes, 1, time_signature = '5/4')
    staff02_5_4_durations = define_durations(staff02_n_notes, 2, time_signature = '5/4')

    n_notes_4_4_sequence.loc[len(n_notes_4_4_sequence.index)] = (staff01_n_notes, staff01_4_4_durations,
                                                            staff02_n_notes, staff02_4_4_durations)
    n_notes_2_4_sequence.loc[len(n_notes_2_4_sequence.index)] = (staff01_n_notes, staff01_2_4_durations, 
                                                            staff02_n_notes, staff02_2_4_durations)
    n_notes_2_2_sequence.loc[len(n_notes_2_2_sequence.index)] = (staff01_n_notes, staff01_2_2_durations, 
                                                            staff02_n_notes, staff02_2_2_durations)
    n_notes_3_4_sequence.loc[len(n_notes_3_4_sequence.index)] = (staff01_n_notes, staff01_3_4_durations, 
                                                            staff02_n_notes, staff02_3_4_durations)
    n_notes_5_4_sequence.loc[len(n_notes_5_4_sequence.index)] = (staff01_n_notes, staff01_5_4_durations, 
                                                            staff02_n_notes, staff02_5_4_durations)

    for i in range(63):
        ## Aquí seleccionamos sólo la secuencia de n-notas "Previo" ([-2:])
        new_prediction = tf_model(encoded_n_notes)
        ## Aquí aplanamos el arreglo a una sola dimensión
        new_prediction = np.squeeze(new_prediction, axis=0)
        ## Aquí obtenemos las predicciones en formato de logaritmo
        log_prediction = np.log(new_prediction)
        ## Aquí obtenemos las predicciones en formato exponencial
        exp_prediction = np.exp(log_prediction)        
        ## Aquí obtenemos la relación de predicciones más probables
        prediction = exp_prediction / np.sum(exp_prediction)

        ## Aquí obtenemos los Id's las "n" predicciones más probables de las "m" que la red neuronal tiene
        ##    como tensor de salida
        n_notes_indexes = heapq.nlargest(top_n, range(len(prediction)), prediction.take)
        n_notes_index = n_notes_indexes[0]
        
        ## Aquí evitamos los ciclos de notas repetidas de inmediato
        if n_notes_index == staff01_n_notes.tolist() + staff02_n_notes.tolist():
            n_notes_index = n_notes_indexes[1]       
        ## Aquí evitamos los ciclos de 5 secuencias de n-notas repetidas
        if len(main_n_notes_id_sequence) > 8:
            if main_n_notes_id_sequence[len(main_n_notes_id_sequence)-9:-4] == main_n_notes_id_sequence[-4:] + [n_notes_index]:
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-9:-4], main_n_notes_id_sequence[-4:] + [n_notes_index])
                n_notes_index = np.random.choice(n_notes_indexes, 1, p = [0, 0.25, 0.25, 0.25, 0.25])
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-9:-4], main_n_notes_id_sequence[-4:] + [n_notes_index])
        ## Aquí evitamos los ciclos de 4 secuencias de n-notas repetidas
        if len(main_n_notes_id_sequence) > 6:
            if main_n_notes_id_sequence[len(main_n_notes_id_sequence)-7:-3] == main_n_notes_id_sequence[-3:] + [n_notes_index]:
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-7:-3], main_n_notes_id_sequence[-3:] + [n_notes_index])
                n_notes_index = np.random.choice(n_notes_indexes, 1, p = [0, 0.25, 0.25, 0.25, 0.25])
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-7:-3], main_n_notes_id_sequence[-3:] + [n_notes_index])
        if len(main_n_notes_id_sequence) > 4:
            if main_n_notes_id_sequence[len(main_n_notes_id_sequence)-5:-2] == main_n_notes_id_sequence[-2:] + [n_notes_index]:
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-5:-2], main_n_notes_id_sequence[-2:] + [n_notes_index])
                n_notes_index = np.random.choice(n_notes_indexes, 1, p = [0, 0.25, 0.25, 0.25, 0.25])
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-5:-2], main_n_notes_id_sequence[-2:] + [n_notes_index])
        if len(main_n_notes_id_sequence) > 2:
            if main_n_notes_id_sequence[len(main_n_notes_id_sequence)-3:-1] == main_n_notes_id_sequence[-1:] + [n_notes_index]:
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-3:-1], main_n_notes_id_sequence[-1:] + [n_notes_index])
                n_notes_index = np.random.choice(n_notes_indexes, 1, p = [0, 0.25, 0.25, 0.25, 0.25])
                ##print(main_n_notes_id_sequence[len(main_n_notes_id_sequence)-3:-1], main_n_notes_id_sequence[-1:] + [n_notes_index])
            
        main_n_notes_id_sequence.append(int(n_notes_index))
        
        ## Compara con la nota anterior, si es igual entonces no la agrega
        encoded_n_notes = np.zeros((1, master_n_notes.shape[0]), dtype=np.float32)
        encoded_n_notes[0][n_notes_index] = True
        encoded_n_notes = np.expand_dims(encoded_n_notes[-2:], axis=0)

        ## Aquí traemos la secuencia de n-notes inicial desde el id
        all_n_notes = master_n_notes.loc[int(n_notes_index)]
        ## Aquí separamos la secuencia completa de n-notes en las 2 manos, representados por cada pentagrama
        staff01_n_notes = all_n_notes[0:5].values
        staff02_n_notes = all_n_notes[5:10].values
        ## Aquí definimos la duración de las notas en cada secuencia de n-notes por pentagrama
        staff01_4_4_durations = define_durations(staff01_n_notes, 1)
        staff02_4_4_durations = define_durations(staff02_n_notes, 2)
        staff01_2_4_durations = define_durations(staff01_n_notes, 1, time_signature = '2/4')
        staff02_2_4_durations = define_durations(staff02_n_notes, 2, time_signature = '2/4')
        staff01_2_2_durations = define_durations(staff01_n_notes, 1, time_signature = '2/2')
        staff02_2_2_durations = define_durations(staff02_n_notes, 2, time_signature = '2/2')
        staff01_3_4_durations = define_durations(staff01_n_notes, 1, time_signature = '3/4')
        staff02_3_4_durations = define_durations(staff02_n_notes, 2, time_signature = '3/4')
        staff01_5_4_durations = define_durations(staff01_n_notes, 1, time_signature = '5/4')
        staff02_5_4_durations = define_durations(staff02_n_notes, 2, time_signature = '5/4')

        n_notes_4_4_sequence.loc[len(n_notes_4_4_sequence.index)] = (staff01_n_notes, staff01_4_4_durations,
                                                                staff02_n_notes, staff02_4_4_durations)
        n_notes_2_4_sequence.loc[len(n_notes_2_4_sequence.index)] = (staff01_n_notes, staff01_2_4_durations, 
                                                                staff02_n_notes, staff02_2_4_durations)
        n_notes_2_2_sequence.loc[len(n_notes_2_2_sequence.index)] = (staff01_n_notes, staff01_2_2_durations, 
                                                                staff02_n_notes, staff02_2_2_durations)
        n_notes_3_4_sequence.loc[len(n_notes_3_4_sequence.index)] = (staff01_n_notes, staff01_3_4_durations, 
                                                                staff02_n_notes, staff02_3_4_durations)
        n_notes_5_4_sequence.loc[len(n_notes_5_4_sequence.index)] = (staff01_n_notes, staff01_5_4_durations, 
                                                                staff02_n_notes, staff02_5_4_durations)    

    data = {}
    data["n_notes_4_4_sequence"] = json.loads(n_notes_4_4_sequence.to_json(orient = "records"))
    data["n_notes_2_4_sequence"] = json.loads(n_notes_2_4_sequence.to_json(orient = "records"))
    data["n_notes_2_2_sequence"] = json.loads(n_notes_2_2_sequence.to_json(orient = "records"))
    data["n_notes_3_4_sequence"] = json.loads(n_notes_3_4_sequence.to_json(orient = "records"))
    data["n_notes_5_4_sequence"] = json.loads(n_notes_5_4_sequence.to_json(orient = "records"))
    return data

def init():
    global tf_model
    model_root = os.getenv("AZUREML_MODEL_DIR")
    # the name of the folder in which to look for tensorflow model files
    tf_model_folder = "model"

    tf_model = tf.saved_model.load(os.path.join(model_root, tf_model_folder))

def run(raw_data):
    data = np.array(json.loads(raw_data)["data"], dtype=np.float32)

    # make prediction
    #out = tf_model(data)
    #y_hat = np.argmax(out, axis=1)

    return make_music(data)
