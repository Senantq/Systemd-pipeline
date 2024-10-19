#Sur VGGFace2
import cv2, os, gc, csv
import tensorflow
import numpy as np
from os import path
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D
from scipy.spatial import distance
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping

def generate_unique_vectors(num_vectors, vector_length):
    vectors_list = []
    vectors = set()
    while len(vectors_list) < num_vectors:
        vector = tuple(np.random.randint(0, 2, vector_length))
        if vector not in vectors:
            vectors.add(vector)
            vectors_list.append(vector)
    return vectors_list

def load_images_and_vectors(base_dir, proportion, target_folder, ethnie_exclue, vector_length=56, target_size=(300, 300), test_proportion=0.1):
    X_train, y_train, X_test, y_test = [], [], [], []
    vectors_dict, ethnie_dict = {}, {}
    train_counts, test_counts, test_image_counts = {}, {}, {}
    num_min_subdirs, num_min_test_subdirs = float('inf'), float('inf')

    for ethnie in os.listdir(base_dir):
        ethnie_path = os.path.join(base_dir, ethnie)
        if os.path.isdir(ethnie_path) and ethnie != ethnie_exclue:
            subdirs = [d for d in os.listdir(ethnie_path) if os.path.isdir(os.path.join(ethnie_path, d))]
            num_subdirs = len(subdirs)
            if ethnie == target_folder:
                num_to_keep = max(1, int(num_subdirs * proportion)) if proportion > 0 else 0
                num_min_subdirs = min(num_min_subdirs, num_to_keep)
            num_min_test_subdirs = min(num_min_test_subdirs, num_subdirs)

    for ethnie in os.listdir(base_dir):
        ethnie_path = os.path.join(base_dir, ethnie)
        if os.path.isdir(ethnie_path) and ethnie != ethnie_exclue:
            subdirs = [d for d in os.listdir(ethnie_path) if os.path.isdir(os.path.join(ethnie_path, d))]

            if ethnie == target_folder:
                if proportion > 0:
                    selected_subdirs = np.random.choice(subdirs, num_min_subdirs, replace=False)
                else:
                    selected_subdirs = []
                current_test_proportion = test_proportion  # Utiliser la proportion spécifiée pour l'ethnie cible
            else:
                selected_subdirs = subdirs
                current_test_proportion = 0.1  # Utiliser 10% pour les autres ethnies

            train_counts[ethnie] = 0
            test_counts[ethnie] = 0
            test_image_counts[ethnie] = 0
            vectors = generate_unique_vectors(len(selected_subdirs), vector_length)

            for subdir, vector in zip(selected_subdirs, vectors):
                vectors_dict[subdir] = vector
                ethnie_dict[subdir] = ethnie
                subdir_path = os.path.join(ethnie_path, subdir)
                subdir_images_train, subdir_images_test = [], []

                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    if file_path.lower().endswith(('jpg')):
                        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, target_size) / 255.0
                            if np.random.rand() < current_test_proportion:
                                subdir_images_test.append(img)
                            else:
                                subdir_images_train.append(img)

                if subdir_images_test:
                    X_test.extend(subdir_images_test)
                    y_test.extend([vector] * len(subdir_images_test))
                    test_counts[ethnie] += 1
                    test_image_counts[ethnie] += len(subdir_images_test)
                if subdir_images_train:
                    X_train.extend(subdir_images_train)
                    y_train.extend([vector] * len(subdir_images_train))
                    train_counts[ethnie] += 1
            print(f"Ethnie: {ethnie} - Sous-dossiers conservés dans le dataset d'entraînement: {train_counts[ethnie]}, dataset de test: {test_counts[ethnie]}")
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), vectors_dict, ethnie_dict, test_image_counts

def save_mean(y_train, prediction, run, nb_cc_value, ethnie, proportion, vectors_dict, l1_value, test_prop):
    cpt_correct = 0
    with open('/home/senantq/ORE/Analyses_stats/Codes/local_results/ID_results/resultsEAAf.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(prediction)):
            correct_vector_distance = distance.euclidean(prediction[i], y_train[i])
            other_distances = [distance.euclidean(prediction[i], y_train[j]) for j in range(len(y_train)) if j != i]
            if correct_vector_distance <= min(other_distances):
                correct = 1
                cpt_correct += 1
            else:
                correct = 0

            mse_loss = np.mean((prediction[i] - y_train[i]) ** 2)
            subdir = None
            for key, value in vectors_dict.items():
                if np.array_equal(value, y_train[i]):
                    subdir = key
                    break
            if subdir is None:
                raise ValueError("Le vecteur n'a pas été trouvé dans le dictionnaire vectors_dict.")
            writer.writerow([run, nb_cc_value, ethnie, proportion, subdir, correct, mse_loss, l1_value, test_prop])
    print("nb correct: ", cpt_correct, 'accuracy: ', np.round((cpt_correct / len(prediction)), 3))
def create_conv(nb_cc_value, l1_value):
    model = Sequential()
    model.add(tensorflow.keras.layers.RandomFlip(mode="horizontal"))
    model.add(tensorflow.keras.layers.RandomZoom(height_factor = (0,-0.15), width_factor=(0,-0.15)))
    model.add(tensorflow.keras.layers.RandomContrast(factor = (0.1)))
    model.add(Conv2D(16, (3,3), activation = 'relu', kernel_regularizer=l1(l1_value)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation = 'relu', kernel_regularizer=l1(l1_value)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_regularizer=l1(l1_value)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation = 'relu', kernel_regularizer=l1(l1_value)))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Conv2D(256, (3,3), activation = 'relu', kernel_regularizer=l1(l1_value)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(nb_cc_value, activation='relu', kernel_regularizer=l1(l1_value)))
    model.add(Dense(56, activation = 'sigmoid'))
    model.build((None,150,150,1))
    lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.4, staircase=False)
    optimizer = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule, momentum = 0.9)
    model.compile(loss= ['mse'], optimizer = optimizer, metrics = ['mse'])
    return model
# -------------------------------------------------Initialization

ethnies = ['East_Asians', 'Afro_Americans']
ethnie_exclue = 'Caucasians'
base_directory = '/home/senantq/ORE/Analyses_stats/Codes/VGG ORE'
prop = [10] # proportion d'invidus montrés au sein de l'ethnie
l1_values = [0.1]
epoques = 20
#train_indiv_prop = [1, 60, 90, 95, 99.9]
test_proportion = [97] # proportion de fois où un individu est>
#(en phase d'entraînement = 1-test_prop)
# ne pas aller plus bas que 10% de test_prop
nb_cc = [2]
run = 1
if not path.exists('/home/senantq/ORE/Analyses_stats/Codes/local_results/ID_results/resultsEAAf.csv'):
    with open('/home/senantq/ORE/Analyses_stats/Codes/local_results/ID_results/resultsEAAf.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'nb_cc','ethnie','prop_indiv_per_ethnicity', 'subdir', 'correct', 'MSE', 'l1_value', 'train_prop_per_indiv'])

for nb_cc_value in nb_cc:
    for ethnie in ethnies:
        for proportion in prop:
            proportion = proportion/100.
            for l1_value in l1_values:
                for test_prop in test_proportion:
                    test_prop = test_prop/100.
                    X_train, y_train, X_test, y_test, vectors_dict, ethnie_dict, test_image_counts = load_images_and_vectors(target_folder=ethnie, base_dir = base_directory, proportion=proportion, ethnie_exclue=ethnie_exclue, target_size=(150,150), test_proportion=test_prop)
                    train_prop = 1 - test_prop
                    print('nb_cc: ', nb_cc_value, 'l1_value:', l1_value, '; prop_indiv_per_ethnicity:', proportion, '; ethnie : ', ethnie, 'train prop:', train_prop)

                    # ---------------------------------------TRAIN
                    model = create_conv(nb_cc_value, l1_value)
                    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=epoques, batch_size=48, shuffle=True, verbose=1)
                    # ---------------------------------------TEST
                    prediction = model.predict(X_test, batch_size=64)
                    save_mean(y_test, prediction, run, nb_cc_value, ethnie, proportion, vectors_dict, l1_value, train_prop)
                    # Nettoyage
                    del X_train, y_train, X_test, y_test, vectors_dict, ethnie_dict, test_image_counts, model, history, prediction
                    tensorflow.keras.backend.clear_session()
                    gc.collect()
