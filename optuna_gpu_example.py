import optuna
from optuna.integration import TFKerasPruningCallback

import argparse
import os
import pickle

# GPU Allocator
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import models
from tensorflow.keras import optimizers

# GPU memory growth & eager functions setting
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.run_functions_eagerly(True)
tf.keras.utils.set_random_seed(0)


INPUT_SIZE = 224
MAX_FILTER_SIZE = 128

def build_data_generator(trial, path):
    data_generator = ImageDataGenerator(rescale=1./255)
    return data_generator.flow_from_directory(
              path,
              target_size=(INPUT_SIZE, INPUT_SIZE),
              batch_size=32,
              class_mode='categorical')


def create_model(trial):
    model = models.Sequential()
    model.add(InputLayer(input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    # Suggest the number of layers
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        # Suggest the number of filters, kernel size, and activation
        n_filters = trial.suggest_int('n_filters_{}'.format(i), 32, MAX_FILTER_SIZE, step=32)
        kernel_size = trial.suggest_int('kernel_size_{}'.format(i), 3, 7, step=2)
        activation = trial.suggest_categorical('activation_{}'.format(i), ['relu', 'elu'])
        model.add(Conv2D(n_filters, (3, 3), 
                         activation=activation))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    # Suggest Dropout
    try_dropout = trial.suggest_int('is_dropout', 0, 1)
    if try_dropout:
        model.add(Dropout(.5))
    # Suggest dense layer node size
    dense_node = trial.suggest_int('dense_node', 32, MAX_FILTER_SIZE, step=32)
    model.add(Dense(dense_node, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model


def objective(trial):
    # Clearing the session each time is very import
    tf.keras.backend.clear_session()

    # Create a callback for pruning and early stopping
    callbacks = [
        EarlyStopping(patience=3), 
        TFKerasPruningCallback(trial, 'val_acc')
    ]
    # Call Data
    train_generator = build_data_generator(trial, 'data/train')
    validation_generator = build_data_generator(trial, 'data/valid')

    # Create and fit model
    model = create_model(trial)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['acc'])
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=0,
        callbacks=callbacks
    )
    return history.history['val_acc'][-1]


if __name__ == '__main__':
    # fetch user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_name', type=str, required=True)
    parser.add_argument('--n_trials', type=int, required=True)
    args = parser.parse_args()

    # If the first time the study has been run save the sampler
    db_exists = os.path.exists(f"{args.study_name}.db")
    if not db_exists:
        study = optuna.create_study(
            direction='maximize', 
            storage=f"sqlite:///{args.study_name}.db",
            study_name=args.study_name,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
        # Save the sampler
        with open(f'{args.study_name}_sampler.pkl', 'wb') as f:
            pickle.dump(study.sampler, f)
    else:
    # if the study has been run before load the sampler
        restored_sampler = pickle.load(open("sampler.pkl", "rb"))
        study = optuna.create_study(
            direction='maximize', 
            storage=f"sqlite:///{args.study_name}.db",
            study_name=args.study_name,
            load_if_exists=True,
            sammpler=restored_sampler
        ) 
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

