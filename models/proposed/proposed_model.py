import pandas as pd
import os
import numpy as np

import collections
import gc

import tensorflow as tf


from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

import warnings

warnings.filterwarnings("ignore")

data_path = "../../data/intermediate/"

type_of_ner = "new"

x_train_lstm = pd.read_pickle(data_path + type_of_ner + "_x_train.pkl")
x_dev_lstm = pd.read_pickle(data_path + type_of_ner + "_x_dev.pkl")
x_test_lstm = pd.read_pickle(data_path + type_of_ner + "_x_test.pkl")

y_train = pd.read_pickle(data_path + type_of_ner + "_y_train.pkl")
y_dev = pd.read_pickle(data_path + type_of_ner + "_y_dev.pkl")
y_test = pd.read_pickle(data_path + type_of_ner + "_y_test.pkl")


ner_word2vec = pd.read_pickle(
    data_path + type_of_ner + "_ner_word2vec_limited_dict.pkl"
)
ner_fasttext = pd.read_pickle(
    data_path + type_of_ner + "_ner_fasttext_limited_dict.pkl"
)
ner_concat = pd.read_pickle(data_path + type_of_ner + "_ner_combined_limited_dict.pkl")

train_ids = pd.read_pickle(data_path + type_of_ner + "_train_ids.pkl")
dev_ids = pd.read_pickle(data_path + type_of_ner + "_dev_ids.pkl")
test_ids = pd.read_pickle(data_path + type_of_ner + "_test_ids.pkl")

# Reset Keras Session
def reset_keras(model):
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass

    gc.collect()  # if it's done something you should see a number being outputted


def make_prediction_cnn(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i >= 0.5 else 0 for i in probs]
    return probs, y_pred


def save_scores_cnn(
    predictions,
    probs,
    ground_truth,
    embed_name,
    problem_type,
    iteration,
    hidden_unit_size,
    sequence_name,
    type_of_ner,
):

    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc = accuracy_score(ground_truth, predictions)
    F1 = f1_score(ground_truth, predictions)

    result_dict = {}
    result_dict["auc"] = auc
    result_dict["auprc"] = auprc
    result_dict["acc"] = acc
    result_dict["F1"] = F1

    result_path = "results/cnn/"
    file_name = str(sequence_name) + "-" + str(hidden_unit_size) + "-" + embed_name
    file_name = (
        file_name
        + "-"
        + problem_type
        + "-"
        + str(iteration)
        + "-"
        + type_of_ner
        + "-cnn-.p"
    )
    pd.to_pickle(result_dict, os.path.join(result_path, file_name))

    print(auc, auprc, acc, F1)


def print_scores_cnn(
    predictions,
    probs,
    ground_truth,
    model_name,
    problem_type,
    iteration,
    hidden_unit_size,
):
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc = accuracy_score(ground_truth, predictions)
    F1 = f1_score(ground_truth, predictions)

    print("AUC: ", auc, "AUPRC: ", auprc, "F1: ", F1)


def get_subvector_data(size, embed_name, data):
    if embed_name == "concat":
        vector_size = 200
    else:
        vector_size = 100

    x_data = {}
    for k, v in data.items():
        number_of_additional_vector = len(v) - size
        vector = []
        for i in v:
            vector.append(i)
        if number_of_additional_vector < 0:
            number_of_additional_vector = np.abs(number_of_additional_vector)

            temp = vector[:size]
            for i in range(0, number_of_additional_vector):
                temp.append(np.zeros(vector_size))
            x_data[k] = np.asarray(temp)
        else:
            x_data[k] = np.asarray(vector[:size])

    return x_data


def avg_ner_model(layer_name, number_of_unit, embedding_name):

    if embedding_name == "concat":
        input_dimension = 200
    else:
        input_dimension = 100

    sequence_input = tf.keras.layers.Input(shape=(24, 104))

    input_avg = tf.keras.layers.Input(shape=(input_dimension,), name="avg")
    #     x_1 = Dense(256, activation='relu')(input_avg)
    #     x_1 = Dropout(0.3)(x_1)

    if layer_name == "GRU":
        x = tf.keras.layers.GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = tf.keras.layers.LSTM(number_of_unit)(sequence_input)

    x = tf.keras.layers.Concatenate()([x, input_avg])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    logits_regularizer = tf.keras.regularizers.L2(0.01)

    preds = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        use_bias=False,
        kernel_initializer=tf.initializers.GlorotUniform(),
        kernel_regularizer=logits_regularizer,
    )(x)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=0.01)
    model = tf.keras.models.Model(inputs=[sequence_input, input_avg], outputs=preds)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])

    return model


def proposedmodel(layer_name, number_of_unit, embedding_name, ner_limit, num_filter):
    if embedding_name == "concat":
        input_dimension = 200
    else:
        input_dimension = 100

    sequence_input = tf.keras.layers.Input(shape=(24, 104))

    input_img = tf.keras.layers.Input(
        shape=(ner_limit, input_dimension), name="cnn_input"
    )

    text_conv1d = tf.keras.layers.Conv1D(
        filters=num_filter,
        kernel_size=3,
        padding="valid",
        strides=1,
        dilation_rate=1,
        activation="relu",
        kernel_initializer=tf.initializers.GlorotUniform(),
    )(input_img)

    text_conv1d = tf.keras.layers.Conv1D(
        filters=num_filter * 2,
        kernel_size=3,
        padding="valid",
        strides=1,
        dilation_rate=1,
        activation="relu",
        kernel_initializer=tf.initializers.GlorotUniform(),
    )(text_conv1d)

    text_conv1d = tf.keras.layers.Conv1D(
        filters=num_filter * 3,
        kernel_size=3,
        padding="valid",
        strides=1,
        dilation_rate=1,
        activation="relu",
        kernel_initializer=tf.initializers.GlorotUniform(),
    )(text_conv1d)

    # concat_conv = keras.layers.Concatenate()([text_conv1d, text_conv1d_2, text_conv1d_3])
    text_embeddings = tf.keras.layers.GlobalMaxPooling1D()(text_conv1d)
    # text_embeddings = Dense(128, activation="relu")(text_embeddings)

    if layer_name == "GRU":
        x = tf.keras.layers.GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = tf.keras.layers.LSTM(number_of_unit)(sequence_input)

    # concatenated = keras.layers.Concatenate()([x, text_embeddings])
    concatenated = tf.keras.layers.Concatenate()([x, text_embeddings])

    concatenated = tf.keras.layers.Dense(512, activation="relu")(concatenated)
    concatenated = tf.keras.layers.Dropout(0.2)(concatenated)
    # concatenated = Dense(256, activation='relu')(concatenated)
    # concatenated = Dense(512, activation='relu')(concatenated)

    # concatenated = Dense(512, activation='relu')(concatenated)
    logits_regularizer = tf.keras.regularizers.L2(0.01)
    preds = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        use_bias=False,
        kernel_initializer=tf.initializers.GlorotUniform(),
        kernel_regularizer=logits_regularizer,
    )(concatenated)

    # opt = Adam(lr=1e-4, decay = 0.01)

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=0.01)

    # opt = Adam(lr=0.001)

    model = tf.keras.models.Model(inputs=[sequence_input, input_img], outputs=preds)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])

    return model


embedding_types = ["word2vec", "fasttext", "concat"]
embedding_dict = [ner_word2vec, ner_fasttext, ner_concat]

target_problems = ["mort_hosp", "mort_icu", "los_3", "los_7"]

num_epoch = 100
model_patience = 5
monitor_criteria = "val_loss"
# monitor_criteria = 'val_acc'
batch_size = 64

filter_number = 32
ner_representation_limit = 64
activation_func = "relu"

sequence_models = ["LSTM", "GRU"]
sequence_hidden_units = [128, 256]


maxiter = 11
for sequence_model in sequence_models:
    print("Layer: ", sequence_model)
    for sequence_hidden_unit in sequence_hidden_units:
        print("Hidden unit: ", sequence_hidden_unit)
        for embed_dict, embed_name in zip(embedding_dict, embedding_types):
            print("Embedding: ", embed_name)
            print("=============================")

            temp_train_ner = dict((k, embed_dict[k]) for k in train_ids)
            temp_dev_ner = dict((k, embed_dict[k]) for k in dev_ids)
            temp_test_ner = dict((k, embed_dict[k]) for k in test_ids)

            x_train_dict = get_subvector_data(
                ner_representation_limit, embed_name, temp_train_ner
            )
            x_dev_dict = get_subvector_data(
                ner_representation_limit, embed_name, temp_dev_ner
            )
            x_test_dict = get_subvector_data(
                ner_representation_limit, embed_name, temp_test_ner
            )

            x_train_dict_sorted = collections.OrderedDict(sorted(x_train_dict.items()))
            x_dev_dict_sorted = collections.OrderedDict(sorted(x_dev_dict.items()))
            x_test_dict_sorted = collections.OrderedDict(sorted(x_test_dict.items()))

            x_train_ner = np.asarray(list(x_train_dict_sorted.values()))
            x_dev_ner = np.asarray(list(x_dev_dict_sorted.values()))
            x_test_ner = np.asarray(list(x_test_dict_sorted.values()))

            for iteration in range(1, maxiter):
                print("Iteration number: ", iteration)

                for each_problem in target_problems:
                    print("Problem type: ", each_problem)
                    print("__________________")

                    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
                        monitor=monitor_criteria, patience=model_patience
                    )

                    best_model_name = (
                        str(ner_representation_limit)
                        + "-basiccnn1d-"
                        + str(embed_name)
                        + "-"
                        + str(each_problem)
                        + "-"
                        + "best_model.hdf5"
                    )

                    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        best_model_name,
                        monitor=monitor_criteria,
                        verbose=1,
                        save_best_only=True,
                        mode="min",
                        period=1,
                    )

                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor=monitor_criteria,
                        factor=0.2,
                        patience=2,
                        min_lr=0.00001,
                        epsilon=1e-4,
                        mode="min",
                    )

                    callbacks = [early_stopping_monitor, checkpoint]

                    # model = textCNN(sequence_model, sequence_hidden_unit, embed_name, ner_representation_limit)
                    # model = avg_ner_model(sequence_model, sequence_hidden_unit, embed_name)
                    model = proposedmodel(
                        sequence_model,
                        sequence_hidden_unit,
                        embed_name,
                        ner_representation_limit,
                        filter_number,
                    )
                    model.fit(
                        [x_train_lstm, x_train_ner],
                        y_train[each_problem],
                        epochs=num_epoch,
                        verbose=1,
                        validation_data=([x_dev_lstm, x_dev_ner], y_dev[each_problem]),
                        callbacks=callbacks,
                        batch_size=batch_size,
                    )

                    model.load_weights(best_model_name)

                    probs, predictions = make_prediction_cnn(
                        model, [x_test_lstm, x_test_ner]
                    )
                    print_scores_cnn(
                        predictions,
                        probs,
                        y_test[each_problem],
                        embed_name,
                        each_problem,
                        iteration,
                        sequence_hidden_unit,
                    )
                    save_scores_cnn(
                        predictions,
                        probs,
                        y_test[each_problem],
                        embed_name,
                        each_problem,
                        iteration,
                        sequence_hidden_unit,
                        sequence_model,
                        type_of_ner,
                    )

                    reset_keras(model)
                    # del model
                    tf.compat.v1.keras.backend.clear_session()
                    gc.collect()
