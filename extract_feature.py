# !/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# This code is an adaptation from Toni Heittola's code [task1 baseline dcase 2018](https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task1/)
# Copyright Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah, 2018, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
import dcase_util
import sys
import numpy
import os
import random
import pickle
import pandas

import tensorflow as tf
from keras import backend as K
import keras

#from evaluation_measures import get_f_measure_by_class, event_based_evaluation, segment_based_evaluation
from evaluation_measures import get_f_measure_by_class, event_based_evaluation, event_based_evaluation_df
from Dataset_dcase2018 import DCASE2018_Task4_DevelopmentSet

dcase_util.utils.setup_logging(logging_file='task4.log')
print(keras.__version__)

random.seed(10)
numpy.random.seed(42)

tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)


def main(parameters):
    log = dcase_util.ui.ui.FancyLogger()
    log.title('DCASE2018 / Task4')

    overwirte_preprocessing = False
    overwrite_learning = True
    overwrite_testing = True
    overwrite_evaluation = True

    # =====================================================================
    # Parameters
    # =====================================================================
    # Process parameters
    param = dcase_util.containers.DCASEAppParameterContainer(
        parameters,
        path_structure={
            'FEATURE_EXTRACTOR': [
                'DATASET',
                'FEATURE_EXTRACTOR'
            ],
            'FEATURE_NORMALIZER': [
                'DATASET',
                'FEATURE_EXTRACTOR'
            ],
            'LEARNER': [
                'DATASET',
                'FEATURE_EXTRACTOR',
                'FEATURE_NORMALIZER',
                'FEATURE_SEQUENCER',
                'LEARNER'
            ],
            'RECOGNIZER': [
                'DATASET',
                'FEATURE_EXTRACTOR',
                'FEATURE_NORMALIZER',
                'FEATURE_SEQUENCER',
                'LEARNER',
                'RECOGNIZER'
            ],
        }
    ).process()

    # Make sure all system paths exists
    dcase_util.utils.Path().create(
        paths=list(param['path'].values())
    )

    # Initialize
    keras_model_first_pass = None
    keras_model_second_pass = None

    # =====================================================================
    # Dataset
    # =====================================================================
    # Get dataset and initialize it
    db = DCASE2018_Task4_DevelopmentSet(included_content_types=['all'],
                                        local_path="",
                                        data_path=param.get_path('path.dataset'),
                                        audio_paths=[
                                            os.path.join("dataset", "audio", "train"),
                                            #os.path.join("dataset", "audio", "train", "unlabel_in_domain"),
                                            #os.path.join("dataset", "audio", "train", "unlabel_out_of_domain"),
                                            os.path.join("dataset", "audio", "test"),
                                            os.path.join("dataset", "audio", "eval")
                                        ]
                                        ).initialize()

    # Active folds
    folds = db.folds(
        mode=param.get_path('dataset.parameters.evaluation_mode')
    )
    active_fold_list = param.get_path('dataset.parameters.fold_list')
    if active_fold_list:
        folds = list(set(folds).intersection(active_fold_list))

    # =====================================================================
    # Feature extraction stage
    # =====================================================================
    if param.get_path('flow.feature_extraction'):
        log.section_header('Feature Extraction / Train material')

        # Prepare feature extractor
        mel_extractor = dcase_util.features.MelExtractor(
            **param.get_path('feature_extractor.parameters.mel')
        )

        # Loop over all audio files in the dataset and extract features for them.
        # for audio_filename in db.audio_files:
        for audio_filename in db.audio_files:
            # Get filename for feature data from audio filename
            feature_filename = dcase_util.utils.Path(
                path=audio_filename
            ).modify(
                path_base=param.get_path('path.application.feature_extractor'),
                filename_extension='.cpickle'
            )

            if not os.path.isfile(feature_filename) or overwirte_preprocessing:
                log.line(
                    data=os.path.split(audio_filename)[1],
                    indent=2
                )

                # Load audio data
                audio = dcase_util.containers.AudioContainer().load(
                    filename=audio_filename,
                    mono=True,
                    fs=param.get_path('feature_extractor.fs')
                )

                # Extract features and store them into FeatureContainer, and save it to the disk
                dcase_util.containers.FeatureContainer(
                    data=mel_extractor.extract(audio.data),
                    time_resolution=param.get_path('feature_extractor.hop_length_seconds')
                ).save(
                    filename=feature_filename
                )

        log.foot()

    # =====================================================================
    # Feature normalization stage
    # =====================================================================

    if param.get_path('flow.feature_normalization'):
        log.section_header('Feature Normalization')

        # Get filename for the normalization factors
        features_norm_filename = os.path.join(
            param.get_path('path.application.feature_normalizer'),
            'normalize_values.cpickle'
        )

        if not os.path.isfile(features_norm_filename) or overwirte_preprocessing:
            normalizer = dcase_util.data.Normalizer(
                filename=features_norm_filename
            )

            #  Loop through all training data, two train folds
            for fold in folds:
                for filename in db.train(fold=fold).unique_files:
                    # Get feature filename
                    feature_filename = dcase_util.utils.Path(
                        path=filename
                    ).modify(
                        path_base=param.get_path('path.application.feature_extractor'),
                        filename_extension='.cpickle',
                    )

                    # Load feature matrix
                    features = dcase_util.containers.FeatureContainer().load(
                        filename=feature_filename
                    )

                    # Accumulate statistics
                    normalizer.accumulate(
                        data=features.data
                    )

            # Finalize and save
            normalizer.finalize().save()

        log.foot()

    # Create processing chain for features
    feature_processing_chain = dcase_util.processors.ProcessingChain()
    for chain in param.get_path('feature_processing_chain'):
        processor_name = chain.get('processor_name')
        init_parameters = chain.get('init_parameters', {})

        # Inject parameters
        if processor_name == 'dcase_util.processors.NormalizationProcessor':
            init_parameters['filename'] = features_norm_filename

        if init_parameters.get('enable') is None or init_parameters.get('enable') is True:
            feature_processing_chain.push_processor(
                processor_name=processor_name,
                init_parameters=init_parameters,
            )


def data_generator(items, feature_path, many_hot_encoder, feature_processing_chain, batch_size=1, shuffle=True, mode='weak'):
    """ Transform MetaDataContainer into batches of data

    Parameters
    ----------

    items : MetaDataContainer, items to be generated

    feature_path : String, base path where features are stored

    many_hot_encoder : ManyHotEncoder, class to encode data

    feature_processing_chain : ProcessingChain, chain to process data

    batch_size : int, size of the batch to be returned

    shuffle : bool, shuffle the items before creating the batch

    mode : "weak" or "strong", indicate to return labels as tags (1/file) or event_labels (1/frame)

    Return
    ------

    (batch_X, batch_y): generator, arrays containing batches of data.

    """
    while True:
        batch_X = []
        batch_y = []
        if shuffle:
            random.shuffle(items)
        for item in items:
            # Get feature filename
            feature_filename = dcase_util.utils.Path(
                path=item.filename
            ).modify(
                path_base=feature_path,
                filename_extension='.cpickle',
            )

            features = feature_processing_chain.process(
                filename=feature_filename
            )
            input_data = features.data.reshape(features.shape[:-1]).T

            # Target
            targets = item.tags
            targets = many_hot_encoder.encode(targets, length_frames=1).data.flatten()
            if mode == "strong":
                targets = numpy.repeat(targets.reshape((1,) + targets.shape), input_data.shape[0], axis=0)

            if batch_size == 1:
                batch_X = input_data.reshape((1,) + input_data.shape)
                batch_y = targets.reshape((1,) + targets.shape)
                yield batch_X, batch_y
            else:
                batch_X.append(input_data)
                batch_y.append(targets)
                if len(batch_X) == batch_size and len(batch_y) == batch_size:
                    yield numpy.array(batch_X), numpy.array(batch_y)

                    batch_X = []
                    batch_y = []



if __name__ == "__main__":
    # Read parameters file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task4_crnn.yaml'
    )

    try:
        sys.exit(main(parameters))
    except (ValueError, IOError) as e:
        sys.exit(e)
