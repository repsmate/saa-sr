import argparse
import os
from glob import glob
from typing import Any

from torch import threshold
from deep_speaker.audio import Audio
from deep_speaker.utils import ensures_dir
import numpy as np

from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

##
# 1) define a parsing funtion to extract speaker_id from file-paths within the audio directory
# 2) apply it to the main function
 
def choose_rand_sample(fname_by_spk_id: dict):
    speaker_ids = list(fname_by_spk_id.keys())
    rand_spk_id = np.random.choice(speaker_ids)
    rand_sample = np.random.choice(fname_by_spk_id[rand_spk_id])
    return rand_sample, rand_spk_id

def choose_rand_sample_by_spk_id(fname_by_spk_id: dict, spk_id: str):
    rand_sample = np.random.choice(fname_by_spk_id[spk_id])
    return rand_sample

def update_confusion_matrix_dict(
                            confusion_mtx_dict: dict, 
                            are_samples_similar: bool, 
                            label_1: str, 
                            label_2: str):
    if are_samples_similar and label_1 == label_2:
        confusion_mtx_dict['tp'] += 1
        
    if (not are_samples_similar) and label_1 != label_2:
        confusion_mtx_dict['tn'] += 1
    
    if (not are_samples_similar) and label_1 == label_2:
        confusion_mtx_dict['fn'] += 1

    if are_samples_similar and label_1 != label_2:
        confusion_mtx_dict['fp'] += 1
    return confusion_mtx_dict

def init_confusion_mtx_dict():
    confusion_mtx_dict = {
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0
    }
    return confusion_mtx_dict

def print_results(confusion_mtx_dict: dict):
    tp = confusion_mtx_dict['tp']
    tn = confusion_mtx_dict['tn']
    fp = confusion_mtx_dict['fp']
    fn = confusion_mtx_dict['fn']
    true_preds = confusion_mtx_dict['tp'] + confusion_mtx_dict['tn']
    false_preds = confusion_mtx_dict['fp'] + confusion_mtx_dict['fn']
    all_preds = true_preds + false_preds
    accuracy = true_preds / all_preds
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    print(f"Number of predictions: {all_preds} \
        \n\tAccuracy: {round(accuracy, 2)}% \
        \n\tRecall: {round(recall, 2)}% \
        \n\tPrecision: {round(precision, 2)}% \
        \n\tNPV: {round(npv, 2)}")

def init_sr_model(model_path: str):
    model = DeepSpeakerModel()
    model.m.load_weights(model_path, by_name=True)
    return model

def infer_sr_is_positive(model: DeepSpeakerModel, test_sample_path: str, positive_sample_path: str, negative_sample_path) -> bool:
    mfcc_test = sample_from_mfcc(read_mfcc(test_sample_path, SAMPLE_RATE), NUM_FRAMES)
    mfcc_pos = sample_from_mfcc(read_mfcc(positive_sample_path, SAMPLE_RATE), NUM_FRAMES)
    mfcc_neg = sample_from_mfcc(read_mfcc(negative_sample_path, SAMPLE_RATE), NUM_FRAMES)
    predict_test = model.m.predict(np.expand_dims(mfcc_test, axis=0))
    predict_pos = model.m.predict(np.expand_dims(mfcc_pos, axis=0))
    predict_neg = model.m.predict(np.expand_dims(mfcc_neg, axis=0))
    score_pos = batch_cosine_similarity(predict_test, predict_pos)
    score_neg = batch_cosine_similarity(predict_test, predict_neg)
    return score_pos > score_neg

def parse_spk_id(audio_file_path: str):
    fname = os.path.basename(audio_file_path)
    speaker_id, ctx_id, channel, vad_id, window_id = fname.replace('.wav', '').split('_')
    return channel

def main(audio_dir: str, sr_model_path: str, spk_id_parsing_fn: Any, ext: str = '.wav', n_test_iterations: int = 10000):
    audio_files = glob(os.path.join(audio_dir, f'*{ext}'))
    fname_by_spk_id = {
    }
    for audio_file in audio_files:
        spk_id = spk_id_parsing_fn(audio_file)
        if spk_id not in fname_by_spk_id.keys():
            fname_by_spk_id[spk_id] = [audio_file]
        else:
            fname_by_spk_id[spk_id].append(audio_file)
    
    
    speaker_ids = list(fname_by_spk_id.keys())
    assert len(speaker_ids) == 2, f'Too many speakers: {speaker_ids}. Must be only 2.'

    bool_to_spk_id_dict = {
        True: speaker_ids[0],
        False: speaker_ids[1]
    }

    sr_model = init_sr_model(sr_model_path)
    confusion_mtx_dict = init_confusion_mtx_dict() 
    for _ in range(n_test_iterations):
        rand_sample_test, spk_id_test = choose_rand_sample(fname_by_spk_id)
        rand_sample_pos = choose_rand_sample_by_spk_id(fname_by_spk_id, bool_to_spk_id_dict[True])
        rand_sample_neg = choose_rand_sample_by_spk_id(fname_by_spk_id, bool_to_spk_id_dict[False])
        #rand_sample2, rand_spk_id2 = choose_rand_sample(fname_by_spk_id)
        

        is_positive_sample = infer_sr_is_positive(sr_model, rand_sample_test, rand_sample_pos, rand_sample_neg)
        #are_samples_similar = similarity_score >= threshold

        confusion_mtx_dict = update_confusion_matrix_dict(
                                confusion_mtx_dict=confusion_mtx_dict, 
                                are_samples_similar=is_positive_sample, 
                                label_1=spk_id_test, 
                                label_2=bool_to_spk_id_dict[is_positive_sample]
                                )
    print_results(confusion_mtx_dict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='', action='store', type=str, required=True, help='Audio file directory containg .wav files.')
    parser.add_argument('--model_path', default='', action='store', type=str, required=True, help='.h5 model path')
    parser.add_argument('--n_tests', default=10000, action='store', type=int, required=False, help='Number of test iterations')
    args = parser.parse_args()
    main(
        audio_dir=args.audio_dir, 
        sr_model_path=args.model_path, 
        n_test_iterations=args.n_tests,
        spk_id_parsing_fn=parse_spk_id,
        ext = '.wav'
        )