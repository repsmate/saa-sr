import argparse
import os
from glob import glob
from typing import Any, List, Tuple
import pickle
from tqdm import tqdm

from deep_speaker.audio import Audio
from deep_speaker.utils import ensures_dir
import numpy as np

from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
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
                            is_positive_sample: bool, 
                            label_1: str, 
                            label_2: str):
    if is_positive_sample and label_1 == label_2:
        confusion_mtx_dict['tp'] += 1

    if is_positive_sample and label_1 != label_2:
        confusion_mtx_dict['fp'] += 1

    if (not is_positive_sample) and label_1 == label_2:
        confusion_mtx_dict['tn'] += 1
    
    if (not is_positive_sample) and label_1 != label_2:
        confusion_mtx_dict['fn'] += 1



    return confusion_mtx_dict

def init_confusion_mtx_dict():
    confusion_mtx_dict = {
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0
    }
    return confusion_mtx_dict

def get_results_dict(confusion_mtx_dict: dict):
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
    result_dict = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'npv': npv,
    }
    return result_dict

def init_sr_model(model_path: str):
    model = DeepSpeakerModel()
    model.m.load_weights(model_path, by_name=True)
    return model

def infer_on_file(model: DeepSpeakerModel, fpath: str):
    mfcc = sample_from_mfcc(read_mfcc(fpath, SAMPLE_RATE), NUM_FRAMES)
    embeddings = model.m.predict(np.expand_dims(mfcc, axis=0))
    return embeddings

def compute_mfcc_from_files_batch(fpaths: List[str]):
    mfccs=[]
    for fpath in fpaths:
        mfcc = sample_from_mfcc(read_mfcc(fpath, SAMPLE_RATE), NUM_FRAMES)
        mfccs.append(mfcc)
    return np.array(mfccs)

def get_concatenated_samples_and_indexes(accumulated_samples: List[Tuple[str, str]]):
    test_paths = [test_path for _, test_path, pos_path, neg_path in accumulated_samples]
    return test_paths
    # all_samples = np.array(test_paths + pos_paths + neg_paths)

    # indexes_test = np.array([False for _ in range(len(all_samples))])
    # indexes_test[:len(test_paths)] = True

    # indexes_pos =  np.array([False for _ in range(len(all_samples))])
    # indexes_pos[len(test_paths):len(test_paths)+len(pos_paths)] = True

    # indexes_neg =  np.array([False for _ in range(len(all_samples))])
    # indexes_neg[len(test_paths)+len(pos_paths):] = True

    # samples_dict = {
    #     'samples': all_samples,
    #     'indexes': {
    #                 'test': indexes_test,
    #                 'pos': indexes_pos,
    #                 'neg': indexes_neg
    #             }
    # }

    return samples_dict

def similarity_score(voice_features: np.ndarray, ref_voice_features: np.ndarray) -> List[float]:
    return np.mean(np.dot(voice_features, ref_voice_features.T), axis=1)

def infer_sr_is_positive(
                model: DeepSpeakerModel, 
                accumulated_samples: List[Tuple[str, str]],
                ref_embs_dict: dict,
                bool_to_spk_id_dict: dict
                ) -> List[bool]:
    test_paths = [test_path for _, test_path in accumulated_samples]
    mfccs = compute_mfcc_from_files_batch(test_paths)
    #print('ceva mfcc shape', np.shape(mfccs))
    embeddings = model.m.predict(mfccs)
    scores_pos = similarity_score(
                        embeddings,
                        ref_embs_dict[bool_to_spk_id_dict[True]]
                        )
    scores_neg = similarity_score(
                        embeddings,
                        ref_embs_dict[bool_to_spk_id_dict[False]]
                        )
    #print('ceva shape', np.shape(mfccs), \
        # np.shape(embeddings), \
        # np.shape(ref_embs_dict[bool_to_spk_id_dict[True]]), np.shape(ref_embs_dict[bool_to_spk_id_dict[True]]), \
        # np.shape(scores_neg), np.shape(scores_pos))
    return scores_pos > scores_neg

def parse_spk_id(audio_file_path: str):
    fname = os.path.basename(audio_file_path)
    speaker_id, ctx_id, channel, vad_id, window_id = fname.replace('.wav', '').split('_')
    return channel

def build_initial_ref_embs(fname_by_spk_id_dict: dict, model: DeepSpeakerModel, n_ref_embs: int = 64):
    spk_ids = list(fname_by_spk_id_dict.keys())
    embeddings_dict = {spk_id : [] for spk_id in spk_ids}
    for spk_id in spk_ids:
        rand_spk_paths = np.random.choice(fname_by_spk_id_dict[spk_id], size=n_ref_embs, replace=False)
        for rand_spk_path in rand_spk_paths:
            embedding = infer_on_file(model, rand_spk_path)
            embeddings_dict[spk_id].append(embedding)
        embeddings_dict[spk_id] = np.concatenate(embeddings_dict[spk_id])
    return embeddings_dict

def accumulate_samples(accumulated_samples: List[Tuple[str, str]], fname_by_spk_id: dict):
    rand_sample_test, spk_id_test = choose_rand_sample(fname_by_spk_id)
    accumulated_samples.append((spk_id_test, rand_sample_test))
    return accumulated_samples

def infer_and_compute_results(
                        sr_model: DeepSpeakerModel,
                        confusion_mtx_dict: dict,
                        accumulated_samples: List[Tuple[str, str]],
                        bool_to_spk_id_dict: dict,
                        ref_embs_dict: dict,
                    ):
    is_positive_predictions = infer_sr_is_positive(sr_model, accumulated_samples, ref_embs_dict, bool_to_spk_id_dict)
    #are_samples_similar = similarity_score >= threshold
    for (spk_id_test, _), is_positive_prediction in zip(accumulated_samples, is_positive_predictions):
        confusion_mtx_dict = update_confusion_matrix_dict(
                                confusion_mtx_dict=confusion_mtx_dict, 
                                is_positive_sample=is_positive_prediction, 
                                label_1=spk_id_test, 
                                label_2=bool_to_spk_id_dict[is_positive_prediction]
                                )
    return confusion_mtx_dict

def main(
        audio_dir: str, 
        sr_model_path: str, 
        spk_id_parsing_fn: Any, 
        ext: str = '.wav', 
        batch_size: int = 100,
        n_ref_embs: int = 64, 
        n_trials: int = 10,
        n_test_iterations: int = 10000, 
        output_dir: str = 'out'
                    ):
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
    ref_embs_dict = build_initial_ref_embs(fname_by_spk_id, sr_model) 
    best_accuracy, best_embs_dict, best_results_dict = 0, {}, {}
    for _ in tqdm(range(n_trials),total=n_trials, desc='Trials'):
        ref_embs_dict = build_initial_ref_embs(fname_by_spk_id, sr_model) 
        confusion_mtx_dict = init_confusion_mtx_dict() 
        accumulated_samples = []
        for _ in tqdm(range(n_test_iterations), total=n_test_iterations, desc='Test iterations'):
            accumulated_samples = accumulate_samples(accumulated_samples, fname_by_spk_id)
            if len(accumulated_samples) > batch_size:
                confusion_mtx_dict = infer_and_compute_results(
                                                            sr_model=sr_model,
                                                            confusion_mtx_dict=confusion_mtx_dict,
                                                            accumulated_samples=accumulated_samples,
                                                            bool_to_spk_id_dict=bool_to_spk_id_dict,
                                                            ref_embs_dict=ref_embs_dict,
                                                        )
                accumulated_samples = []
        if len(accumulated_samples):
            confusion_mtx_dict = infer_and_compute_results(
                                                            sr_model=sr_model,
                                                            confusion_mtx_dict=confusion_mtx_dict,
                                                            accumulated_samples=accumulated_samples,
                                                            bool_to_spk_id_dict=bool_to_spk_id_dict,
                                                            ref_embs_dict=ref_embs_dict,
                                                        )
            accumulated_samples = []

        results_dict = get_results_dict(confusion_mtx_dict)
        if results_dict['accuracy'] > best_accuracy:
            best_accuracy = results_dict['accuracy']
            best_embs_dict = ref_embs_dict.copy()
            best_results_dict = results_dict.copy()
    print(f"Best results: {best_results_dict}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    for spk_id, embs in best_embs_dict.items():
        output_path = os.path.join(output_dir, f"{spk_id}_{n_ref_embs}.npy")
        with open(output_path, 'wb') as w:
            np.save(w, embs)
            print(f"Wrote embeddings file at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='', action='store', type=str, required=True, help='Audio file directory containg .wav files.')
    parser.add_argument('--model_path', default='', action='store', type=str, required=True, help='.h5 model path')
    parser.add_argument('--n_tests', default=10000, action='store', type=int, required=False, help='Number of test iterations')
    parser.add_argument('--batch_size', default=100, action='store', type=int, required=False, help='Number of samples in one batch')
    parser.add_argument('--n_ref_embs', default=64, action='store', type=int, required=False, help='Number of reference embeddings')
    
    args = parser.parse_args()
    main(
        audio_dir=args.audio_dir, 
        sr_model_path=args.model_path, 
        n_test_iterations=args.n_tests,
        batch_size=args.batch_size,
        n_ref_embs=args.n_ref_embs,
        spk_id_parsing_fn=parse_spk_id,
        ext = '.wav'
        )