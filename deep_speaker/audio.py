import logging
import os
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from glob import glob
import os
from python_speech_features import fbank
from tqdm import tqdm

from deep_speaker.constants import SAMPLE_RATE, NUM_FBANKS
from deep_speaker.utils import find_files, ensures_dir

logger = logging.getLogger(__name__)


def read_mfcc(input_filename, sample_rate):
    audio = Audio.read(input_filename, sample_rate)
    if audio is None:
        return None
    mfcc = mfcc_fbank(audio, sample_rate)
    return mfcc


def extract_speaker_and_utterance_ids(filepath: str):  # LIBRI.
    # 'audio/dev-other/116/288045/116-288045-0000.flac'
    fname = os.path.basename(filepath)
    speaker_id, ctx_id, channel, vad_id, window_id = fname.replace('.wav', '').split('_')
    speaker = channel
    utterance = '-'.join([speaker_id, ctx_id, vad_id, window_id])
    return speaker, utterance

def find_audio_files(audio_dir: str, ext: str = '.wav'):
    return glob(os.path.join(audio_dir, f"*{ext}"))

class Audio:

    def __init__(self, cache_dir: str, audio_dir: str = None, sample_rate: int = SAMPLE_RATE, ext='wav'):
        self.ext = ext
        self.cache_dir = os.path.join(cache_dir, 'audio-fbanks')
        ensures_dir(self.cache_dir)
        if audio_dir is not None:
            self.build_cache(audio_dir, sample_rate)
        self.speakers_to_utterances = defaultdict(dict)
        for cache_file in find_files(self.cache_dir, ext='npy'):
            # /path/to/speaker_utterance.npy
            speaker_id, utterance_id = Path(cache_file).stem.split('_')
            self.speakers_to_utterances[speaker_id][utterance_id] = cache_file

    @property
    def speaker_ids(self):
        return sorted(self.speakers_to_utterances)

    @staticmethod
    def read(filename, sample_rate=SAMPLE_RATE):
        try:
            audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        except:
            print(f"Could not open: {filename}")
            audio = None
            sr = sample_rate
        assert sr == sample_rate
        return audio

    def build_cache(self, audio_dir, sample_rate):
        logger.info(f'audio_dir: {audio_dir}.')
        logger.info(f'sample_rate: {sample_rate:,} hz.')
        audio_files = find_audio_files(audio_dir, ext=self.ext)
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any {self.ext} files in {audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in {audio_dir}.')
        print('ceva example path', audio_files[0])
        with tqdm(audio_files) as bar:
            for audio_file_path in bar:
                #bar.set_description(audio_file_path)
                self.cache_audio_file(audio_file_path, sample_rate)

    def cache_audio_file(self, input_filepath, sample_rate):
        sp, utt = extract_speaker_and_utterance_ids(input_filepath)
        cache_filename = os.path.join(self.cache_dir, f'{sp}_{utt}.npy')
        if not os.path.isfile(cache_filename):
            try:
                mfcc = read_mfcc(input_filepath, sample_rate)
                if mfcc is not None:
                    np.save(cache_filename, mfcc)
            except librosa.util.exceptions.ParameterError as e:
                logger.error(e)

def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc


def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    # delta_1 = delta(filter_banks, N=1)
    # delta_2 = delta(delta_1, N=1)
    # frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]
