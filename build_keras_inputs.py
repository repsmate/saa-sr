import argparse
from deep_speaker.batcher import KerasFormatConverter
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES

def build_keras_inputs(working_dir: str, n_samples_for_training: int = 600, n_samples_for_testing: int = 100):
    # counts_per_speaker: If you specify --counts_per_speaker 600,100, that means for each speaker,
    # you're going to generate 600 samples for training and 100 for testing. One sample is 160 frames
    # by default (~roughly 1.6 seconds).
    counts_per_speaker = (n_samples_for_training, n_samples_for_testing)
    print(f"N samples for training per speaker: {n_samples_for_training}")
    print(f"N samples for testing per speaker: {n_samples_for_testing}")
    kc = KerasFormatConverter(working_dir)
    kc.generate(max_length=NUM_FRAMES, counts_per_speaker=counts_per_speaker)
    kc.persist_to_disk()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='', action='store', type=str, required=True, help='MFCC file directory containg .npy files.')
    parser.add_argument('--n_train', default=1000000, action='store', type=int, required=False, help='n_samples_for_training')
    parser.add_argument('--n_test', default=10000, action='store', type=int, required=False, help='n_samples_for_testing')
    args = parser.parse_args()
    build_keras_inputs(args.audio_dir, args.n_train, args.n_test)