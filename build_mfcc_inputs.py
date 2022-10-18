import argparse
import os
from deep_speaker.audio import Audio
from deep_speaker.utils import ensures_dir

def main(audio_dir: str, output_dir: str, sample_rate: int = 16000):
    print('ceva working dir', output_dir)
    ensures_dir(output_dir)
    Audio(cache_dir=output_dir, audio_dir=audio_dir, sample_rate=sample_rate, ext='.wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='', action='store', type=str, required=True, help='Audio file directory containg .wav files.')
    parser.add_argument('--output_dir', default='', action='store', type=str, required=True, help='Output audio file directory')
    parser.add_argument('--sample_rate', default=16000, action='store', type=int, required=False, help='Sample rate')
    args = parser.parse_args()
    main(args.audio_dir, args.output_dir, args.sample_rate)