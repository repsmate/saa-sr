import argparse
from deep_speaker.train import start_training

def train_model(working_dir: str, pre_training_phase: bool = True):
    # PRE TRAINING

    # commit a5030dd7a1b53cd11d5ab7832fa2d43f2093a464
    # Merge: a11d13e b30e64e
    # Author: Philippe Remy <premy.enseirb@gmail.com>
    # Date:   Fri Apr 10 10:37:59 2020 +0900
    # LibriSpeech train-clean-data360 (600, 100). 0.985 on test set (enough for pre-training).

    # TRIPLET TRAINING
    # [...]
    # Epoch 175/1000
    # 2000/2000 [==============================] - 919s 459ms/step - loss: 0.0077 - val_loss: 0.0058
    # Epoch 176/1000
    # 2000/2000 [==============================] - 917s 458ms/step - loss: 0.0075 - val_loss: 0.0059
    # Epoch 177/1000
    # 2000/2000 [==============================] - 927s 464ms/step - loss: 0.0075 - val_loss: 0.0059
    # Epoch 178/1000
    # 2000/2000 [==============================] - 948s 474ms/step - loss: 0.0073 - val_loss: 0.0058
    if pre_training_phase:
        print(f"Pretraining..")
    start_training(working_dir, pre_training_phase=pre_training_phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='', action='store', type=str, required=True, help='Audio file directory containg .wav files.')
    parser.add_argument('--pretraining', default=True, action='store', type=int, required=False, help='If true, use pretraining  Else training')
    args = parser.parse_args()
    train_model(args.audio_dir, args.pretraining)