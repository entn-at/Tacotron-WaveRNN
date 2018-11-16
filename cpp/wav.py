import os
import sys

sys.path.append('..')

import numpy as np
from hparams import hparams
from datasets.audio import save_wavernn_wav


def main():
    os.chdir('..')

    # Get hparams
    hp = hparams.parse('')

    # Get path
    input_path = os.path.join('cpp', 'output.txt')
    output_path = os.path.join('cpp', 'output.wav')

    print(f'Loading input data from {input_path}')
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [float(v.rstrip()) for v in f.readlines()]
        data = np.array(data)

        print(f'Saving wav into {output_path}')
        save_wavernn_wav(data, output_path, hp.sample_rate)

    print('Finish!!!')


if __name__ == '__main__':
    main()
