import os
import sys

sys.path.append('..')

import numpy as np
import torch
from wavernn.model import Model
from wavernn.train import _bits, _pad

device = torch.device('cpu')
torch.set_printoptions(profile='full')


def main():
    os.chdir('../')

    # Get path
    checkpoint_path = os.path.join('logs-WaveRNN', 'wavernn_pretrained', 'wavernn_model.pyt')
    mels_dir = os.path.join('tacotron_output', 'eval')
    cpp_dir = os.path.join('cpp')
    params_dir = os.path.join(cpp_dir, 'params')
    inputs_dir = os.path.join(cpp_dir, 'inputs')
    os.makedirs(cpp_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)

    # Initialize Model
    model = Model(rnn_dims=512, fc_dims=512, bits=_bits, pad=_pad,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).to(device)

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'Loading model from {checkpoint_path}')

    # Save params
    print(f'Saving params into {params_dir}')
    for i, (name, param) in enumerate(model.named_parameters()):
        if (i > 67):
            param = torch.squeeze(param.data)
            with open(os.path.join(params_dir, f'{name}.txt'), 'w', encoding='utf-8') as f:
                if (param.dim() == 1):
                    f.write(f'static const double {name}[{param.shape[0]}] = ')
                    f.write('{')
                    for j, val in enumerate(param):
                        f.write(str(val.item()))
                        if (j != len(param) - 1):
                            f.write(', ')
                    f.write('};')
                else:
                    f.write(f'static const double {name}[{param.shape[0]}][{param.shape[1]}] = ')
                    f.write('{')
                    for j, row in enumerate(param):
                        f.write('{')
                        for k, val in enumerate(row):
                            f.write(str(val.item()))
                            if (j != len(row) - 1):
                                f.write(', ')
                        f.write('}')
                        if (j != len(param) - 1):
                            f.write(', ')
                    f.write('};')

    # Load mels
    print(f'Loading mels from {mels_dir}')
    filenames = [f for f in sorted(os.listdir(mels_dir)) if f.endswith('.npy')]
    mels = torch.FloatTensor(np.load(os.path.join(mels_dir, filenames[0])).T).to(device).unsqueeze(0)
    mels, aux = model.upsample(mels)
    mels = torch.squeeze(mels).detach().numpy()
    aux = torch.squeeze(aux).detach().numpy()
    aux = [aux[:, model.aux_dims * i:model.aux_dims * (i + 1)] for i in range(4)]

    # Save inputs
    print(f'Saving inputs into {inputs_dir}')
    np.savetxt(os.path.join(inputs_dir, 'mels.txt'), mels)
    np.savetxt(os.path.join(inputs_dir, 'aux_0.txt'), aux[0])
    np.savetxt(os.path.join(inputs_dir, 'aux_1.txt'), aux[1])
    np.savetxt(os.path.join(inputs_dir, 'aux_2.txt'), aux[2])
    np.savetxt(os.path.join(inputs_dir, 'aux_3.txt'), aux[3])

    print('Finish!!!')


if __name__ == '__main__':
    main()
