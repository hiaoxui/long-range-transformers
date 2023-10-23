import json
import os
from argparse import ArgumentParser
import shutil


def cli():
    parser = ArgumentParser()
    parser.add_argument('-s', type=str, help='source folder')
    parser.add_argument('-d', type=str, help='destination folder')
    parser.add_argument('-t', type=str, help='transformer type')
    args = parser.parse_args()
    return args


def run():
    args = cli()
    os.makedirs(args.d, exist_ok=True)
    if not os.path.exists(os.path.join(args.d, 'vocabulary')):
        shutil.copytree(os.path.join(args.s, 'vocabulary'), os.path.join(args.d, 'vocabulary'))
    cfg = json.load(open(os.path.join(args.s, 'config.json')))
    cfg['model']['text_field_embedder']['token_embedders']['tokens']['attn_args'] = {
        'type': args.t
    }
    with open(os.path.join(args.d, 'config.json'), 'w') as fp:
        json.dump(cfg, fp, indent=4)
    weight_out = os.path.join(args.d, 'weights.th')
    if not os.path.exists(weight_out):
        for fn in ['weights.th', 'best.th']:
            weight_file_path = os.path.join(args.s, fn)
            if os.path.exists(weight_file_path):
                shutil.copy(weight_file_path, weight_out)


if __name__ == '__main__':
    run()
