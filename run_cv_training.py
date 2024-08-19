import argparse
from pathlib import Path
from sklearn.model_selection import KFold
from loguru import logger
import torch

from model import ModelManager
from noise import create_noise_transform, TRANSFORM_NAMES
from utils import recover_ids
from training import train_finetune, train_denoise
import configparser


def main(args, config):

    mm = ModelManager(config['model_name'])
    train_ids = recover_ids(args.data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Starting {args.n_folds}-fold cross validation using {len(train_ids)} samples")

    nt = create_noise_transform(config['denoising_transform_name'])

    kf = KFold(n_splits=args.n_folds, random_state=args.random_state, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(train_ids)):
        if i <= args.skip_folds:
            logger.info(f"Skipping Fold ({i + 1}/{args.n_folds})")
            continue

        logger.info(f"Starting Fold ({i + 1}/{args.n_folds})")
        ids_train, ids_validation = train_ids[train_index], train_ids[test_index]

        model = mm.create_model()
        model.to(device)

        # pretraining
        train_denoise(model, device, 'temp', ids_train, args.data_dir, nt,
                      config=config, debug=args.debug)

        # finetune
        logger.info("Finetuning model")
        res = train_finetune(model, device, str(args.model_dir / f'model-{i}'), ids_train, ids_validation,
                             data_dir=args.data_dir, config=config, debug=args.debug)

        logger.success(f"Completed training {i}")

        del model

        if args.debug:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training script")

    parser.add_argument("--model_dir", type=Path, default=".",
                        help="Directory to save the output model weights in h5 format")
    parser.add_argument("--data_dir", type=Path, default=".",
                        help="Path to the raw features")
    parser.add_argument("--config_file", type=Path, default="./config.ini",
                        help="Configuration file containing model & training parameters in a Windows .ini format")
    parser.add_argument("--n_folds", type=int, default=4, choices=range(2, 11),
                        help="Number of folds/models")
    parser.add_argument("--skip_folds", type=int, default=-1,
                        help="Number of initial folds to skip")
    parser.add_argument("--random_state", type=int, default=2024,
                        help="Controls the randomness")
    parser.add_argument("--debug", action='store_true',
                        help="Run on a small subset of the data for debugging")

    args = parser.parse_args()

    config_parser = configparser.ConfigParser()
    config_parser.read(args.config_file)
    config = config_parser['DEFAULT']

    main(args, config)
