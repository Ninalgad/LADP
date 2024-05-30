import argparse
from pathlib import Path
from sklearn.model_selection import KFold
from loguru import logger
import torch

from model import ModelManager
from noise import create_noise_transform
from utils import recover_ids
from training import train_finetune, train_denoise


def main(args):

    mm = ModelManager(args.model_name)
    train_ids = recover_ids(args.data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Starting {args.n_folds}-fold cross validation using {len(train_ids)} samples")

    nt = create_noise_transform(args.denoising_transform_name)

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
        train_denoise(model, device, 'temp', ids_train, noising_transform=nt,
                      data_dir=args.data_dir, inp_size=args.image_size, num_epochs=args.num_denoising_epochs,
                      batch_size=args.batch_size, lr=args.lr_pretrain, debug=args.debug)

        # finetune
        logger.info("Finetuning model")
        res = train_finetune(model, device, str(args.model_dir / f'model-{i}'), ids_train, ids_validation,
                             data_dir=args.data_dir, inp_size=args.image_size, num_epochs=args.num_ft_epochs,
                             batch_size=args.batch_size, lr=args.lr_finetune, debug=args.debug)

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
    parser.add_argument("--n_folds", type=int, default=4, choices=range(2,11),
                        help="Number of folds/models")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size to resize model input")
    parser.add_argument("--model_name", type=str, choices=["ternausnet", "resnet34", "resnet18"], default='ternausnet',
                        help="Name of the model file")
    parser.add_argument("--denoising_transform_name", type=str, default='ladp', choices=['ladp', 'dae'],
                        help="Name of image nosing technique")
    parser.add_argument("--skip_folds", type=int, default=-1,
                        help="Number of initial folds to skip")
    parser.add_argument("--num_denoising_epochs", type=int, default=30,
                        help="Number of denoising epochs")
    parser.add_argument("--num_ft_epochs", type=int, default=30,
                        help="Number of finetuning epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr_pretrain", type=float, default=1e-4,
                        help="Learning Rate for pretraining")
    parser.add_argument("--lr_finetune", type=float, default=1e-4,
                        help="Learning Rate for finetuning")
    parser.add_argument("--random_state", type=int, default=2023,
                        help="Controls the randomness")
    parser.add_argument("--debug", action='store_true',
                        help="Run on a small subset of the data for debugging")

    args = parser.parse_args()
    main(args)
