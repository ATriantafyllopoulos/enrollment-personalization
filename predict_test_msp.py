import argparse
import os
import torch
import tqdm
import yaml

from omegaconf import (
    OmegaConf
)

from data import create_msp
from models import create_model
from training import criterion_weights, evaluate_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predict on MSP test set")
    parser.add_argument("root")
    parser.add_argument("model")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_dataset, _, test_dataset, encoder = create_msp(args.root)

    with open(os.path.join(args.model, "config.yaml"), "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)
    cfg = OmegaConf.create(cfg)
    model = create_model(cfg.model, output_dim=len(test_dataset.labels))
    model.load_state_dict(torch.load(os.path.join(args.model, "best.pth.tar")))
    weight = criterion_weights(train_dataset)
    criterion = torch.nn.CrossEntropyLoss(weight=weight.to(args.device))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )

    test_loss, test_uar, test_outputs = evaluate_loader(
        model=model,
        criterion=criterion,
        device=args.device,
        loader=test_loader,
        silent=False
    )
    print(f"Test UAR: {test_uar:.3f}")
    with open(os.path.join(args.model, "results.yaml"), "w") as fp:
        yaml.dump(
            {
                "test": {
                    "uar": test_uar,
                    "loss": test_loss
                }
            },
            fp
        )
    df = test_dataset.data
    df["outputs"] = test_outputs
    df["predictions"] = df["outputs"].apply(encoder.decode)
    df.to_csv(os.path.join(args.model, "test.csv"), index=False)