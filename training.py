import audmetric
import numpy as np
import os
import torch
import tqdm
import yaml

from data import create_data
from models import create_model


def unpack_data(data, device):
    new_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_data[key] = unpack_data(value, device)
        else:
            new_data[key] = value.to(device)
    return new_data


def train_epoch(
    model,
    optimizer,
    criterion,
    loader,
    device
):
    model.to(device)
    model.train()
    total_loss = 0
    for data in tqdm.tqdm(loader, total=len(loader), disable=True):
        data = unpack_data(data, device)
        output = model(data)
        loss = criterion(
            output["instance"]["output"],
            data["instance"]["label"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
    total_loss /= len(loader) + 1
    return total_loss


def evaluate_loader(
    model,
    criterion,
    loader,
    device,
    silent=True
):
    model.to(device)
    model.eval()
    total_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for data in tqdm.tqdm(loader, total=len(loader), disable=silent):
            data = unpack_data(data, device)
            output = model(data)
            loss = criterion(
                output["instance"]["output"],
                data["instance"]["label"]
            )
            targets.append(data["instance"]["label"].cpu().item())
            outputs.append(output["instance"]["output"].cpu().argmax(dim=1).item())
            total_loss += loss.cpu().item()
    total_loss /= len(loader) + 1
    targets = np.array(targets)
    outputs = np.array(outputs)
    uar = audmetric.unweighted_average_recall(targets, outputs)
    return total_loss, uar, outputs


def criterion_weights(dataset):
    frequency = (
        dataset.data[dataset.target_column]
        .map(dataset.target_transform.encode)
        .value_counts()
        .sort_index()
        .values
    )
    weight = torch.tensor(1 / frequency, dtype=torch.float32)
    weight /= weight.sum()
    return weight

def training(cfg):
    experiment_folder = cfg.meta.results_root
    os.makedirs(experiment_folder, exist_ok=True)
    train_dataset, dev_dataset, test_dataset, encoder = create_data(cfg.data)
    encoder.to_yaml(os.path.join(experiment_folder, "encoder.yaml"))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.hparams.bs,
        num_workers=2
    )
    data = next(iter(train_loader))
    print("EXAMPLE DATA (BATCHED):")
    print(data["instance"]["input"].shape, data["instance"]["label"])
    print(data["adaptation"]["neutral"]["input"].shape, data["adaptation"]["neutral"]["label"].shape)
    print(data["adaptation"]["emotional"]["input"].shape, data["adaptation"]["emotional"]["label"].shape)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )

    model = create_model(cfg.model, output_dim=len(train_dataset.labels))
    model = model.to(cfg.meta.device)
    # print(model)
    print("DRY RUN:")
    with torch.no_grad():
        output = model(unpack_data(data, cfg.meta.device))
        print(output["instance"]["output"].shape)
    # exit()

    if cfg.hparams.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hparams.lr)
    else:
        raise NotImplementedError(cfg.hparams.optimizer)
    
    weight = criterion_weights(train_dataset)
    print("CRITERION WEIGHTS:")
    print(weight)
    criterion = torch.nn.CrossEntropyLoss(weight=weight.to(cfg.meta.device))
    print("CRITERION:")
    print(criterion)

    best_uar = 0
    best_state = model.cpu().state_dict()
    best_epoch = 0
    for epoch in range(cfg.hparams.epochs):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg.meta.device,
            loader=train_loader
        )
        dev_loss, uar, outputs = evaluate_loader(
            model=model,
            criterion=criterion,
            device=cfg.meta.device,
            loader=dev_loader
        )
        print((
            f"[{epoch+1}/{cfg.hparams.epochs} -- "
            f"Loss: {train_loss:.3f}/{dev_loss:.3f} "
            f"UAR (dev): {uar:.3f} "
        ))
        torch.save(model.cpu().state_dict(), os.path.join(experiment_folder, "last.pth.tar"))
        if uar >= best_uar:
            best_uar = uar
            best_state = model.cpu().state_dict()
            best_epoch = epoch
            torch.save(best_state, os.path.join(experiment_folder, "best.pth.tar"))
    
    print("Training finished...")
    print(f"Best UAR found at epoch {epoch+1}: {best_uar:.3f}")
    best_state = torch.load(os.path.join(experiment_folder, "best.pth.tar"))
    model.load_state_dict(best_state)
    test_loss, test_uar, test_outputs = evaluate_loader(
        model=model,
        criterion=criterion,
        device=cfg.meta.device,
        loader=test_loader
    )
    print(f"Test UAR: {test_uar:.3f}")
    with open(os.path.join(experiment_folder, "results.yaml"), "w") as fp:
        yaml.dump(
            {
                "dev": {
                    "uar": best_uar,
                    "epoch": epoch
                },
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
    df.to_csv(os.path.join(experiment_folder, "test.csv"), index=False)