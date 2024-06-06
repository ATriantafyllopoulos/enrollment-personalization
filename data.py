import audiofile
import audobject
import audtorch
import numpy as np
import os
import pandas as pd
import torch


class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(self.labels))
        self.inverse_map = {code: label for code,
                            label in zip(codes, self.labels)}
        self.map = {label: code for code,
                    label in zip(codes, self.labels)}

    def __call__(self, x):
        return self.encode(x)

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]


def create_data(cfg):
    if cfg.name == "aibo":
        return create_aibo(
            root=cfg.root,
            task=cfg.task
        )
    elif cfg.name == "msppodcast":
        return create_msp(cfg.root)


def create_msp(root):
    df = pd.read_csv(os.path.join(root, "Labels", "labels_consensus.csv"))
    df = df.loc[df["EmoClass"].isin(["N", "H", "A", "S"])]
    df = df.loc[df["SpkrID"] != "Unknown"]

    # remove speakers with < 5 utterances
    speaker_vals = df["SpkrID"].value_counts()
    speaker_vals = speaker_vals.loc[df["SpkrID"].value_counts() >= 5].index
    df = df.loc[df["SpkrID"].isin(speaker_vals)]

    # remove speakers with < 1 "N"
    speaker_neutral = df.groupby("SpkrID")["EmoClass"].apply(lambda x: sum(x == "N"))
    speaker_neutral = speaker_neutral.loc[speaker_neutral >= 1].index
    df = df.loc[df["SpkrID"].isin(speaker_neutral)]

    df_train = df.loc[df["Split_Set"] == "Train"]
    df_dev = df.loc[df["Split_Set"] == "Development"]
    df_test = df.loc[df["Split_Set"] == "Test1"]

    encoder = LabelEncoder(list(df["EmoClass"].unique()))
    train_dataset = MSPPodcast(
        root=root,
        df=df_train,
        transform=audtorch.transforms.RandomCrop(48000),
        adaptation_transform=audtorch.transforms.RandomCrop(48000),
        target_transform=encoder
    )
    print("TRAIN DATASET:")
    print(train_dataset)

    dev_dataset = MSPPodcast(
        root=root,
        df=df_dev,
        adaptation_transform=audtorch.transforms.RandomCrop(48000),
        target_transform=encoder
    )
    print("DEV DATASET:")
    print(dev_dataset)

    test_dataset = MSPPodcast(
        root=root,
        df=df_test,
        adaptation_transform=audtorch.transforms.RandomCrop(48000),
        target_transform=encoder
    )
    print("TEST DATASET:")
    print(test_dataset)

    print("EXAMPLE DATA:")
    data = train_dataset[0]
    print(data["instance"]["input"].shape, data["instance"]["label"])
    print(data["adaptation"]["neutral"]["input"].shape, data["adaptation"]["neutral"]["label"])
    print(data["adaptation"]["emotional"]["input"].shape, data["adaptation"]["emotional"]["label"])

    return train_dataset, dev_dataset, test_dataset, encoder


def create_aibo(root, task="5cl"):
    df = pd.read_csv(
        os.path.join(
            root,
            "labels",
            "IS2009EmotionChallenge",
            f"chunk_labels_{task}_corpus.txt"
        ),
        header=None,
        sep=" "
    )
    df = df.rename(
        columns={
            0: "id",
            1: "class",
            2: "conf"
        }
    )
    df["file"] = df["id"].apply(lambda x: x + ".wav")
    df["school"] = df["id"].apply(lambda x: x.split("_")[0])
    df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
    df_test = df.loc[df["school"] == "Mont"]
    df_train_dev = df.loc[df["school"] == "Ohm"]
    speakers = sorted(df_train_dev["speaker"].unique())
    df_train = df_train_dev.loc[df_train_dev["speaker"].isin(
        speakers[:-2])]
    df_dev = df_train_dev.loc[df_train_dev["speaker"].isin(speakers[-2:])]

    encoder = LabelEncoder(list(df["class"].unique()))
    train_dataset = AIBO(
        root=root,
        df=df_train,
        target_column="class",
        task=task,
        transform=audtorch.transforms.RandomCrop(48000),
        adaptation_transform=audtorch.transforms.RandomCrop(48000),
        target_transform=encoder
    )
    print("TRAIN DATASET:")
    print(train_dataset)

    dev_dataset = AIBO(
        root=root,
        df=df_dev,
        target_column="class",
        task=task,
        adaptation_transform=audtorch.transforms.RandomCrop(48000),
        target_transform=encoder
    )
    print("DEV DATASET:")
    print(dev_dataset)

    test_dataset = AIBO(
        root=root,
        df=df_test,
        target_column="class",
        task=task,
        adaptation_transform=audtorch.transforms.RandomCrop(48000),
        target_transform=encoder
    )
    print("TEST DATASET:")
    print(test_dataset)

    print("EXAMPLE DATA:")
    data = train_dataset[0]
    print(data["instance"]["input"].shape, data["instance"]["label"])
    print(data["adaptation"]["neutral"]["input"].shape, data["adaptation"]["neutral"]["label"])
    print(data["adaptation"]["emotional"]["input"].shape, data["adaptation"]["emotional"]["label"])

    return train_dataset, dev_dataset, test_dataset, encoder


class AIBO(torch.utils.data.Dataset):
    emotional_classes = {
        "2cl": "IDL",
        "5cl": "N"
    }
    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        task: str = "2cl",
        target_column: str = "class",
        transform=None,
        adaptation_transform=None,
        target_transform=None
    ) -> None:
        super().__init__()
        self.root = root
        self.df = df
        self.task = task
        self.target_column = target_column
        self.transform = transform
        self.adaptation_transform = adaptation_transform
        self.target_transform = target_transform

        self.adaptation_set = self.df.groupby(["speaker", "class"]).apply(lambda x: x.sort_values(by="file")["file"].values[0])
        self.speakers = self.df["speaker"].unique()
        self.labels = self.df[self.target_column].unique().tolist()
        self.indices = self.df.loc[~self.df["file"].isin(self.adaptation_set)].index
        self.data = self.df.loc[self.indices]
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        row = self.df.loc[index]
        audio = audiofile.read(os.path.join(self.root, "wav", row["file"]))[0]
        label = row[self.target_column]

        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            label = self.target_transform(label)

        speaker = row["speaker"]
        adaptation_set = self.adaptation_set.loc[speaker]
        adaptation_data = {}
        for ad_label, ad_file in adaptation_set.items():
            ad_audio = audiofile.read(os.path.join(self.root, "wav", ad_file))[0]
            if self.adaptation_transform is not None:
                ad_audio = self.adaptation_transform(ad_audio)
            if self.target_transform is not None:
                ad_label = self.target_transform(ad_label)
            adaptation_data[ad_label] = ad_audio

        neutral_label = self.emotional_classes[self.task]
        emotional_labels = list(set(self.labels) - set([neutral_label]))

        if self.target_transform is not None:
            neutral_label = self.target_transform(neutral_label)
            emotional_labels = [
                self.target_transform(x) 
                for x in emotional_labels
            ]
        
        for x in emotional_labels:
            if x not in adaptation_data:
                adaptation_data[x] = np.zeros(48000)
        if neutral_label not in adaptation_data:
            adaptation_data[neutral_label] = np.zeros(48000)

        return {
            "instance": {
                "input": audio.astype(np.float32),
                "label": label
            },
            "adaptation": {
                "neutral": {
                    "input": adaptation_data[neutral_label].astype(np.float32),
                    "label": neutral_label
                },
                "emotional": {
                    "input": np.stack([adaptation_data[x] for x in emotional_labels]).astype(np.float32),
                    "label": np.stack(emotional_labels)
                }
            }
        }
    def __repr__(self):
        s = (
            f"FAU-AIBO(\n"
            f"  Task: {self.task} "
            f"{{{','.join(self.labels)}}}\n"
            f"  Data (all): {len(self.df)}\n"
            f"  Data (inference): {len(self.indices)}\n"
            f"  Data (enrollment): {len(self.adaptation_set)}\n"
            f"  Speakers: {len(self.speakers)}\n"
            f")"
        )
        return s


class MSPPodcast(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        target_column: str = "EmoClass",
        transform=None,
        adaptation_transform=None,
        target_transform=None
    ) -> None:
        super().__init__()
        self.root = root
        self.df = df
        self.target_column = target_column
        self.transform = transform
        self.adaptation_transform = adaptation_transform
        self.target_transform = target_transform

        self.adaptation_set = self.df.groupby(["SpkrID", self.target_column]).apply(lambda x: x.sort_values(by="FileName")["FileName"].values[0])
        self.speakers = self.df["SpkrID"].unique()
        self.labels = self.df[self.target_column].unique().tolist()
        self.indices = self.df.loc[~self.df["FileName"].isin(self.adaptation_set)].index
        self.data = self.df.loc[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        row = self.df.loc[index]
        audio = audiofile.read(os.path.join(self.root, "Audios", row["FileName"]))[0]
        label = row[self.target_column]

        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            label = self.target_transform(label)

        speaker = row["SpkrID"]
        adaptation_set = self.adaptation_set.loc[speaker]
        adaptation_data = {}
        for ad_label, ad_file in adaptation_set.items():
            ad_audio = audiofile.read(os.path.join(self.root, "Audios", ad_file))[0]
            if self.adaptation_transform is not None:
                ad_audio = self.adaptation_transform(ad_audio)
            if self.target_transform is not None:
                ad_label = self.target_transform(ad_label)
            adaptation_data[ad_label] = ad_audio

        neutral_label = "N"
        emotional_labels = list(set(self.labels) - set([neutral_label]))

        if self.target_transform is not None:
            neutral_label = self.target_transform(neutral_label)
            emotional_labels = [
                self.target_transform(x) 
                for x in emotional_labels
            ]
        
        for x in emotional_labels:
            if x not in adaptation_data:
                adaptation_data[x] = np.zeros(48000)
        if neutral_label not in adaptation_data:
            adaptation_data[neutral_label] = np.zeros(48000)

        return {
            "instance": {
                "input": audio.astype(np.float32),
                "label": label
            },
            "adaptation": {
                "neutral": {
                    "input": adaptation_data[neutral_label].astype(np.float32),
                    "label": neutral_label
                },
                "emotional": {
                    "input": np.stack([adaptation_data[x] for x in emotional_labels]).astype(np.float32),
                    "label": np.stack(emotional_labels)
                }
            }
        }
    def __repr__(self):
        s = (
            f"MSP-Podcast(\n"
            f"  Labels: {{{','.join(self.labels)}}}\n"
            f"  Data (all): {len(self.df)}\n"
            f"  Data (inference): {len(self.indices)}\n"
            f"  Data (enrollment): {len(self.adaptation_set)}\n"
            f"  Speakers: {len(self.speakers)}\n"
            f")"
        )
        return s


if __name__ == "__main__":

    def get_msp(df):
        df = df.reset_index()
        df = df.rename(columns={0: "FileName"})
        return df
    train, dev, test, _ = create_msp("msp-data")
    get_msp(train.adaptation_set)[["SpkrID", "FileName"]].to_csv("adaptation-sets/msp/train.csv", index=False)
    get_msp(dev.adaptation_set)[["SpkrID", "FileName"]].to_csv("adaptation-sets/msp/dev.csv", index=False)
    get_msp(test.adaptation_set)[["SpkrID", "FileName"]].to_csv("adaptation-sets/msp/test.csv", index=False)
    def get_aibo(df):
        df = df.reset_index()
        df = df.rename(columns={0: "file"})
        return df
    train, dev, test, _ = create_aibo("aibo-data", task="2cl")
    get_aibo(train.adaptation_set)[["speaker", "file"]].to_csv("adaptation-sets/aibo-2cl/train.csv", index=False)
    get_aibo(dev.adaptation_set)[["speaker", "file"]].to_csv("adaptation-sets/aibo-2cl/dev.csv", index=False)
    get_aibo(test.adaptation_set)[["speaker", "file"]].to_csv("adaptation-sets/aibo-2cl/test.csv", index=False)

    train, dev, test, _ = create_aibo("aibo-data", task="5cl")
    get_aibo(train.adaptation_set)[["speaker", "file"]].to_csv("adaptation-sets/aibo-5cl/train.csv", index=False)
    get_aibo(dev.adaptation_set)[["speaker", "file"]].to_csv("adaptation-sets/aibo-5cl/dev.csv", index=False)
    get_aibo(test.adaptation_set)[["speaker", "file"]].to_csv("adaptation-sets/aibo-5cl/test.csv", index=False)