import glob
import torch
from torch.utils.data import Dataset

from closd.diffusion_planner.data_loaders.tensors import lengths_to_mask


class ReflowPairDataset(Dataset):

    def __init__(self, pair_files, base_dataset):

        # Accept:
        #
        #   file.pt
        #   file1.pt,file2.pt
        #   seed*.pt
        #
        specs = [
            x.strip()
            for x in pair_files.split(",")
            if x.strip()
        ]

        paths = []

        for spec in specs:
            matches = sorted(glob.glob(spec))

            if matches:
                paths.extend(matches)
            else:
                raise FileNotFoundError(
                    f"No reflow pair files matched [{spec}]"
                )

        # Remove accidental duplicates.
        paths = list(dict.fromkeys(paths))

        print("===== Loading RF-2 reflow shards =====")

        for p in paths:
            print(" ", p)

        print("number of shards:", len(paths))
        print("======================================")

        z0 = []
        z1 = []
        prefix = []
        lengths = []
        target_cond = []
        is_heading = []

        text = []
        target_joint_names = []
        db_key = []
        metadata = []

        for path in paths:

            print(
                f"Loading reflow shard [{path}]..."
            )

            payload = torch.load(
                path,
                map_location="cpu",
            )

            z0.append(
                payload["z0"].float()
            )

            z1.append(
                payload["z1"].float()
            )

            prefix.append(
                payload["prefix"].float()
            )

            lengths.append(
                payload["lengths"].long()
            )

            target_cond.append(
                payload["target_cond"].float()
            )

            is_heading.append(
                payload["is_heading"]
            )

            text.extend(
                payload["text"]
            )

            target_joint_names.extend(
                payload["target_joint_names"]
            )

            db_key.extend(
                payload.get(
                    "db_key",
                    [None] * len(payload["z0"]),
                )
            )

            metadata.append(
                payload.get(
                    "metadata",
                    {},
                )
            )

        self.z0 = torch.cat(z0, dim=0)
        self.z1 = torch.cat(z1, dim=0)
        self.prefix = torch.cat(prefix, dim=0)

        self.lengths = torch.cat(
            lengths,
            dim=0,
        )

        self.target_cond = torch.cat(
            target_cond,
            dim=0,
        )

        self.is_heading = torch.cat(
            is_heading,
            dim=0,
        )

        self.text = text
        self.target_joint_names = target_joint_names
        self.db_key = db_key
        self.metadata = metadata
        self.paths = paths

        n = len(self.z0)

        assert len(self.z1) == n
        assert len(self.prefix) == n
        assert len(self.lengths) == n
        assert len(self.target_cond) == n
        assert len(self.is_heading) == n
        assert len(self.text) == n
        assert len(self.target_joint_names) == n

        # Needed by target-location loss.
        self.t2m_dataset = base_dataset.t2m_dataset
        self.mean = base_dataset.mean
        self.std = base_dataset.std

        if hasattr(base_dataset, "mean_gpu"):
            self.mean_gpu = base_dataset.mean_gpu

        if hasattr(base_dataset, "std_gpu"):
            self.std_gpu = base_dataset.std_gpu

        print("===== RF-2 dataset loaded =====")
        print("total pairs :", n)
        print("z0          :", self.z0.shape)
        print("z1          :", self.z1.shape)
        print("prefix      :", self.prefix.shape)
        print("===============================")

    def __len__(self):
        return len(self.z0)

    def __getitem__(self, idx):

        return {
            "z0": self.z0[idx],
            "z1": self.z1[idx],

            "prefix": self.prefix[idx],
            "lengths": self.lengths[idx],

            "target_cond": self.target_cond[idx],
            "is_heading": self.is_heading[idx],

            "text": self.text[idx],

            "target_joint_names":
                self.target_joint_names[idx],

            "db_key": self.db_key[idx],
        }


def reflow_collate(batch):

    z0 = torch.stack([
        x["z0"]
        for x in batch
    ])

    z1 = torch.stack([
        x["z1"]
        for x in batch
    ])

    prefix = torch.stack([
        x["prefix"]
        for x in batch
    ])

    lengths = torch.stack([
        x["lengths"]
        for x in batch
    ]).long()

    target_cond = torch.stack([
        x["target_cond"]
        for x in batch
    ])

    is_heading = torch.stack([
        x["is_heading"]
        for x in batch
    ])

    text = [
        x["text"]
        for x in batch
    ]

    target_joint_names = [
        x["target_joint_names"]
        for x in batch
    ]

    db_key = [
        x["db_key"]
        for x in batch
    ]

    mask = lengths_to_mask(
        lengths,
        z1.shape[-1],
    ).unsqueeze(1).unsqueeze(1)

    cond = {
        "y": {
            "mask": mask,
            "lengths": lengths,

            "prefix": prefix,
            "text": text,

            "reflow_noise": z0,

            "target_cond": target_cond,
            "target_joint_names":
                target_joint_names,

            "is_heading": is_heading,

            "db_key": db_key,
        }
    }

    return z1, cond
