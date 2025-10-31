import datetime
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from karpathy_model import KarpathyModel
from next_byte_tokenizer import NextByteTokenizer


class RunData:

    def __init__(self, directory, run_name):
        self.root_dir = Path(directory)
        if not self.root_dir.is_dir():
            raise Exception(f"Could not find directory {self.root_dir}")

        self.run_dir = self.root_dir / "runs" / run_name
        if not self.run_dir.is_dir():
            raise Exception(f"No runs directory {self.run_dir}")

        self.data_dir = self.root_dir / "data"
        if not self.data_dir.is_dir():
            raise Exception(f"No data directory {self.data_dir}")

        self.checkpoints_dir = self.run_dir / "checkpoints"
        if not self.checkpoints_dir.is_dir():
            self.checkpoints_dir.mkdir()

        self.train_data = json.loads((self.run_dir / "train.json").read_text())
        self.model_data = json.loads((self.run_dir / "model.json").read_text())


def meta_file(checkpoint_dir):
    return checkpoint_dir / "meta.json"


def safetensors_file(checkpoint_dir):
    return checkpoint_dir / "model.safetensors"


def save_checkpoint(
    run, descriptor, model, tokenizer,
    epoch, train_loss, val_loss, is_best_epoch
):
    now = datetime.datetime.now(datetime.UTC)
    save_dir = run.checkpoints_dir / f"{now:%Y%m%dZ%H%M%S}-{descriptor}"
    save_dir_tmp = save_dir.with_suffix(".tmp")
    save_dir_tmp.mkdir()
    meta = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "id_to_byte": tokenizer.id_to_byte,
    }
    meta_file(save_dir_tmp).write_text(json.dumps(meta) + "\n")

    save_file(model.state_dict(), safetensors_file(save_dir_tmp))

    save_dir_tmp.rename(save_dir)

    symlink_target = Path(".") / save_dir.name
    if is_best_epoch:
        best_path = run.checkpoints_dir / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)

    latest_path = run.checkpoints_dir / "latest"
    latest_path.unlink(missing_ok=True)
    latest_path.symlink_to(symlink_target, target_is_directory=True)


def load_checkpoint(run, checkpoint):
    checkpoint_dir = run.checkpoints_dir / checkpoint
    if not checkpoint_dir.is_dir():
        raise Exception(f"Could not find checkpoint dir {checkpoint_dir}")

    meta = json.loads(meta_file(checkpoint_dir).read_text())
    state = load_file(safetensors_file(checkpoint_dir))

    tokenizer = NextByteTokenizer(meta["id_to_byte"])
    model = KarpathyModel(vocab_size=tokenizer.vocab_size, **run.model_data)
    model.load_state_dict(state)

    return model, tokenizer
