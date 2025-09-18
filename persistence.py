import json
from datetime import datetime
from pathlib import Path

from safetensors.torch import save_file


def save_checkpoint(
    checkpoints_dir, descriptor, model,
    epoch, train_loss, val_loss, is_best_epoch
):
    save_dir = checkpoints_dir / f"{datetime.utcnow():%Y%m%dZ%H%M%S}-{descriptor}"
    save_dir_tmp = save_dir.with_suffix(".tmp")
    save_dir_tmp.mkdir()
    meta = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    (save_dir_tmp / "meta.json").write_text(json.dumps(meta) + "\n")

    save_file(model.state_dict(), save_dir_tmp / "model.safetensors")

    save_dir_tmp.rename(save_dir)

    symlink_target = Path(".") / save_dir.name
    if is_best_epoch:
        best_path = checkpoints_dir / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)

    latest_path = checkpoints_dir / "latest"
    latest_path.unlink(missing_ok=True)
    latest_path.symlink_to(symlink_target, target_is_directory=True)
