import os
import signal
import time

import torch

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


_REQUEST_STOP = False


def _signal_handler(signum, frame):  # pragma: no cover
    global _REQUEST_STOP
    _REQUEST_STOP = True


def _install_signal_handlers():  # pragma: no cover
    for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None), getattr(signal, "SIGINT", None)):
        if s is None:
            continue
        try:
            signal.signal(s, _signal_handler)
        except Exception:
            pass


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def _apply_lr_schedule(optimizer, lr_schedule, epoch: int) -> None:
    if not lr_schedule:
        return

    current_lr = None
    for start_epoch, lr in lr_schedule:
        if epoch >= start_epoch:
            current_lr = lr
        else:
            break

    if current_lr is None:
        return

    for group in optimizer.param_groups:
        group["lr"] = current_lr


def _atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


class train_model:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config

        self.early_stopping = EarlyStopping(getattr(config, "EARLY_STOPPING_PATIENCE", 10))

        self.checkpoint_path = getattr(config, "CHECKPOINT_PATH", None)
        self.resume = bool(getattr(config, "RESUME", False))
        self.save_every = int(getattr(config, "SAVE_EVERY", 1))

    def _compute_loss(self, outputs, y, stn):
        if getattr(self.criterion, "needs_station_ids", False):
            return self.criterion(outputs, y, stn)
        return self.criterion(outputs["H"].unsqueeze(-1), y)

    def _save_checkpoint(self, epoch: int):
        if not self.checkpoint_path:
            return
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "early_stopping": {
                "best_loss": self.early_stopping.best_loss,
                "counter": self.early_stopping.counter,
                "early_stop": self.early_stopping.early_stop,
            },
            "timestamp": time.time(),
        }
        _atomic_torch_save(ckpt, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def _maybe_resume(self):
        if not (self.resume and self.checkpoint_path and os.path.exists(self.checkpoint_path)):
            return 1

        ckpt = _load_checkpoint(self.checkpoint_path, self.device)
        try:
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint weights from {self.checkpoint_path}: {e}")

        es = ckpt.get("early_stopping", {})
        self.early_stopping.best_loss = float(es.get("best_loss", self.early_stopping.best_loss))
        self.early_stopping.counter = int(es.get("counter", self.early_stopping.counter))
        self.early_stopping.early_stop = bool(es.get("early_stop", self.early_stopping.early_stop))

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resuming from checkpoint: {self.checkpoint_path} (next epoch {start_epoch}/{self.config.NUM_EPOCHS})")
        return start_epoch

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for x_dict, y, stn in self.train_loader:
            if _REQUEST_STOP:
                break

            H = x_dict["H"].to(self.device)
            D = x_dict["D"].to(self.device)
            S = x_dict["S"].to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model({"H": H, "D": D}, S)
            loss = self._compute_loss(outputs, y, stn)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item()) * H.size(0)

        return total_loss / len(self.train_loader.dataset)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x_dict, y, stn in self.val_loader:
                if _REQUEST_STOP:
                    break

                H = x_dict["H"].to(self.device)
                D = x_dict["D"].to(self.device)
                S = x_dict["S"].to(self.device)
                y = y.to(self.device)

                outputs = self.model({"H": H, "D": D}, S)
                loss = self._compute_loss(outputs, y, stn)

                total_loss += float(loss.item()) * H.size(0)

        return total_loss / len(self.val_loader.dataset)

    def fit(self):
        _install_signal_handlers()

        start_epoch = self._maybe_resume()

        for epoch in range(start_epoch, self.config.NUM_EPOCHS + 1):
            _apply_lr_schedule(self.optimizer, getattr(self.config, "LR_SCHEDULE", None), epoch)

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            print(
                f"Epoch {epoch}/{self.config.NUM_EPOCHS} "
                f"| Train Loss: {train_loss:.6f} "
                f"| Val Loss: {val_loss:.6f}"
            )

            if wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

            if getattr(self.config, "USE_EARLY_STOPPING", False):
                improved = self.early_stopping.step(val_loss)
                if improved:
                    torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                    print("Best model saved.")

            if self.checkpoint_path and (self.save_every and epoch % self.save_every == 0 or _REQUEST_STOP or epoch == self.config.NUM_EPOCHS):
                self._save_checkpoint(epoch)

            if _REQUEST_STOP:
                print("Received stop signal; checkpoint saved and training will exit.")
                return

            if getattr(self.config, "USE_EARLY_STOPPING", False) and self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print("Training finished.")
