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
    for s in (getattr(signal, 'SIGTERM', None), getattr(signal, 'SIGUSR1', None), getattr(signal, 'SIGINT', None)):
        if s is None:
            continue
        try:
            signal.signal(s, _signal_handler)
        except Exception:
            pass


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
        group['lr'] = current_lr


def _atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def train_model(
    model,
    train_loader,
    validation_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    patience=10,
    best_model_path='best_model.pth',
    early_stopping=False,
    lr_schedule=None,
    checkpoint_path=None,
    resume=False,
    save_every=1,
):
    # Train with optional early stopping and checkpoint resume.

    _install_signal_handlers()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 1

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = _load_checkpoint(checkpoint_path, device)
        try:
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint weights from {checkpoint_path}: {e}")
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
        patience_counter = int(ckpt.get('patience_counter', patience_counter))
        history = ckpt.get('history', history)
        print(f"Resuming from checkpoint: {checkpoint_path} (next epoch {start_epoch}/{num_epochs})")

    for epoch in range(start_epoch, num_epochs + 1):
        _apply_lr_schedule(optimizer, lr_schedule, epoch)

        model.train()
        train_loss = 0.0
        train_valid_samples = 0

        for x_dyn_batch, x_static_batch, y_batch, stn_batch in train_loader:
            if _REQUEST_STOP:
                break

            mask = torch.isnan(x_dyn_batch).any(dim=(1, 2)) | torch.isnan(y_batch).any(dim=1)

            if mask.any():
                keep = (~mask).cpu().tolist()
                x_dyn_batch = x_dyn_batch[keep]
                x_static_batch = x_static_batch[keep]
                y_batch = y_batch[keep]
                stn_batch = [s for s, k in zip(stn_batch, keep) if k]

            if len(y_batch) == 0:
                continue

            x_dyn_batch = x_dyn_batch.to(device)
            x_static_batch = x_static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model((x_dyn_batch, x_static_batch))
            if getattr(criterion, 'needs_station_ids', False):
                loss = criterion(outputs, y_batch, stn_batch)
            else:
                loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(x_dyn_batch.size(0))
            train_loss += float(loss.item()) * batch_size
            train_valid_samples += batch_size

        if train_valid_samples == 0:
            raise ValueError('All training samples were filtered due to NaN.')

        train_loss /= train_valid_samples

        model.eval()
        val_loss = 0.0
        val_valid_samples = 0

        with torch.no_grad():
            for x_dyn_batch, x_static_batch, y_batch, stn_batch in validation_loader:
                if _REQUEST_STOP:
                    break

                mask = torch.isnan(x_dyn_batch).any(dim=(1, 2)) | torch.isnan(y_batch).any(dim=1)

                if mask.any():
                    keep = (~mask).cpu().tolist()
                    x_dyn_batch = x_dyn_batch[keep]
                    x_static_batch = x_static_batch[keep]
                    y_batch = y_batch[keep]
                    stn_batch = [s for s, k in zip(stn_batch, keep) if k]

                if len(y_batch) == 0:
                    continue

                x_dyn_batch = x_dyn_batch.to(device)
                x_static_batch = x_static_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model((x_dyn_batch, x_static_batch))
                if getattr(criterion, 'needs_station_ids', False):
                    loss = criterion(outputs, y_batch, stn_batch)
                else:
                    loss = criterion(outputs, y_batch)

                batch_size = int(x_dyn_batch.size(0))
                val_loss += float(loss.item()) * batch_size
                val_valid_samples += batch_size

        if val_valid_samples == 0:
            raise ValueError('All validation samples were filtered due to NaN.')

        val_loss /= val_valid_samples

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

        if checkpoint_path and (((save_every and epoch % int(save_every) == 0)) or _REQUEST_STOP or epoch == num_epochs):
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'history': history,
                'timestamp': time.time(),
            }
            _atomic_torch_save(ckpt, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if _REQUEST_STOP:
            print('Received stop signal; checkpoint saved and training will exit.')
            return history

        if early_stopping and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return history
