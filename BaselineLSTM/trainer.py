import torch
from tqdm import tqdm
import wandb


def train_model(model,
                train_loader,
                validation_loader,
                optimizer,
                criterion,
                device,
                num_epochs,
                patience=10,
                best_model_path="best_model.pth",
                early_stopping=False):

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):

        # ======================
        # 1️⃣ Train
        # ======================
        model.train()

        train_loss = 0
        train_valid_samples = 0

        #train_loader_iter = tqdm(
            #train_loader,
            #desc=f"Epoch {epoch} [Train]",
            #leave=False
        #)
        train_loader_iter=train_loader

        for x_dyn_batch, x_static_batch, y_batch, stn_batch in train_loader_iter:

            mask = torch.isnan(x_dyn_batch).any(dim=(1, 2)) | \
                   torch.isnan(y_batch).any(dim=1)

            if mask.any():
                x_dyn_batch = x_dyn_batch[~mask]
                x_static_batch = x_static_batch[~mask]
                y_batch = y_batch[~mask]

            if len(y_batch) == 0:
                continue

            x_dyn_batch = x_dyn_batch.to(device)
            x_static_batch = x_static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model((x_dyn_batch, x_static_batch))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = x_dyn_batch.size(0)

            train_loss += loss.item() * batch_size
            train_valid_samples += batch_size

            #train_loader_iter.set_postfix(loss=loss.item())

        if train_valid_samples == 0:
            raise ValueError("All training samples were filtered due to NaN.")

        train_loss /= train_valid_samples

        # ======================
        # 2️⃣ Validation
        # ======================
        model.eval()

        val_loss = 0
        val_valid_samples = 0

        #val_loader_iter = tqdm(
            #validation_loader,
            #desc=f"Epoch {epoch} [Val]",
            #leave=False
        #)
        val_loader_iter = validation_loader

        with torch.no_grad():
            for x_dyn_batch, x_static_batch, y_batch, stn_batch in val_loader_iter:

                mask = torch.isnan(x_dyn_batch).any(dim=(1, 2)) | \
                       torch.isnan(y_batch).any(dim=1)

                if mask.any():
                    x_dyn_batch = x_dyn_batch[~mask]
                    x_static_batch = x_static_batch[~mask]
                    y_batch = y_batch[~mask]

                if len(y_batch) == 0:
                    continue

                x_dyn_batch = x_dyn_batch.to(device)
                x_static_batch = x_static_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model((x_dyn_batch, x_static_batch))
                loss = criterion(outputs, y_batch)

                batch_size = x_dyn_batch.size(0)

                val_loss += loss.item() * batch_size
                val_valid_samples += batch_size

                #val_loader_iter.set_postfix(loss=loss.item())

        if val_valid_samples == 0:
            raise ValueError("All validation samples were filtered due to NaN.")

        val_loss /= val_valid_samples

        # ======================
        # 3️⃣ Logging
        # ======================
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })

        # ======================
        # 4️⃣ Early Stopping
        # ======================
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # save the best model
                torch.save(model.state_dict(), best_model_path)

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return history