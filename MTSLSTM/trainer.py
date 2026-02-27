import torch
import wandb
from tqdm import tqdm


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
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class train_model:
    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, device, config):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config

        self.early_stopping = EarlyStopping(config.EARLY_STOPPING_PATIENCE)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        #loader = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        loader = self.train_loader

        for batch in loader:
            x_dict, y, _ = batch

            H = x_dict["H"].to(self.device)
            D = x_dict["D"].to(self.device)
            S = x_dict["S"].to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model({"H": H, "D": D}, S)
            loss = self.criterion(outputs["H"].unsqueeze(-1), y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * H.size(0)
            #loader.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader.dataset)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        #loader = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        loader = self.val_loader

        with torch.no_grad():
            for batch in loader:
                x_dict, y, _ = batch

                H = x_dict["H"].to(self.device)
                D = x_dict["D"].to(self.device)
                S = x_dict["S"].to(self.device)
                y = y.to(self.device)

                outputs = self.model({"H": H, "D": D}, S)
                loss = self.criterion(outputs["H"].unsqueeze(-1), y)

                total_loss += loss.item() * H.size(0)
                #loader.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader.dataset)

    def fit(self):
        for epoch in range(1, self.config.NUM_EPOCHS + 1):

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            print(
                f"Epoch {epoch}/{self.config.NUM_EPOCHS} "
                f"| Train Loss: {train_loss:.6f} "
                f"| Val Loss: {val_loss:.6f}"
            )

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)

            if self.config.USE_EARLY_STOPPING:
                improved = self.early_stopping.step(val_loss)

                if improved:
                    torch.save(self.model.state_dict(),
                            self.config.BEST_MODEL_PATH)
                    print("✅ Best model saved.")

                if self.early_stopping.early_stop:
                    print("🛑 Early stopping triggered.")
                    break

        print("🎉 Training finished.")