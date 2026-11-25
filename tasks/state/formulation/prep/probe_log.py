"""Can a LightningModule.training_step run outside Trainer.fit with a constructed Trainer attached?"""
import types, torch, lightning.pytorch as pl

class M(pl.LightningModule):
    def __init__(self): super().__init__(); self.lin = torch.nn.Linear(4, 2)
    def training_step(self, batch, idx):
        out = self.lin(batch).sum()
        self.log("train_loss", out)
        return out
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)
    def forward(self, x): return self.lin(x)

m = M()
x = torch.randn(8, 4)
for label, trainer in (("no trainer", None), ("constructed trainer", pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False))):
    m._trainer = trainer
    try:
        loss = m.training_step(x, 0)
        loss.backward()
        print(f"{label}: training_step OK, loss={float(loss):.4f}")
    except Exception as e:
        print(f"{label}: FAIL {type(e).__name__}: {str(e)[:150]}")
