import torch
import torch.nn.functional as F


@torch.no_grad()
def msp_unknown_scores(model, loader, device):
    """
    MSP baseline.

    Unknown score:
        1 - max softmax probability over known classes

    Larger score means more likely to be unknown.

    Expected batch format:
        batch["x"]: [B, 2, L]
        batch["open_set_y"]: 0 = known, 1 = unknown
    """
    model.eval()

    y_true = []
    scores = []

    for batch in loader:
        x = batch["x"].to(device)
        open_set_y = batch["open_set_y"]

        _, logits = model(x)

        probs = F.softmax(logits, dim=1)
        max_prob = probs.max(dim=1).values

        unknown_score = 1.0 - max_prob

        y_true.extend(open_set_y.cpu().numpy().tolist())
        scores.extend(unknown_score.cpu().numpy().tolist())

    return y_true, scores