import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_known_prototypes(model, train_loader, num_known_classes, device):
    """
    Compute known-class prototypes from training features.

    Prototype:
        mean normalized feature for each known class
    """
    model.eval()

    feats_by_class = [[] for _ in range(num_known_classes)]

    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"]

        feat, _ = model(x)
        feat = F.normalize(feat, dim=1)

        for i in range(x.size(0)):
            cls = int(y[i].item())
            if 0 <= cls < num_known_classes:
                feats_by_class[cls].append(feat[i].detach().cpu())

    prototypes = []

    for k in range(num_known_classes):
        if len(feats_by_class[k]) == 0:
            raise RuntimeError(f"No samples found for known class {k}")

        p = torch.stack(feats_by_class[k], dim=0).mean(dim=0)
        p = F.normalize(p, dim=0)
        prototypes.append(p)

    prototypes = torch.stack(prototypes, dim=0)

    return prototypes


@torch.no_grad()
def prototype_distance_unknown_scores(model, loader, prototypes, device):
    """
    Prototype distance baseline.

    Unknown score:
        1 - max cosine similarity to known prototypes

    Larger score means more likely to be unknown.
    """
    model.eval()

    prototypes = prototypes.to(device)

    y_true = []
    scores = []

    for batch in loader:
        x = batch["x"].to(device)
        open_set_y = batch["open_set_y"]

        feat, _ = model(x)
        feat = F.normalize(feat, dim=1)

        sim = feat @ prototypes.T
        max_sim = sim.max(dim=1).values

        unknown_score = 1.0 - max_sim

        y_true.extend(open_set_y.cpu().numpy().tolist())
        scores.extend(unknown_score.cpu().numpy().tolist())

    return y_true, scores