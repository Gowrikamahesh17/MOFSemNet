"""Step 6b: more explainers for the best models.

  * XGBoost            -> TreeSHAP (exact), through shap_analysis.compute_shap_values
  * no-edge GCN        -> SHAP (expected gradients) on the input columns; the network sees one MOF at a time
  * GCN with a graph   -> PyG GNNExplainer: how much of the explanation sits on the MOF's own features vs. its neighbours
  * LIME               -> local linear explanations for Random Forest / XGBoost, to compare with SHAP

GNNs are trained with the "v2" setup that fixed the earlier under-training: z-scored target and z-scored continuous
input columns (statistics from training / non-test MOFs only).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------------------------- GNN training
def _build_model(kind: str, d_in: int, hidden: int = 64):
    """kind: "gcn" (2 GCNConv), "graphsage" (2 PyG SAGEConv; ignores edge weights) or "residual_gcn"
    (MLP + gated GCN term; gate starts at 0). Receptive field: 2 hops, 2 hops, 1 hop."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv

    class GraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1, self.conv2 = SAGEConv(d_in, hidden), SAGEConv(hidden, 1)

        def forward(self, x, edge_index, edge_weight=None):
            h = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
            return self.conv2(h, edge_index)

    class ResidualGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_in, self.conv, self.lin_out = nn.Linear(d_in, hidden), GCNConv(hidden, hidden), nn.Linear(hidden, 1)
            self.gate = nn.Parameter(torch.zeros(1))

        def forward(self, x, edge_index, edge_weight=None):
            h = F.dropout(F.relu(self.lin_in(x)), p=0.3, training=self.training)
            h = h + self.gate * F.relu(self.conv(h, edge_index, edge_weight))
            return self.lin_out(h)

    if kind == "graphsage":
        return GraphSAGE()
    if kind == "residual_gcn":
        return ResidualGCN()

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1, self.conv2 = GCNConv(d_in, hidden), GCNConv(hidden, 1)

        def forward(self, x, edge_index, edge_weight=None):
            h = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)), p=0.3, training=self.training)
            return self.conv2(h, edge_index, edge_weight)

    return GCN()


def standardise_inputs(X: np.ndarray, non_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """z-score the non-binary columns using non-test MOFs only. Returns (X_std, mu, sd, non_binary_mask)."""
    non_binary = ~((X == 0) | (X == 1)).all(axis=0)
    mu = X[non_test][:, non_binary].mean(0)
    sd = X[non_test][:, non_binary].std(0)
    sd = np.where(sd == 0, 1.0, sd)
    out = X.copy()
    out[:, non_binary] = (out[:, non_binary] - mu) / sd
    return out.astype(np.float32), mu, sd, non_binary


HOPS = {"gcn": 2, "graphsage": 2, "residual_gcn": 1}


def train_gcn(X_std, y, edge_index, edge_weight, train_idx, val_idx, seed, epochs=200, lr=0.005, weight_decay=1e-3, patience=20,
              kind="gcn"):
    """Trains a 2-layer GCN (hidden 64) on the standardised target. Returns (model, y_mean, y_std)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std())
    x = torch.tensor(X_std, dtype=torch.float)
    yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float)
    ei = torch.tensor(edge_index, dtype=torch.long)
    ew = torch.tensor(edge_weight, dtype=torch.float) if edge_weight is not None and len(edge_weight) else None
    tr, va = torch.tensor(train_idx), torch.tensor(val_idx)
    model = _build_model(kind, X_std.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        F.mse_loss(model(x, ei, ew).squeeze(-1)[tr], yt[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(x, ei, ew).squeeze(-1)[va], yt[va]).item()
        if vl < best:
            best, wait, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model, y_mean, y_std


def gcn_predict(model, X_std, edge_index, edge_weight, y_mean, y_std) -> np.ndarray:
    import torch

    ew = torch.tensor(edge_weight, dtype=torch.float) if edge_weight is not None and len(edge_weight) else None
    with torch.no_grad():
        return model(torch.tensor(X_std), torch.tensor(edge_index, dtype=torch.long), ew).squeeze(-1).numpy() * y_std + y_mean


# ---------------------------------------------------------------------------------------------- SHAP for the no-edge GNN
def gnn_shap_values(model, X_std, background_idx, explain_idx, y_std, seed=0, nsamples=100):
    """Expected-gradients SHAP for a GCN with NO edges (each MOF is processed independently), in Angstrom.
    Returns {"shap_values": (n_explain, n_features), "additivity_error_mean"/"_max": gap between base + sum(SHAP) and the
    prediction (expected gradients is an approximation, so the gap is reported, not enforced)}."""
    import shap
    import torch

    torch.manual_seed(seed)
    empty = torch.zeros((2, 0), dtype=torch.long)

    class Net(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x, empty).reshape(-1, 1)

    net = Net(model).eval()
    bg = torch.tensor(X_std[background_idx])
    ex = torch.tensor(X_std[explain_idx])
    explainer = shap.GradientExplainer(net, bg)
    values = explainer.shap_values(ex, nsamples=nsamples)
    values = np.asarray(values[0] if isinstance(values, list) else values)
    if values.ndim == 3:
        values = values[..., 0]
    with torch.no_grad():
        base = float(net(bg).mean())
        pred = net(ex).squeeze(-1).numpy()
    gap = np.abs(values.sum(axis=1) + base - pred) * y_std
    return {"shap_values": values * y_std, "additivity_error_mean": float(gap.mean()), "additivity_error_max": float(gap.max())}


# ---------------------------------------------------------------------------------------------- GNNExplainer for the graph GCN
def gnn_explainer_summary(model, X_std, edge_index, edge_weight, node_ids, hops=2, epochs=100, seed=0):
    """PyG GNNExplainer on each node's 2-hop neighbourhood. Per node: share of the feature mask on the node's own row
    (the rest sits on neighbours' features), mean edge mask, number of neighbourhood nodes, and the mask mass per row."""
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph

    torch.manual_seed(seed)
    x = torch.tensor(X_std)
    ei = torch.tensor(edge_index, dtype=torch.long)
    ew = torch.tensor(edge_weight, dtype=torch.float) if edge_weight is not None and len(edge_weight) else None
    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=epochs), explanation_type="model",
                          node_mask_type="attributes", edge_mask_type="object",
                          model_config=dict(mode="regression", task_level="node", return_type="raw"))
    rows, row_masks = [], []
    for node in node_ids:
        subset, sub_ei, mapping, edge_mask = k_hop_subgraph(int(node), hops, ei, relabel_nodes=True, num_nodes=x.size(0))
        sub_ew = ew[edge_mask] if ew is not None else None
        exp = explainer(x[subset], sub_ei, index=int(mapping), edge_weight=sub_ew) if sub_ew is not None else \
            explainer(x[subset], sub_ei, index=int(mapping))
        nm = exp.node_mask.detach().abs().numpy()           # (n_sub, F)
        total = float(nm.sum()) or 1.0
        em = exp.edge_mask.detach().numpy() if getattr(exp, "edge_mask", None) is not None and exp.edge_mask.numel() else np.zeros(1)
        rows.append({"node": int(node), "n_neighbourhood_nodes": int(len(subset)), "n_edges": int(sub_ei.size(1)),
                     "share_on_own_features": float(nm[int(mapping)].sum() / total), "mean_edge_mask": float(em.mean())})
        row_masks.append(nm[int(mapping)])
    return rows, np.stack(row_masks)


# ---------------------------------------------------------------------------------------------- LIME
def lime_explanations(predict_fn, X_train, X_explain, feature_names, categorical_idx, num_features=15, num_samples=15000, seed=0):
    """Local linear explanations (LIME, regression) for each row of X_explain. Returns a list of {feature_index: weight}
    (only the num_features selected features are non-zero) and the LIME local-model R2 per row."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(X_train, feature_names=list(feature_names), categorical_features=list(categorical_idx),
                                     mode="regression", discretize_continuous=True, random_state=seed)
    out, scores = [], []
    for i in range(len(X_explain)):
        exp = explainer.explain_instance(X_explain[i], predict_fn, num_features=num_features, num_samples=num_samples)
        # for regression LIME stores the weights under key 1 and a NEGATED copy under key 0
        out.append({int(f): float(w) for f, w in exp.local_exp[1]})
        scores.append(float(exp.score))
    return out, scores


def group_shares(values: np.ndarray, groups: list[str], net: bool = False) -> dict[str, float]:
    """Share of the explanation per feature group. net=False: summed per-column mean |value| (as in Step 6a);
    net=True: mean over rows of |sum of the group's values| (the group's net contribution to each prediction)."""
    groups = np.asarray(groups)
    if net:
        per = {g: float(np.abs(values[:, groups == g].sum(axis=1)).mean()) for g in dict.fromkeys(groups)}
    else:
        mean_abs = np.abs(values).mean(axis=0)
        per = {g: float(mean_abs[groups == g].sum()) for g in dict.fromkeys(groups)}
    total = sum(per.values()) or 1.0
    return {g: v / total for g, v in per.items()}


# ---------------------------------------------------------------------------------------------- explainers for GNNs WITH a graph
def node_subgraph(edge_index, edge_weight, node, hops, n_nodes):
    """k-hop neighbourhood of `node`, relabelled. Returns (subset, sub_edge_index, sub_edge_weight or None, target position)."""
    import torch
    from torch_geometric.utils import k_hop_subgraph

    ei = torch.tensor(edge_index, dtype=torch.long)
    subset, sub_ei, mapping, edge_mask = k_hop_subgraph(int(node), hops, ei, relabel_nodes=True, num_nodes=n_nodes)
    sub_ew = torch.tensor(edge_weight, dtype=torch.float)[edge_mask] if edge_weight is not None and len(edge_weight) else None
    return subset, sub_ei, sub_ew, int(mapping)


def graph_gnn_shap(model, X_std, edge_index, edge_weight, nodes, hops, pool_idx, y_std, n_background=10, nsamples=50, seed=0):
    """SHAP (expected gradients) for GNNs with a graph. For each node the model input is the feature matrix of its k-hop
    neighbourhood (n_sub x n_features, flattened); the background is random rows from the training pool. Returns per node
    the attribution of the node's OWN features (n_features,) and the share of the total |attribution| that sits on the
    neighbours' features, in Angstrom."""
    import shap
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    F_ = X_std.shape[1]
    rows = []
    for node in nodes:
        subset, sub_ei, sub_ew, target = node_subgraph(edge_index, edge_weight, node, hops, X_std.shape[0])
        n_sub = len(subset)

        class Net(torch.nn.Module):
            def forward(self, xflat):
                outs = [model(xflat[b].view(n_sub, F_), sub_ei, sub_ew)[target].reshape(1) for b in range(xflat.shape[0])]
                return torch.stack(outs)

        net = Net().eval()
        ex = torch.tensor(X_std[subset.numpy()]).reshape(1, -1)
        bg = torch.tensor(np.stack([X_std[rng.choice(pool_idx, size=n_sub)].reshape(-1) for _ in range(n_background)]))
        values = np.asarray(shap.GradientExplainer(net, bg).shap_values(ex, nsamples=nsamples))
        values = values.reshape(n_sub, F_)
        with torch.no_grad():
            gap = abs(float(values.sum()) + float(net(bg).mean()) - float(net(ex))) * y_std
        mass = np.abs(values).sum()
        rows.append({"node": int(node), "n_sub": n_sub, "own": values[target] * y_std,
                     "share_on_neighbours": float(1 - np.abs(values[target]).sum() / mass) if mass else 0.0,
                     "additivity_gap": gap})
    return rows


def graph_predict_fn(model, X_std, edge_index, edge_weight, node, hops, y_mean, y_std, chunk=400):
    """Returns predict(rows) -> predicted PLD (A) of `node` when its own feature row is replaced by each row of `rows`
    (neighbours keep their real features). Evaluates many perturbed copies in one batched forward pass."""
    import torch

    subset, sub_ei, sub_ew, target = node_subgraph(edge_index, edge_weight, node, hops, X_std.shape[0])
    base = torch.tensor(X_std[subset.numpy()])
    n_sub = len(subset)

    def predict(rows):
        rows = torch.tensor(np.asarray(rows, dtype=np.float32))
        out = []
        for start in range(0, len(rows), chunk):
            r = rows[start:start + chunk]
            m = len(r)
            x = base.repeat(m, 1)
            x[torch.arange(m) * n_sub + target] = r
            off = (torch.arange(m) * n_sub)[:, None, None]
            ei = (sub_ei[None] + off).permute(1, 0, 2).reshape(2, -1)
            ew = sub_ew.repeat(m) if sub_ew is not None else None
            with torch.no_grad():
                out.append(model(x, ei, ew).squeeze(-1)[torch.arange(m) * n_sub + target].numpy())
        return np.concatenate(out) * y_std + y_mean

    return predict


def lime_one(predict_fn, X_train, x, feature_names, categorical_idx, num_features=15, num_samples=5000, seed=0):
    """LIME for a single row with a given predict function (weights under local_exp[1], see lime_explanations)."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(X_train, feature_names=list(feature_names), categorical_features=list(categorical_idx),
                                     mode="regression", discretize_continuous=True, random_state=seed)
    exp = explainer.explain_instance(x, predict_fn, num_features=num_features, num_samples=num_samples)
    return {int(f): float(w) for f, w in exp.local_exp[1]}, float(exp.score)


def group_shapley_exact(predict_rows, X_explain, X_background, group_indices):
    """EXACT Shapley values over feature GROUPS (e.g. pore / fingerprint / metal) for a model that predicts each row on its
    own. v(S) = mean prediction when the groups in S take the explained row's values and the others take background values.
    With 3 groups that is 8 coalitions per row, so the result is exact (given the background), unlike expected gradients.
    Returns (n_rows, n_groups) contributions."""
    import itertools
    import math

    g = len(group_indices)
    out = np.zeros((len(X_explain), g))
    for r, x in enumerate(X_explain):
        value = {}
        for mask in itertools.product((0, 1), repeat=g):
            z = X_background.copy()
            for gi, on in enumerate(mask):
                if on:
                    z[:, group_indices[gi]] = x[group_indices[gi]]
            value[mask] = float(predict_rows(z).mean())
        for gi in range(g):
            phi = 0.0
            for mask in itertools.product((0, 1), repeat=g):
                if mask[gi]:
                    continue
                size = sum(mask)
                w = math.factorial(size) * math.factorial(g - size - 1) / math.factorial(g)
                with_g = tuple(1 if i == gi else m for i, m in enumerate(mask))
                phi += w * (value[with_g] - value[mask])
            out[r, gi] = phi
    return out
