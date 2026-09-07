import copy
import torch
import numpy as np
import math
import sys
import tqdm
import time

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from .primitive_fitting_utils import triangulate_and_mesh

# Training settings every pipeline fits an INR with. The fit_inr signature
# defaults below are the library's own and are deliberately weaker.
INR_NETWORK_PARAMETERS = {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5}
INR_MAX_STEPS          = 5000
INR_NOISE_MAGNITUDE_3D = 0.05
INR_NOISE_MAGNITUDE_UV = 0.05
INR_INITIAL_LR         = 1e-1


def inr_fit_kwargs(max_steps=INR_MAX_STEPS, seed=None):
    kw = {
        "max_steps": max_steps,
        "noise_magnitude_3d": INR_NOISE_MAGNITUDE_3D,
        "noise_magnitude_uv": INR_NOISE_MAGNITUDE_UV,
        "initial_lr": INR_INITIAL_LR,
    }
    if seed is not None:
        kw["seed"] = seed
    return kw


def encoder_to_uv(output, is_closed):
    # [B, 2] => [B, 1], depending on open/closed parameter configuration
    res =  torch.atan2(output[:, 0], output[:, 1]) / np.pi if is_closed else torch.nn.functional.tanh(output[:, 0])
    return res.unsqueeze(-1)

def uv_to_decoder(output, is_closed):
    # [B, 1] => [B, 2]
    if is_closed:
        output *= np.pi
        latent_1 = output.cos()
        latent_2 = output.sin()
        res = torch.cat([latent_1, latent_2], dim = -1)

    else:
        # Point2CAD replicates in the function definition but zero-pads during training: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L592
        res = torch.cat([output, torch.zeros_like(output)], dim = -1)

    return res

def inr_recon_loss(X, Xhat):
    # No reduce operation, the caller reduces if needed
    return torch.abs(X - Xhat).sum(dim = -1)

def inr_error(data_loader, model, cluster_mean, cluster_scale):
    # Point2CAD trains with L1 but reports the INR fitness error in L2; the mean is scale-invariant, so the norm is taken in normalized space and rescaled once
    was_training = model.training
    model.eval()
    total_error = None
    total_points = 0
    try:
        with torch.no_grad():
            for batch in data_loader:
                X = batch[0]
                Xhat, _ = model.forward(X)
                per_point = torch.linalg.norm(X - Xhat, dim=-1)
                batch_sum = per_point.sum()
                total_error = batch_sum if total_error is None else total_error + batch_sum
                total_points += per_point.shape[0]
    finally:
        if was_training:
            model.train()

    return ((total_error / total_points) * cluster_scale).item()

def automatic_batch_size(N, max_memory_mb = 10):
    max_allowed = math.ceil((max_memory_mb * (2 ** 20)) / 3)
    return min(N, max_allowed)

class CustomLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        bound_weight = kwargs.pop("bound_weight", None)
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            if bound_weight is not None:
                self.weight.uniform_(-bound_weight, bound_weight)

class SiLUBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, use_shortcut = True):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.weight = torch.nn.Parameter(torch.ones((1,)))
        self.linear = torch.nn.Linear(dim_in, dim_out)
        self.norm = torch.nn.BatchNorm1d(dim_out)

        if use_shortcut:
            self.residual_map = torch.nn.Identity() if dim_in == dim_out else torch.nn.Linear(dim_in, dim_out)

    def forward(self, X):
        shortcut = X
        X = self.linear(X)
        X = self.norm(X)
        X = torch.nn.functional.silu(X)

        if self.use_shortcut:
            X = (self.weight * X + self.residual_map(shortcut)) / 2 ** 0.5

        return X

class SIRENBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, angular_freq = 30):
        super().__init__()
        bound_weight = 1 / dim_in

        # Bias off here: Point2CAD computes sin(w(Ax + b)) while SIREN computes sin(wAx + b), so the bias is applied manually
        self.linear = CustomLinear(dim_in, dim_out, bound_weight = bound_weight, bias = False)
        self.bias = torch.nn.Parameter(torch.zeros((dim_out,)))
        self.angular_freq = angular_freq

    def forward(self, X):
        X = self.linear(X)
        X = torch.sin(self.angular_freq * X + self.bias)
        return X

class INREncoder(torch.nn.Module):
    def __init__(self, hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fraction_siren = fraction_siren
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed

        self.siren_dim = int(hidden_dim * fraction_siren)
        self.silu_dim = hidden_dim - self.siren_dim

        self.siren_layer = SIRENBlock(3, self.siren_dim)
        self.silu_layer = SiLUBlock(3, self.silu_dim, use_shortcut = use_shortcut)
        self.last_linear = torch.nn.Linear(hidden_dim, 4) # Maps to [u1, u2, v1, v2]

    def forward(self, X):
        # [B, 3] => [B, 2]
        X_siren = self.siren_layer(X) # [B, siren_dim]
        X_silu = self.silu_layer(X) # [B, silu_dim] siren_dim + silu_dim = hidden_dim

        X = torch.concat([X_siren, X_silu], dim = -1)
        X = self.last_linear(X)

        # Encoder-to-UV rules are not in the paper: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L758
        X_u = X[:, :2]
        X_v = X[:, 2:]

        U = encoder_to_uv(X_u, self.is_u_closed) # [B, 1]
        V = encoder_to_uv(X_v, self.is_v_closed) # [B, 1]

        return torch.cat([U, V], dim = -1)

class INRDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fraction_siren = fraction_siren
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed

        self.siren_dim = int(hidden_dim * fraction_siren)
        self.silu_dim = hidden_dim - self.siren_dim

        self.siren_layer = SIRENBlock(4, self.siren_dim)
        self.silu_layer = SiLUBlock(4, self.silu_dim, use_shortcut = use_shortcut)
        self.last_linear = torch.nn.Linear(hidden_dim, 3)

    def forward(self, X):
        # [B, 2] => [B, 3]
        U_lifted = uv_to_decoder(X[:, [0]], self.is_u_closed) # [B, 2]
        V_lifted = uv_to_decoder(X[:, [1]], self.is_v_closed) # [B, 2]
        UV = torch.cat([U_lifted, V_lifted], dim = -1) # [B, 4]

        X_siren = self.siren_layer(UV) # [B, siren_dim]
        X_silu = self.silu_layer(UV) # [B, silu_dim], siren_dim + silu_dim = hidden_dim

        X = torch.cat([X_siren, X_silu], dim = -1) # [B, hidden_dim]
        return self.last_linear(X)

class INRNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = False):
        super().__init__()
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed
        self.encoder = INREncoder(hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = use_shortcut)
        self.decoder = INRDecoder(hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = use_shortcut)

    def forward(self, X, cluster_mean = None, cluster_scale = None):
        if cluster_mean is not None and cluster_scale is not None:
            cluster_mean = torch.tensor(cluster_mean, device = X.device)
            cluster_scale = torch.tensor(cluster_scale, device = X.device)
            X = (X - cluster_mean) / cluster_scale

        uv = self.encoder(X)
        Xhat = self.decoder(uv)

        if cluster_mean is not None and cluster_scale is not None:
            Xhat = Xhat * cluster_scale + cluster_mean

        return Xhat, uv

    def forward_encoder(self, X, cluster_mean = None, cluster_scale = None):        
        if cluster_mean is not None and cluster_scale is not None:
            cluster_mean = torch.tensor(cluster_mean, device = X.device)
            cluster_scale = torch.tensor(cluster_scale, device = X.device)
            X = (X - cluster_mean) / cluster_scale

        return self.encoder(X)

    def forward_decoder(self, uv):
        Xhat = self.decoder(uv)        
        return Xhat

    def sample_points(self, mesh_dim, uv_bb_min, uv_bb_max, cluster_mean, cluster_scale, uv_margin = 0):
        uv_length = uv_bb_max - uv_bb_min
        uv_bb_min_extended = uv_bb_min - uv_length * uv_margin
        uv_bb_max_extended = uv_bb_max + uv_length * uv_margin

        if self.is_u_closed:
            uv_bb_min_extended[0] = max(uv_bb_min_extended[0], -1)
            uv_bb_max_extended[0] = min(uv_bb_max_extended[0], 1)

        if self.is_v_closed:
            uv_bb_min_extended[1] = max(uv_bb_min_extended[1], -1)
            uv_bb_max_extended[1] = min(uv_bb_max_extended[1], 1)

        device = next(self.parameters()).device

        u, v = torch.meshgrid(
            torch.linspace(uv_bb_min_extended[0], uv_bb_max_extended[0], mesh_dim, device = device),
            torch.linspace(uv_bb_min_extended[1], uv_bb_max_extended[1], mesh_dim, device = device),
            indexing = "ij"
        )

        # Cartesian product of two linspaces, where the first coordinate moves faster.
        uv = torch.stack((u, v), dim = 2).reshape(-1, 2)
        with torch.no_grad():
            X = self.forward_decoder(uv)
            cluster_mean = torch.tensor(cluster_mean, device = device)
            cluster_scale = torch.tensor(cluster_scale, device = device)
            X = X * cluster_scale + cluster_mean

        return X

    def sample_mesh(self, mesh_dim, uv_bb_min, uv_bb_max, cluster, cluster_mean, cluster_scale,
                    uv_margin = 0.1, threshold_multiplier = 3, spacing = None,
                    uv_points = None, alpha = 10.0):
        device = next(self.parameters()).device
        points = self.sample_points(mesh_dim, uv_bb_min, uv_bb_max, cluster_mean, cluster_scale, uv_margin = uv_margin)

        mask = None
        meshes = triangulate_and_mesh(points.cpu().numpy(), mesh_dim, mesh_dim, "inr", mask = mask)
        return meshes

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min = 0.0, last_epoch = -1):
        assert max_steps > warmup_steps, "max_steps must be greater than warmup_steps"

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min # minimum achievable learning rate, will practically always be 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cos_term = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cos_term
            for base_lr in self.base_lrs
        ]

def fit_inr_single(network_parameters, device, dl, dl_generator, cluster_mean, cluster_scale,
            steps_per_epoch,
            is_u_closed = False,
            is_v_closed = False,
            max_steps = 1000,
            warmup_steps_ratio = 0.05,
            initial_lr = 1e-2,
            noise_magnitude_3d = 0.005,
            noise_magnitude_uv = 0.005,
            eval_every = 5):
    start = time.time()

    model = INRNetwork(**network_parameters, is_u_closed = is_u_closed, is_v_closed = is_v_closed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr  = initial_lr)
    warmup_steps = int(max_steps * warmup_steps_ratio)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)

    # Separate eval loader: shuffle=False and no generator so mid-training eval does not perturb the training iterator's RNG
    eval_dl = DataLoader(dl.dataset, batch_size=dl.batch_size,
                         shuffle=False, drop_last=False)

    loop = tqdm.tqdm(range(max_steps), desc="Training the INR network",
                     disable=not sys.stderr.isatty())

    best_error = float("inf")
    best_state_dict = None
    best_step = -1

    for i, _ in enumerate(loop):
        noise_schedule = (max_steps - 1 - i) / (max_steps - 1)

        optimizer.zero_grad()

        X = next(dl_generator)
        X_original = X[0]
        if noise_magnitude_3d == 0:
            X_noised = X_original

        else:
            noise_x = torch.randn(size = X_original.shape, device = device)
            X_noised = X_original + noise_magnitude_3d * noise_schedule * noise_x

        uv = model.forward_encoder(X_noised)

        if noise_magnitude_uv != 0:
            noise_uv = torch.randn(size = uv.shape, device = device)
            uv = uv + noise_magnitude_uv * noise_schedule * noise_uv

        Xhat = model.forward_decoder(uv)
        recon_loss = inr_recon_loss(X_original, Xhat).mean()
        recon_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % eval_every == 0 or i == max_steps - 1:
            step_error = inr_error(eval_dl, model, cluster_mean, cluster_scale)
            if step_error < best_error:
                best_error = step_error
                best_step = i
                best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.cuda.synchronize()
    end = time.time()

    error = inr_error(eval_dl, model, cluster_mean, cluster_scale)
    if best_state_dict is not None:
        tqdm.tqdm.write(f"  [inr] best_error={best_error:.6f}  final_error={error:.6f}  at step {best_step}/{max_steps - 1}")

    result = {
        "surface_type": "inr",
        "error": error,
        "params": {
            "network_parameters": network_parameters,
            "model": model,
        },
        "metadata": {
            "fitting_time_seconds": end - start
        }
    }

    return result


def fit_inr(cluster, network_parameters, device = "cuda:0",
            max_steps = 1000,
            warmup_steps_ratio = 0.05,
            initial_lr = 1e-2,
            max_memory_mb = 10,
            noise_magnitude_3d = 0.005,
            noise_magnitude_uv = 0.005,
            seed = 42):

    if not torch.cuda.is_available():
        device = "cpu"

    N = cluster.shape[0]
    batch_size = automatic_batch_size(N, max_memory_mb = max_memory_mb)
    steps_per_epoch = math.ceil(N / batch_size)
    tqdm.tqdm.write(f"  [inr] batch_size={batch_size}  steps_per_epoch={steps_per_epoch}")

    cluster_mean = cluster.mean(axis = 0)
    cluster_std = cluster.std(axis = 0)
    # Dividing by the maximum STD preserves aspect ratios between coordinates; z-score standardization would not
    cluster_scale = cluster_std.max()
    # Account for the 1e-6 factor when passing points through the INR at inference
    cluster = (cluster - cluster_mean) / (cluster_scale + 1e-6)
    cluster_mean_torch = torch.tensor(cluster_mean, dtype = torch.float32).to(device)
    cluster_scale_torch = torch.tensor(cluster_scale, dtype = torch.float32).to(device)
    cluster = torch.tensor(cluster, device = device)
    dataset = TensorDataset(cluster)
    best_model = None

    torch.backends.cudnn.deterministic = True

    inr_t0 = time.time()
    for u in [True, False]:
        for v in [True, False]:
            # Global seed: same init, noise and batch order for every closedness combination
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            dl_gen = torch.Generator()
            dl_gen.manual_seed(seed)
            dl = DataLoader(dataset, batch_size = batch_size, drop_last = False, shuffle = True, generator = dl_gen)

            def get_next_item():
                while True:
                    for X in dl:
                        yield X

            dl_generator = get_next_item()
            current_model = fit_inr_single(network_parameters, device, dl, dl_generator, cluster_mean_torch, cluster_scale_torch,
                                           steps_per_epoch,
                                           is_u_closed = u,
                                           is_v_closed = v,
                                           max_steps = max_steps,
                                           warmup_steps_ratio = warmup_steps_ratio,
                                           initial_lr = initial_lr,
                                           noise_magnitude_3d = noise_magnitude_3d,
                                           noise_magnitude_uv = noise_magnitude_uv)
            fit_time = current_model["metadata"]["fitting_time_seconds"]
            tqdm.tqdm.write(f"  [inr] u_closed={u}  v_closed={v}  best_error={current_model['error']:.6f}  time={fit_time:.2f}s")
            if best_model is None or best_model["error"] > current_model["error"]:
                best_model = current_model  
    inr_total = time.time() - inr_t0
    tqdm.tqdm.write(f"  [inr] best: error={best_model['error']:.6f}  total_time={inr_total:.2f}s")
    best_model["params"]["cluster_mean"] = cluster_mean
    best_model["params"]["cluster_scale"] = cluster_scale
    model = best_model["params"]["model"]

    uvs = []
    with torch.no_grad():
        for X in dl:
            X = X[0]
            uv = model.forward_encoder(X).cpu().numpy()
            uvs.append(uv)

    uvs = np.concatenate(uvs, axis = 0)
    uv_bb_min = uvs.min(axis = 0)
    uv_bb_max = uvs.max(axis = 0)

    best_model["params"]["uv_bb_min"] = uv_bb_min
    best_model["params"]["uv_bb_max"] = uv_bb_max
    best_model["params"]["uv_points"] = uvs

    return best_model
