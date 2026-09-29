import torch
import torch.nn.functional as F


VALID_INTERVENTIONS = ("original", "low_mean")


class InterventionManagerInternVL:
    def __init__(self, model, target_layers,
                 split_ratio=0.25, intervention="low_mean"):
        if intervention not in VALID_INTERVENTIONS:
            raise ValueError(f"Unknown intervention: {intervention}")
        self.source_layers = [6, 7, 8]
        self.target_layers = list(set(int(t) for t in target_layers))
        self.split_ratio   = split_ratio
        self.intervention  = intervention

        self.captured_attentions = []
        self.intv_indices = None

        self.patched_modules_map = {}
        self.vision_layers = self._find_vision_layers(model)
        if intervention != "original":
            self._register_patches()

        print(f"[InterventionManagerInternVL] intervention={intervention}, "
              f"split_ratio={split_ratio}, "
              f"target={self.target_layers}")

    def _find_vision_layers(self, model):
        if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder"):
            return model.vision_model.encoder.layers
        raise AttributeError("Could not find model.vision_model.encoder.layers")

    def _get_attn_module(self, layer):
        if hasattr(layer, "attn"):
            return layer.attn
        raise AttributeError("Layer has no 'attn' module")

    def _register_patches(self):
        all_layers = sorted(set(self.source_layers) | set(self.target_layers))
        for layer_idx in all_layers:
            attn_mod = self._get_attn_module(self.vision_layers[layer_idx])
            if attn_mod in self.patched_modules_map:
                continue
            orig = attn_mod.forward
            self.patched_modules_map[attn_mod] = orig
            attn_mod.forward = self._make_patched_forward(orig, attn_mod, layer_idx)

    def _compute_indices(self, seq_length, device):
        if not self.captured_attentions:
            return
        avg = torch.stack(self.captured_attentions, dim=0).float().mean(dim=0)
        attn_score = avg.sum(dim=1).mean(dim=0)
        self.captured_attentions = []

        k = max(1, int(seq_length * self.split_ratio))

        attn_score_protected = attn_score.clone()
        attn_score_protected[0] = float("inf")

        sorted_idx = torch.argsort(attn_score_protected, descending=True)
        l_indices = sorted_idx[seq_length - k:]
        l_indices = l_indices[l_indices != 0]
        self.intv_indices = l_indices.to(device)

    def _make_patched_forward(self, orig_fwd, attn_mod, layer_idx):
        mgr = self

        def patched(x: torch.Tensor):
            B, N, C = x.shape
            num_heads = attn_mod.num_heads
            head_dim  = attn_mod.head_dim

            qkv = attn_mod.qkv(x).reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if hasattr(attn_mod, "q_norm") and hasattr(attn_mod, "k_norm"):
                q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)

            scale = getattr(attn_mod, "scale", 1.0 / (head_dim ** 0.5))
            q_scaled = q * scale
            attn_weights = q_scaled @ k.transpose(-2, -1)

            if layer_idx in mgr.target_layers:
                if mgr.intv_indices is None:
                    mgr._compute_indices(N, v.device)

                if mgr.intv_indices is not None and mgr.intv_indices.numel() > 0:
                    idx = mgr.intv_indices
                    mean_v = v.mean(dim=2, keepdim=True)
                    v[:, :, idx, :] = mean_v.expand(-1, -1, idx.size(0), -1)

            attn_probs = attn_weights.softmax(dim=-1)

            if layer_idx in mgr.source_layers:
                with torch.no_grad():
                    avg_attn = attn_probs.mean(dim=1)
                    mgr.captured_attentions.append(avg_attn.detach())

            attn_probs = attn_mod.attn_drop(attn_probs)

            out = attn_probs @ v
            out = out.transpose(1, 2).reshape(B, N, C)
            out = attn_mod.proj(out)
            out = attn_mod.proj_drop(out)
            return out

        return patched

    def clear(self):
        self.captured_attentions = []
        self.intv_indices = None

    def remove_hooks(self):
        for mod, orig in self.patched_modules_map.items():
            mod.forward = orig
        self.patched_modules_map = {}
