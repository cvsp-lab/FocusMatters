import math
import torch
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision


VALID_INTERVENTIONS = ("original", "low_mean")


class InterventionManagerQwen:
    def __init__(self, model, target_layers,
                 split_ratio=0.25, intervention="low_mean"):
        if intervention not in VALID_INTERVENTIONS:
            raise ValueError(f"Unknown intervention: {intervention}")
        self.source_layers = [12, 13, 14, 15, 16, 17, 18]
        self.target_layers = list(set(int(t) for t in target_layers))
        self.split_ratio   = split_ratio
        self.intervention  = intervention

        self.captured_attentions = []
        self.l_indices = None
        self.intv_indices = None

        self.patched_modules_map = {}
        self.vision_layers = self._find_vision_layers(model)
        if intervention != "original":
            self._register_patches()

        print(f"[InterventionManagerQwen] intervention={intervention}, "
              f"split_ratio={split_ratio}, target={target_layers}")

    def _find_vision_layers(self, model):
        if hasattr(model, 'visual') and hasattr(model.visual, 'blocks'):
            return model.visual.blocks
        raise AttributeError("model.visual.blocks not found.")

    def _get_attn_module(self, layer):
        if hasattr(layer, 'attn'):
            return layer.attn
        raise AttributeError("layer.attn not found.")

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
        avg = torch.stack(self.captured_attentions, dim=0).mean(dim=0).to(device)
        attn_score = avg.float().sum(dim=0)
        self.captured_attentions = []
        k = max(1, int(seq_length * self.split_ratio))
        sorted_idx = torch.argsort(attn_score, descending=True)
        self.l_indices = sorted_idx[seq_length - k:]
        self.intv_indices = self.l_indices

    def _make_patched_forward(self, orig_fwd, attn_mod, layer_idx):
        mgr = self

        def patched(hidden_states, cu_seqlens,
                    rotary_pos_emb=None, position_embeddings=None):
            seq_length = hidden_states.shape[0]
            num_heads  = attn_mod.num_heads
            head_dim   = attn_mod.proj.in_features // num_heads

            qkv = attn_mod.qkv(hidden_states)
            q, k, v = (qkv.reshape(seq_length, 3, num_heads, head_dim)
                          .permute(1, 0, 2, 3)
                          .unbind(0))

            if position_embeddings is None:
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos = emb.cos().float()
                sin = emb.sin().float()
            else:
                cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

            base_mask = torch.zeros([1, seq_length, seq_length],
                                     device=q.device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                s, e = cu_seqlens[i-1], cu_seqlens[i]
                base_mask[..., s:e, s:e] = True

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

            if layer_idx in mgr.source_layers:
                with torch.no_grad():
                    aw = (torch.matmul(q.float(), k.float().transpose(-2, -1))
                          / math.sqrt(head_dim))
                    aw.masked_fill_(~base_mask, float('-inf'))
                    aw = F.softmax(aw, dim=-1)
                    avg_aw = aw.mean(dim=0)
                    mgr.captured_attentions.append(avg_aw.detach())

            if layer_idx in mgr.target_layers:
                if mgr.intv_indices is None:
                    mgr._compute_indices(seq_length, q.device)

                if mgr.intv_indices is not None:
                    idx = mgr.intv_indices
                    mean_v = v.mean(dim=1, keepdim=True)
                    v[:, idx, :] = mean_v.expand(-1, idx.size(0), -1)

            attn_output = F.scaled_dot_product_attention(q, k, v, base_mask, dropout_p=0.0)
            attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
            attn_output = attn_mod.proj(attn_output)
            return attn_output

        return patched

    def clear(self):
        self.captured_attentions = []
        self.l_indices = None
        self.intv_indices = None

    def remove_hooks(self):
        for mod, orig in self.patched_modules_map.items():
            mod.forward = orig
        self.patched_modules_map = {}
