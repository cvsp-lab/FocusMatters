import torch
import torch.nn.functional as F


VALID_INTERVENTIONS = ("original", "low_mean")


class InterventionManagerLLaVA:
    def __init__(self, model, target_layers,
                 split_ratio=0.25, intervention="low_mean"):
        if intervention not in VALID_INTERVENTIONS:
            raise ValueError(f"Unknown intervention: {intervention}")
        self.source_layers = [6, 7, 8, 9, 10]
        self.target_layers = list(set(target_layers))
        self.split_ratio   = split_ratio
        self.intervention  = intervention

        self.captured_attentions = []
        self.l_mask = None
        self.intv_mask = None

        self.hooks = []
        self.patched_modules_map = {}

        self.vision_layers = self._find_vision_layers(model)
        if intervention != "original":
            self._register_hooks()

        print(f"[InterventionManagerLLaVA] intervention={intervention}, "
              f"split_ratio={split_ratio}, "
              f"target={target_layers}")

    def _find_vision_layers(self, model):
        vision_tower = None
        if hasattr(model, 'get_model'):
            try:
                vision_tower = model.get_model().get_vision_tower()
            except Exception:
                pass
        if vision_tower is None and hasattr(model, 'llama_model'):
            if hasattr(model.llama_model, 'model') and \
               hasattr(model.llama_model.model, 'vision_tower'):
                vision_tower = model.llama_model.model.vision_tower
        if vision_tower is None and hasattr(model, 'model') and \
           hasattr(model.model, 'vision_tower'):
            vision_tower = model.model.vision_tower
        if vision_tower is not None:
            if isinstance(vision_tower, (list, tuple, torch.nn.ModuleList)):
                vision_tower = vision_tower[0]
            if hasattr(vision_tower, 'vision_tower'):
                vision_tower = vision_tower.vision_tower
            if hasattr(vision_tower, 'vision_model') and \
               hasattr(vision_tower.vision_model, 'encoder'):
                return vision_tower.vision_model.encoder.layers
            if hasattr(vision_tower, 'encoder') and \
               hasattr(vision_tower.encoder, 'layers'):
                return vision_tower.encoder.layers
            if hasattr(vision_tower, 'layers'):
                return vision_tower.layers
        raise AttributeError(f"Cannot find vision layers for {type(model)}")

    def _get_attn_module(self, layer):
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        if hasattr(layer, 'attn'):
            return layer.attn
        raise AttributeError(f"No attn module in {type(layer)}")

    def _register_hooks(self):
        for layer_idx in self.source_layers:
            attn_mod = self._get_attn_module(self.vision_layers[layer_idx])
            if attn_mod not in self.patched_modules_map:
                orig = attn_mod.forward
                self.patched_modules_map[attn_mod] = orig

                def _make_src(orig_fwd):
                    def patched(*a, **kw):
                        kw['output_attentions'] = True
                        try:
                            return orig_fwd(*a, **kw)
                        except TypeError:
                            kw.pop('output_attentions', None)
                            return orig_fwd(*a, **kw)
                    return patched

                attn_mod.forward = _make_src(orig)

            self.hooks.append(
                attn_mod.register_forward_hook(self._hook_capture_attn)
            )

        for layer_idx in self.target_layers:
            attn_mod = self._get_attn_module(self.vision_layers[layer_idx])
            if attn_mod in self.patched_modules_map:
                continue
            orig = attn_mod.forward
            self.patched_modules_map[attn_mod] = orig
            attn_mod.forward = self._make_intv_forward(orig, attn_mod, layer_idx)

    def _hook_capture_attn(self, module, input, output):
        attn_weights = None
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
        elif hasattr(output, 'attn_weights'):
            attn_weights = output.attn_weights
        if attn_weights is not None:
            self.captured_attentions.append(attn_weights.detach().mean(dim=1))

    def _compute_masks(self, bsz, N, device):
        if not self.captured_attentions:
            return
        avg_attn = torch.stack(self.captured_attentions, dim=0).mean(dim=0)
        self.captured_attentions = []
        attn_score = avg_attn.sum(dim=1)

        k = max(1, int(N * self.split_ratio))
        l_mask = torch.zeros(bsz, N, dtype=torch.bool, device=device)

        for b in range(bsz):
            sorted_idx = torch.argsort(attn_score[b], descending=True)
            l_mask[b, sorted_idx[N - k:]] = True

        l_mask[:, 0] = False
        self.l_mask = l_mask
        self.intv_mask = l_mask

    def _make_intv_forward(self, orig_fwd, attn_mod, layer_idx):
        mgr = self

        def patched(*args, **kwargs):
            hidden_states = kwargs.get('hidden_states', args[0] if args else None)
            if hidden_states is None:
                return orig_fwd(*args, **kwargs)

            bsz, tgt_len, embed_dim = hidden_states.size()
            device = hidden_states.device

            if mgr.l_mask is None:
                mgr._compute_masks(bsz, tgt_len, device)

            if mgr.intv_mask is None:
                return orig_fwd(*args, **kwargs)

            num_heads = attn_mod.num_heads
            head_dim  = attn_mod.head_dim
            scale     = attn_mod.scale

            output_attentions = kwargs.get('output_attentions', False)

            query_states = attn_mod.q_proj(hidden_states) * scale
            key_states   = attn_mod._shape(attn_mod.k_proj(hidden_states), -1, bsz)
            value_states = attn_mod._shape(attn_mod.v_proj(hidden_states), -1, bsz)

            proj_shape   = (bsz * num_heads, -1, head_dim)
            query_states = attn_mod._shape(query_states, tgt_len, bsz).view(*proj_shape)
            key_states   = key_states.view(*proj_shape)
            value_states_orig = value_states.view(*proj_shape)
            value_states = value_states_orig.clone()

            intv_mask_flat = mgr.intv_mask.unsqueeze(1).expand(-1, num_heads, -1) \
                                          .reshape(bsz * num_heads, tgt_len)

            mean_v = value_states.mean(dim=1, keepdim=True)
            mean_v_exp = mean_v.expand_as(value_states)
            mask3d = intv_mask_flat.unsqueeze(-1).expand_as(value_states)
            value_states = torch.where(mask3d, mean_v_exp, value_states)

            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)

            if output_attentions:
                attn_weights_reshaped = attn_weights.view(bsz, num_heads, tgt_len, tgt_len)
                attn_probs = F.dropout(attn_weights, p=attn_mod.dropout,
                                       training=attn_mod.training)
                attn_probs = attn_probs.view(bsz * num_heads, tgt_len, tgt_len)
            else:
                attn_weights_reshaped = None
                attn_probs = F.dropout(attn_weights, p=attn_mod.dropout,
                                       training=attn_mod.training)

            attn_output = torch.bmm(attn_probs, value_states)
            attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
            attn_output = attn_mod.out_proj(attn_output)

            return attn_output, attn_weights_reshaped

        return patched

    def clear(self):
        self.captured_attentions = []
        self.l_mask = None
        self.intv_mask = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        for mod, orig in self.patched_modules_map.items():
            mod.forward = orig
        self.patched_modules_map = {}
