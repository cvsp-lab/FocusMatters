import torch
import torch.nn.functional as F
import gc


class DPPMaskManager:
    def __init__(self, model, visual_token_num,
                 source_layers=[12],
                 feature_layer=12,
                 target_layers=[13, 14, 15, 16, 17]):

        self.visual_token_num = visual_token_num
        self.source_layers = list(set(source_layers))
        self.feature_layer = feature_layer
        self.target_layers = list(set(target_layers))

        self.captured_attentions = []
        self.captured_features = None
        self.generated_mask = None
        self.hooks = []

        self.patched_modules_map = {}

        self.vision_layers = self._find_vision_layers(model)
        self._register_hooks()

    def _find_vision_layers(self, model):
        """LLaVA 모델에서 Vision Transformer의 Layer 리스트를 찾습니다."""
        vision_tower = None

        if hasattr(model, 'get_model'):
            try:
                vision_tower = model.get_model().get_vision_tower()
            except:
                pass

        if vision_tower is None and hasattr(model, 'llama_model'):
            if hasattr(model.llama_model, 'model') and hasattr(model.llama_model.model, 'vision_tower'):
                vision_tower = model.llama_model.model.vision_tower

        if vision_tower is None and hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
            vision_tower = model.model.vision_tower

        if vision_tower is not None:
            if isinstance(vision_tower, (list, tuple, torch.nn.ModuleList)):
                vision_tower = vision_tower[0]

            if hasattr(vision_tower, 'vision_tower'):
                vision_tower = vision_tower.vision_tower

            if hasattr(vision_tower, 'vision_model') and hasattr(vision_tower.vision_model, 'encoder'):
                return vision_tower.vision_model.encoder.layers

            if hasattr(vision_tower, 'encoder') and hasattr(vision_tower.encoder, 'layers'):
                return vision_tower.encoder.layers

            if hasattr(vision_tower, 'layers'):
                return vision_tower.layers

        raise AttributeError(f"Could not find vision layers for model type: {type(model)}.")

    def _register_hooks(self):
        self.hooks.append(
            self.vision_layers[self.feature_layer].register_forward_hook(self._hook_capture_feature)
        )

        for layer_idx in self.source_layers:
            self._patch_forward(layer_idx)
            attn_module = self.vision_layers[layer_idx].self_attn
            self.hooks.append(
                attn_module.register_forward_hook(self._hook_capture_attn)
            )

        for layer_idx in self.target_layers:
            self._patch_forward(layer_idx)

    def _patch_forward(self, layer_idx):
        layer = self.vision_layers[layer_idx]
        attn_module = layer.self_attn

        if attn_module in self.patched_modules_map:
            return

        current_forward = attn_module.forward
        self.patched_modules_map[attn_module] = current_forward

        def patched_forward(*args, **kwargs):
            if layer_idx in self.source_layers:
                kwargs['output_attentions'] = True

            if layer_idx in self.target_layers:
                if self.generated_mask is None:
                    self._compute_dpp_mask()

                if self.generated_mask is not None:
                    if 'attention_mask' in kwargs:
                        current_mask = kwargs['attention_mask']
                        kwargs['attention_mask'] = (current_mask + self.generated_mask) if current_mask is not None else self.generated_mask
                    elif len(args) > 1:
                        args = list(args)
                        current_mask = args[1]
                        if current_mask is None:
                            args[1] = self.generated_mask
                        else:
                            args[1] = current_mask + self.generated_mask
                        args = tuple(args)

            return current_forward(*args, **kwargs)

        attn_module.forward = patched_forward

    def _hook_capture_feature(self, module, input, output):
        if isinstance(output, tuple):
            self.captured_features = output[0].detach()
        else:
            self.captured_features = output.detach()

    def _hook_capture_attn(self, module, input, output):
        attn_weights = None
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]

        if attn_weights is not None:
            self.captured_attentions.append(attn_weights.detach().mean(dim=1))

    def _compute_dpp_mask(self):
        if self.captured_features is None or not self.captured_attentions:
            return

        image_features = self.captured_features
        B, N, C = image_features.shape
        device = image_features.device

        image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
        image_normalized = image_normalized.float()
        similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))

        avg_attn = torch.stack(self.captured_attentions, dim=0).mean(dim=0)
        self.captured_attentions = []

        attn_score = avg_attn.sum(dim=1)
        kernel = attn_score.unsqueeze(2) * similarity * attn_score.unsqueeze(1)

        cis = torch.zeros((self.visual_token_num, B, N), device=device)
        di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
        select_idx = torch.empty((self.visual_token_num, B), dtype=torch.long, device=device)

        for i in range(self.visual_token_num):
            j = torch.argmax(di2s, dim=-1)
            select_idx[i] = j

            eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
                / (torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1) + 1e-6)
            cis[i, :, :] = eis
            di2s -= torch.square(eis)
            di2s[torch.arange(B), j] = -float('inf')

        select_idx = select_idx.t()

        mask = torch.full((B, N), float('-inf'), device=device)
        mask.scatter_(1, select_idx, 0.0)
        mask[:, 0] = 0.0  # CLS Token Protect

        self.generated_mask = mask.view(B, 1, 1, N).expand(-1, -1, N, -1)

        del image_features, similarity, kernel, cis, di2s
        gc.collect()
        torch.cuda.empty_cache()

    def clear(self):
        self.captured_attentions = []
        self.captured_features = None
        self.generated_mask = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

        for module, old_forward in self.patched_modules_map.items():
            module.forward = old_forward
        self.patched_modules_map = {}
