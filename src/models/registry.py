"""
Model registry: models register themselves by name and are built from the config.

    @register_model("resnet1d")
    class ResNet1D(nn.Module): ...

    model = build_model(cfg["model"], input_names=["ecg", "ppg"], n_targets=2, n_aux=0)

Every model's forward(x) returns a dict:
    "bp"    (B, n_targets)  standardised SBP/DBP predictions        (required)
    "aux"   (B, n_aux)      auxiliary predictions                    (if n_aux > 0)
    "gates" (B, n_targets)  per-target fusion gates, for analysis    (optional)
"""
MODELS = {}


def register_model(name):
    def deco(cls):
        if name in MODELS:
            raise KeyError(f"Model '{name}' registered twice")
        MODELS[name] = cls
        cls.registry_name = name
        return cls
    return deco


def build_model(model_cfg, input_names, n_targets=2, n_aux=0):
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODELS)}")
    return MODELS[name](input_names=list(input_names), n_targets=n_targets, n_aux=n_aux, **cfg)
