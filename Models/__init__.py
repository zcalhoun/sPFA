def load(name, kwargs):
    if name == "base":
        from .base import BaseModel

        return BaseModel(**kwargs)
    elif name == "vae_vamp":
        from .vae_vamp import VAEVamp

        return VAEVamp(**kwargs)
