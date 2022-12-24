def load(name, kwargs):
    if name == "base":
        from .base import BaseModel

        return BaseModel(**kwargs)
    elif name == "vae_vamp":
        from .vae_vamp import VAEVamp

        return VAEVamp(**kwargs)
    elif name == "deep_encoder":
        from .deep_encoder import DeepEncoder

        return DeepEncoder(**kwargs)
    else:
        raise NotImplementedError("Unknown model name: {}".format(name))
