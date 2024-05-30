from noise.ladp import LADP
from noise.dae import DenoisingAutoEncoder


def create_noise_transform(name, **kwargs):
    """
    Denoising pretraining transform entrypoint, allows to create transform just with
    parameters, without using its class
    """
    transforms = [
        LADP,
        DenoisingAutoEncoder
    ]
    trans_dict = {a().name.lower(): a for a in transforms}
    try:
        model_class = trans_dict[name.lower()]
    except KeyError:
        raise KeyError(
            "Wrong transform type `{}`. Available options are: {}".format(
                name,
                list(trans_dict.keys()),
            )
        )
    return model_class(**kwargs)
