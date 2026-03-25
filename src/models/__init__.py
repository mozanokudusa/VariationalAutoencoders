from .base import VariationalAutoencoder
from .encoders.conv2 import Conv2Encoder
from .decoders.conv2 import Conv2Decoder
from .encoders.conv3 import Conv3Encoder
from .decoders.conv3 import Conv3Decoder

def get_model(config):
    """
    Factory function to instantiate the model based on the config.
    
    Args:
        config (dict): The 'model' section of your YAML configuration.
        
    Returns:
        model (nn.Module): The initialized Variational Autoencoder.
    """
    # Map strings to classes
    encoder_registry = {
        "conv2": Conv2Encoder,
        "conv3": Conv3Encoder, 
    }
    
    decoder_registry = {
        "conv2": Conv2Decoder,
        "conv3": Conv3Decoder,
    }

    # 2. Extract settings from config
    model_cfg = config['model']
    enc_type = model_cfg.get('encoder_type', 'conv2')
    dec_type = model_cfg.get('decoder_type', 'conv2')
    
    # Universal parameters passed to both components
    params = {
        "input_channels": model_cfg['input_channels'],
        "image_size": model_cfg['image_size'],
        "hidden_channels": model_cfg['capacity'],
        "latent_dim": model_cfg['latent_dims']
    }

    # 3. Instantiate the components
    if enc_type not in encoder_registry or dec_type not in decoder_registry:
        raise ValueError(f"Unsupported architecture: {enc_type}/{dec_type}")

    encoder = encoder_registry[enc_type](**params)
    
    # Decoders usually use 'output_channels', so we swap the key name
    dec_params = params.copy()
    dec_params['output_channels'] = dec_params.pop('input_channels')
    decoder = decoder_registry[dec_type](**dec_params)

    # 4. Wrap in the VAE logic
    model = VariationalAutoencoder(encoder, decoder)
    
    return model