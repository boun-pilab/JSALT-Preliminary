import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2GroupNormConvLayer,
    Wav2Vec2LayerNormConvLayer,
    Wav2Vec2NoLayerNormConvLayer,
)

class Sign2VecLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_channels = config.conv_3d_channels[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_channels = config.conv_3d_channels[layer_id]

        self.conv = nn.Conv3d(
            self.in_conv_channels,
            self.out_conv_channels,
            kernel_size=config.conv_3d_kernel[layer_id],
            stride=config.conv_3d_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_channels, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states

class Sign2VecNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_channels = config.conv_3d_channels[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_channels = config.conv_3d_channels[layer_id]

        self.conv = nn.Conv3d(
            self.in_conv_channels,
            self.out_conv_channels,
            kernel_size=config.conv_3d_kernel[layer_id],
            stride=config.conv_3d_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states
    
class Sign2VecGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_channels = config.conv_3d_channels[layer_id - 1] if layer_id > 0 else 3
        self.out_conv_channels = config.conv_3d_channels[layer_id]

        self.conv = nn.Conv3d(
            self.in_conv_channels,
            self.out_conv_channels,
            kernel_size=config.conv_3d_kernel[layer_id],
            stride=config.conv_3d_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_channels, num_channels=self.out_conv_channels, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Sign2VecFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        # 3D Convolutional Layers - to spatio-temporally downsample the input
        if config.feat_extract_norm == "group":
            conv_layers = [Sign2VecGroupNormConvLayer(config, layer_id=0)] + [
                Sign2VecNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_3d_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Sign2VecLayerNormConvLayer(config, layer_id=i) for i in range(config.num_3d_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        
        self.conv_3d_layers = nn.ModuleList(conv_layers)

        # NOTE: Reduced initial transformation layer to hidden_size since applied at 3D-Conv output
        if config.feat_extract_norm == "group":
            conv_layers =  [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        
        self.conv_layers = nn.ModuleList(conv_layers)

        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, hidden_states):
        # hidden_states: (batch_size, channels, time_steps, height, width)
        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for ix, conv_layer in enumerate(self.conv_3d_layers):
            
            print(f'3d LAYER: {ix}')
            print('LAYER_INPUT:', hidden_states.shape)  
            print('CONV_LAYER:',conv_layer)
            print('-------------------')
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        hidden_states = hidden_states.transpose(1,2)
        # merge (channel) and (height, width) dimensions
        hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        hidden_states = hidden_states.transpose(1,2)


        for ix, conv_layer in enumerate(self.conv_layers):

            print(f'1d LAYER: {ix}')
            print('LAYER_INPUT:', hidden_states.shape)  
            print('CONV_LAYER:',conv_layer)
            print('-------------------')
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)


        return hidden_states