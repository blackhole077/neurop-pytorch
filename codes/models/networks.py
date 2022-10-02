from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Operator(nn.Module):
    """A class encapsulation of the Neural Color Operator Block, defined by Wang et al.

    This block uses an encoder-decoder structure, along with manipulations in the feature space to replicate the pixel-based color mapping found in many image editing softwares.


    Note About Convolutional Layer(s)
    ---------------------------------
    As I was originally inspecting this code I noticed that, while the authors described their neural color operator blocks using fully-connected layers (which would be described in PyTorch as nn.Linear layers),
    the actual code appears to use 2D Convolutional layers with 1x1 kernels (and 1-stride). Naturally, I initially thought this to be incorrect and dug further. According to the sources listed below, the 2D Convolutional layers are indeed functioning as 'fully-connected' layers:
    (https://datascience.stackexchange.com/questions/12830/how-are-1x1-convolutions-the-same-as-a-fully-connected-layer)
    (https://sebastianraschka.com/faq/docs/fc-to-conv.html)
    With the second explaining more clearly how this actually works.
    NOTE: For the proposed method to work, the input must be of shape (batch_size, num_channels, 1, 1)
    """

    def __init__(self, in_nc=3, out_nc=3, base_nf=64):
        super(Operator, self).__init__()
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1)
        self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1)
        self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, operator_strength):
        x_code = self.encoder(x)
        # Perform a translation in the feature space by the scalar amount before decoding
        y_code = x_code + operator_strength
        # Transform the translated feature vector and then take the output of applying a leaky ReLU as the new feature vector to decode
        y_code = self.act(self.mid_conv(y_code))
        # Decode the feature vector back into its original space
        y = self.decoder(y_code)
        return y


class Renderer(nn.Module):
    """A class encapsulation of the series of neural color operators applied to an image, implicitly defined by Wang et al. In their paper, this can be understood to be the 'Sequential Image Retouching Pipeline'.
    
    

    """

    def __init__(self, in_nc=3, out_nc=3, base_nf=64):
        super(Renderer, self).__init__()
        self.in_nc = in_nc
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.exposure_operator_block = Operator(in_nc, out_nc, base_nf)
        self.black_clipping_operator_block = Operator(in_nc, out_nc, base_nf)
        self.vibrance_operator_block = Operator(in_nc, out_nc, base_nf)

    def forward(
        self,
        x_exposure,
        x_black_clipping,
        x_vibrance,
        exposure_strength_scalar,
        black_clipping_strength_scalar,
        vibrance_strength_scalar,
    ):

        rec_exposure = self.exposure_operator_block(x_exposure, 0)
        rec_black_clipping = self.black_clipping_operator_block(x_black_clipping, 0)
        rec_vibrance = self.vibrance_operator_block(x_vibrance, 0)

        map_exposure = self.exposure_operator_block(
            x_exposure, exposure_strength_scalar
        )
        map_black_clipping = self.black_clipping_operator_block(
            x_black_clipping, black_clipping_strength_scalar
        )
        map_vibrance = self.vibrance_operator_block(
            x_vibrance, vibrance_strength_scalar
        )

        return (
            rec_exposure,
            rec_black_clipping,
            rec_vibrance,
            map_exposure,
            map_black_clipping,
            map_vibrance,
        )


class Encoder(nn.Module):
    """A class encapsulation of the encoding structure used by the Strength Predictor, defined by Wang et al.

    """    
    def __init__(self, in_nc=3, encode_nf=32):
        super(Encoder, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, encode_nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(encode_nf, encode_nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.max = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        # batch_size, _, _, _ = x.size()
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        # Get the standard deviation and mean for each pixel
        std, mean = torch.std_mean(conv2_out, dim=[2, 3], keepdim=False)
        # Get the maximum value across the channel dimension for all images in the batch?
        maxs = self.max(conv2_out).squeeze(2).squeeze(2)
        # out is presumed to be a 96-dimensional Tensor according to the structure of the paper.
        out = torch.cat([std, mean, maxs], dim=1)
        return out


class StrengthPredictor(nn.Module):
    """A class encapsulation of the Strength Predictor, defined by Wang et al.

    The purpose of the Strength Predictor is to determine the scalar weight to apply to the pixel-based color mapping function the predictor is attached to.
    It does this by taking in a down-sampled version of the current image, encoding it and then running it through one fully-connected (FC) layer with a tanh activation function.
    In their paper, Wang et al. set the number of input features of the FC layer to be 96, as they multiply the number of encoding features (32), by 3.

    Attributes
    ----------
    fc3: nn.Linear
        A fully-connected (FC) layer that takes in a developer-defined number of features as input and outputs one feature.
    tanh: nn.Tanh
        A tanh activation function used by the FC layer to output a scalar to apply to the associated pixel-based color mapping function.
    """

    def __init__(self, fea_dim):
        super(StrengthPredictor, self).__init__()
        self.fc3 = nn.Linear(fea_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, img_fea: torch.Tensor) -> torch.Tensor:

        # val is assumed to be a 1D (i.e., scalar) tensor.
        val = self.tanh(self.fc3(img_fea))
        return val


class NeurOP(nn.Module):
    """A class encapsulation of the Neural color OPerator, defined by Wang et al.

    The purpose of the neural color operator, per Wang et al. is " [...] to mimic the behavior of traditional global color operators."

    The authors claim this is done by learning a pixel-wise conversion function between the input RGB color pixel (i.e., 3D vector representing the color of a pixel)
    along with a 1D scalar vector representing the intensity of the operation to some output RGB color pixel. Since this method is implicitly based on learning a mapping
    between two spaces, the authors use similarity and noise-based loss functions for the learning process.

    Attributes
    ----------
    in_nc: int
        The number of channels in the input. By default, this is set to 3 for the RGB colorspace.
    out_nc: int
        The number of channels in the output. By default, this is set to 3 for the RGB colorspace.
    base_nf: int
        The size of the feature vector that each pixel is transformed into before encoding. By default, this is set to 64.
    encode_nf: int
        The size of the encoded feature vector that is used when predicting the strength of any given neural color operator. By default, this is set to 32.
    load_path: Optional[str]
        A file path that points to a state dict file that is used by the renderer.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        base_nf: int = 64,
        encode_nf: int = 32,
        load_path: Optional[str] = None,
    ):
        super(NeurOP, self).__init__()
        self.fea_dim = encode_nf * 3
        self.image_encoder = Encoder(in_nc, encode_nf)
        renderer = Renderer(in_nc, out_nc, base_nf)
        if load_path is not None:
            renderer.load_state_dict(torch.load(load_path))

        self.black_clipping_renderer = renderer.black_clipping_operator_block
        self.black_clipping_predictor = StrengthPredictor(self.fea_dim)

        self.exposure_renderer = renderer.exposure_operator_block
        self.exposure_predictor = StrengthPredictor(self.fea_dim)

        self.vibrance_renderer = renderer.vibrance_operator_block
        self.vibrance_predictor = StrengthPredictor(self.fea_dim)

        self.renderers: List[Operator] = [
            self.black_clipping_renderer,
            self.exposure_renderer,
            self.vibrance_renderer,
        ]
        self.predict_heads: List[StrengthPredictor] = [
            self.black_clipping_predictor,
            self.exposure_predictor,
            self.vibrance_predictor,
        ]

    def render(self, x, vals):
        b, _, h, w = img.shape
        imgs = []
        for nop, scalar in zip(self.renderers, vals):
            img = nop(img, scalar)
            output_img = torch.clamp(img, 0, 1.0)
            imgs.append(output_img)
        return imgs

    def forward(self, image_to_transform:torch.Tensor, return_vals:bool=True):
        b, _, h, w = image_to_transform.shape
        vals = []
        for neural_operator, strength_predictor in zip(self.renderers, self.predict_heads):
            downscaled_image = F.interpolate(
                input=image_to_transform,
                size=(256, int(256 * w / h)),
                mode="bilinear",
                align_corners=False,
            )
            pixel_feature_vector:torch.Tensor = self.image_encoder(downscaled_image)
            scalar:torch.Tensor = strength_predictor(pixel_feature_vector)
            vals.append(scalar)
            image_to_transform = neural_operator(image_to_transform, scalar)
        # Clip the pixel value into the [0, 1] range
        image_to_transform = torch.clamp(image_to_transform, 0, 1.0)
        if return_vals:
            return image_to_transform, vals
        else:
            return image_to_transform
