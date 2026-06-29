import torchvision
from omegaconf import DictConfig
import torch
import torch.nn as nn


class RangeSolarize(nn.Module):
    """
    Solarize only pixels within a value range, preserving blacks and whites.
    
    Args:
        low: Lower bound of the range to solarize (default 0.3)
        high: Upper bound of the range to solarize (default 0.6)  
        p: Probability of applying the transform (default 0.5)
    """
    def __init__(self, low: float = 0.3, high: float = 0.6, p: float = 0.5):
        super().__init__()
        self.low = low
        self.high = high
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        
        # Create mask for pixels in the target range
        mask = (img >= self.low) & (img <= self.high)
        
        # Solarize (invert) only the masked pixels
        # Inversion within range: new_val = low + high - val
        result = img.clone()
        result[mask] = self.low + self.high - img[mask]
        
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(low={self.low}, high={self.high}, p={self.p})"


class RangeRandomize(nn.Module):
    """
    Replace pixels within a value range with random values, preserving blacks and whites.
    
    Args:
        low: Lower bound of the range to randomize (default 0.3)
        high: Upper bound of the range to randomize (default 0.6)
        random_low: Lower bound for random values (default 0.0)
        random_high: Upper bound for random values (default 1.0)
        p: Probability of applying the transform (default 0.5)
    """
    def __init__(self, low: float = 0.3, high: float = 0.6, 
                 random_low: float = 0.0, random_high: float = 1.0, p: float = 0.5):
        super().__init__()
        self.low = low
        self.high = high
        self.random_low = random_low
        self.random_high = random_high
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        
        # Create mask for pixels in the target range
        mask = (img >= self.low) & (img <= self.high)
        
        # Generate random values in [random_low, random_high]
        random_values = torch.rand_like(img) * (self.random_high - self.random_low) + self.random_low
        
        # Replace masked pixels with random values
        result = img.clone()
        result[mask] = random_values[mask]
        
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(low={self.low}, high={self.high}, random_low={self.random_low}, random_high={self.random_high}, p={self.p})"


class ColorPreservingRandomize(nn.Module):
    """
    Randomize pixels while preserving specific colors (white, black, blue, etc.).
    Works on RGB images with shape (C, H, W) or (B, C, H, W).
    
    Args:
        preserve_white: Whether to preserve white/near-white pixels (default True)
        white_threshold: Min value for all RGB channels to be considered white (default 0.7)
        preserve_black: Whether to preserve black/near-black pixels (default True)
        black_threshold: Max value for all RGB channels to be considered black (default 0.3)
        preserve_blue: Whether to preserve blue pixels (default True)
        blue_dominance: How much B must exceed R and G to be considered blue (default 0.1)
        blue_min: Minimum B value to be considered blue (default 0.3)
        random_low: Lower bound for random values (default 0.0)
        random_high: Upper bound for random values (default 1.0)
        p: Probability of applying the transform (default 0.5)
    """
    def __init__(self, 
                 preserve_white: bool = True,
                 white_threshold: float = 0.7,
                 preserve_black: bool = True, 
                 black_threshold: float = 0.3,
                 preserve_blue: bool = True,
                 blue_dominance: float = 0.1,
                 blue_min: float = 0.3,
                 random_low: float = 0.0,
                 random_high: float = 1.0,
                 p: float = 0.5):
        super().__init__()
        self.preserve_white = preserve_white
        self.white_threshold = white_threshold
        self.preserve_black = preserve_black
        self.black_threshold = black_threshold
        self.preserve_blue = preserve_blue
        self.blue_dominance = blue_dominance
        self.blue_min = blue_min
        self.random_low = random_low
        self.random_high = random_high
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        
        # Handle both (C, H, W) and (B, C, H, W) formats
        if img.dim() == 3:
            R, G, B = img[0], img[1], img[2]
        else:  # (B, C, H, W)
            R, G, B = img[:, 0], img[:, 1], img[:, 2]
        
        # Build preservation mask (True = preserve, don't randomize)
        preserve_mask = torch.zeros_like(R, dtype=torch.bool)
        
        # White: all channels high
        if self.preserve_white:
            white_mask = (R >= self.white_threshold) & (G >= self.white_threshold) & (B >= self.white_threshold)
            preserve_mask = preserve_mask | white_mask
        
        # Black: all channels low
        if self.preserve_black:
            black_mask = (R <= self.black_threshold) & (G <= self.black_threshold) & (B <= self.black_threshold)
            preserve_mask = preserve_mask | black_mask
        
        # Blue: B dominates R and G
        if self.preserve_blue:
            blue_mask = (B >= self.blue_min) & (B > R + self.blue_dominance) & (B > G + self.blue_dominance)
            preserve_mask = preserve_mask | blue_mask
        
        # Randomize mask is the inverse (True = randomize)
        randomize_mask = ~preserve_mask
        
        # Expand mask to all channels
        if img.dim() == 3:
            randomize_mask = randomize_mask.unsqueeze(0).expand(3, -1, -1)
        else:
            randomize_mask = randomize_mask.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # Generate random values
        random_values = torch.rand_like(img) * (self.random_high - self.random_low) + self.random_low
        
        # Apply randomization only to non-preserved pixels
        result = img.clone()
        result[randomize_mask] = random_values[randomize_mask]
        
        return result

    def __repr__(self):
        return (f"{self.__class__.__name__}(preserve_white={self.preserve_white}, white_threshold={self.white_threshold}, "
                f"preserve_black={self.preserve_black}, black_threshold={self.black_threshold}, "
                f"preserve_blue={self.preserve_blue}, blue_dominance={self.blue_dominance}, blue_min={self.blue_min}, p={self.p})")


class RangeGaussianBlur(nn.Module):
    """
    Apply Gaussian blur only to pixels within a value range, preserving blacks and whites.
    Blurs the entire image then blends back original values for pixels outside the range.
    
    Args:
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel (tuple for range, or single value)
        low: Lower bound of the range to blur (default 0.3)
        high: Upper bound of the range to blur (default 0.6)
        p: Probability of applying the transform (default 0.5)
    """
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, 
                 low: float = 0.3, high: float = 0.6, p: float = 0.5):
        super().__init__()
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.low = low
        self.high = high
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        
        # Apply blur to entire image
        blurred = self.blur(img)
        
        # Create mask for pixels in the target range (based on original image)
        mask = (img >= self.low) & (img <= self.high)
        
        # Blend: use blurred values only where mask is True
        result = img.clone()
        result[mask] = blurred[mask]
        
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(low={self.low}, high={self.high}, p={self.p})"


class RangeGaussianNoise(nn.Module):
    """
    Add Gaussian noise only to pixels within a value range, preserving blacks and whites.
    
    Args:
        mean: Mean of the Gaussian noise (default 0.0)
        sigma: Standard deviation of the Gaussian noise (default 0.1)
        low: Lower bound of the range to add noise (default 0.3)
        high: Upper bound of the range to add noise (default 0.6)
        p: Probability of applying the transform (default 0.5)
        clip: Whether to clip output to [0, 1] (default True)
    """
    def __init__(self, mean: float = 0.0, sigma: float = 0.1,
                 low: float = 0.3, high: float = 0.6, p: float = 0.5, clip: bool = True):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.low = low
        self.high = high
        self.p = p
        self.clip = clip

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        
        # Create mask for pixels in the target range
        mask = (img >= self.low) & (img <= self.high)
        
        # Generate noise
        noise = torch.randn_like(img) * self.sigma + self.mean
        
        # Add noise only to masked pixels
        result = img.clone()
        result[mask] = img[mask] + noise[mask]
        
        if self.clip:
            result = result.clamp(0.0, 1.0)
        
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma}, low={self.low}, high={self.high}, p={self.p})"


class ImageAugmentation:
    def __init__(self, shape_meta, transforms: DictConfig):
        self.key_to_transform = {}

        rgb_keys = []
        for obs_key in shape_meta['obs']:
            if shape_meta['obs'][obs_key]['type'] == 'rgb':
                rgb_keys.append(obs_key)

                img_shape = shape_meta['obs'][obs_key]['shape'] # (C, H, W)
                img_size = img_shape[1]

                # compute the transforms for this RGB key
                self.transforms = []
                for transform in transforms:
                    if type(transform) == DictConfig:
                        if transform['type'] == 'RandomCrop':
                            ratio = transform.ratio
                            self.transforms.extend([
                                torchvision.transforms.RandomCrop(size=int(img_size * ratio)),
                                torchvision.transforms.Resize(size=img_size, antialias=True)
                            ])
                        elif transform['type'] == 'CenterCrop':
                            ratio = transform.ratio
                            self.transforms.extend([
                                torchvision.transforms.CenterCrop(size=int(img_size * ratio)),
                                torchvision.transforms.Resize(size=img_size, antialias=True)
                            ])
                        else:
                            raise ValueError(f"Unsupported transform type: {transform['type']}")
                    else:
                        self.transforms.append(transform)

                self.key_to_transform[obs_key] = torch.nn.Sequential(*self.transforms)
    
    def get_transform(self, key) -> torchvision.transforms.Compose:
        return self.key_to_transform[key]
