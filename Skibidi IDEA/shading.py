import numpy as np
from numba import njit, prange
import cv2
import pygame as pg


@njit
def apply_grayscale(arr):
    """Apply grayscale filter to the given image array."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    return arr

@njit
def apply_sepia(arr):
    """Apply sepia filter to the given image array."""
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    tr = (r * 0.393) + (g * 0.769) + (b * 0.189)
    tg = (r * 0.349) + (g * 0.686) + (b * 0.168)
    tb = (r * 0.272) + (g * 0.534) + (b * 0.131)
    
    np.clip(tr, 0, 255, out=tr)
    np.clip(tg, 0, 255, out=tg)
    np.clip(tb, 0, 255, out=tb)
    
    arr[:, :, 0] = tr
    arr[:, :, 1] = tg
    arr[:, :, 2] = tb
    return arr

@njit
def apply_negate(arr):
    """Apply negate (invert) effect to the given image array."""
    arr[:, :, 0] = 255 - arr[:, :, 0]
    arr[:, :, 1] = 255 - arr[:, :, 1]
    arr[:, :, 2] = 255 - arr[:, :, 2]
    return arr

@njit
def apply_blur(arr, kernel_size=3):
    """Apply a simple blur effect using a kernel."""
    rows, cols, channels = arr.shape
    pad_size = kernel_size // 2
    padded_arr = np.pad(arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    result = np.zeros_like(arr)
    
    for i in range(rows):
        for j in range(cols):
            kernel = padded_arr[i:i + kernel_size, j:j + kernel_size, :]
            result[i, j] = np.mean(kernel, axis=(0, 1))
    return result

@njit(fastmath=True)  # Use fastmath optimization
def apply_water_ripple(arr, intensity=5, speed=0.1, block_size=1, smoothness=0.1):
    """Simulate a water ripple effect with smoothness and block processing."""
    rows, cols, channels = arr.shape
    result = np.empty_like(arr)

    # Iterate over the image in blocks of size block_size
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Define block boundaries
            end_i = min(i + block_size, rows)
            end_j = min(j + block_size, cols)

            # Calculate displacement for this block using smoother transitions
            displacement = np.sin(i * speed + j * speed) * intensity + np.cos(i * smoothness + j * smoothness) * intensity

            # Apply displacement to each pixel in the block
            for x in range(i, end_i):
                for y in range(j, end_j):
                    # Calculate new_y with displacement and ensure it's an integer
                    new_y = y + int(displacement)  # Ensure displacement is integer

                    # Ensure new_y stays within bounds of the image (0 to cols-1)
                    if new_y < 0:
                        new_y = 0
                    elif new_y >= cols:
                        new_y = cols - 1

                    # Copy the pixel to the displaced position
                    for c in range(channels):
                        result[x, y, c] = arr[x, new_y, c]

    return result

@njit(fastmath=True)  # Enable fastmath for performance optimization
def apply_chromatic_aberration(image, shift_red=5, shift_green=3, shift_blue=-2):
    """
    Apply a chromatic aberration effect to an image by shifting color channels.

    Parameters:
        image (numpy.ndarray): Input image array with shape (rows, cols, channels).
        shift_red (int): Horizontal pixel shift for the red channel.
        shift_green (int): Horizontal pixel shift for the green channel.
        shift_blue (int): Horizontal pixel shift for the blue channel.

    Returns:
        numpy.ndarray: Image with chromatic aberration effect applied.
    """
    rows, cols, channels = image.shape
    result = np.zeros_like(image)

    for x in range(rows):
        for y in range(cols):
            # Shift red channel
            new_y_red = y + shift_red
            if 0 <= new_y_red < cols:
                result[x, y, 0] = image[x, new_y_red, 0]
            
            # Shift green channel
            new_y_green = y + shift_green
            if 0 <= new_y_green < cols:
                result[x, y, 1] = image[x, new_y_green, 1]
            
            # Shift blue channel
            new_y_blue = y + shift_blue
            if 0 <= new_y_blue < cols:
                result[x, y, 2] = image[x, new_y_blue, 2]

    return result

@njit(fastmath=True)
def apply_bulge(arr, strength=1, center_strength=0.1):
    """Apply a bulge effect to the image with reduced distortion at the center."""
    rows, cols, _ = arr.shape
    center_x, center_y = cols // 2, rows // 2
    result = arr.copy()  # If you want to keep the original array intact

    for i in range(rows):
        for j in range(cols):
            dx, dy = j - center_x, i - center_y
            distance_sq = dx * dx + dy * dy  # Avoid sqrt for efficiency
            distance = np.sqrt(distance_sq)  # Calculate the distance using the squared values
            
            # Apply distortion based on distance from the center
            factor = 1 + (strength * (distance / np.sqrt(rows**2 + cols**2))) * (1 - center_strength)

            # Calculate the new x and y positions based on the bulge factor
            new_x = int(center_x + dx * factor)
            new_y = int(center_y + dy * factor)

            # Manually clip the new positions to stay within bounds
            if new_x < 0:
                new_x = 0
            elif new_x >= cols:
                new_x = cols - 1

            if new_y < 0:
                new_y = 0
            elif new_y >= rows:
                new_y = rows - 1

            # Assign the pixel from the original array to the bulged position
            result[i, j] = arr[new_y, new_x]
    
    return result

@njit
def apply_pinch(arr, strength=10):
    """Apply pinch effect to the image."""
    rows, cols, _ = arr.shape
    center_x, center_y = cols // 2, rows // 2
    result = arr.copy()
    
    for i in range(rows):
        for j in range(cols):
            dx, dy = j - center_x, i - center_y
            distance = np.sqrt(dx**2 + dy**2)
            factor = 1 / (1 + strength / (distance + 1))
            new_x = int(center_x + dx * factor)
            new_y = int(center_y + dy * factor)
            new_x = np.clip(new_x, 0, cols - 1)
            new_y = np.clip(new_y, 0, rows - 1)
            result[i, j] = arr[new_y, new_x]
    return result
  
@njit(fastmath=True, parallel=True)
def apply_radial_blur(arr, max_radius=100, strength=5):
    """Optimized radial blur effect using cumulative sums for performance."""
    rows, cols, channels = arr.shape
    center_x, center_y = cols // 2, rows // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)

    # Precompute cumulative sum for each channel
    cumsum = np.zeros((rows + 1, cols + 1, channels), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                cumsum[i + 1, j + 1, k] = (
                    arr[i, j, k] +
                    cumsum[i, j + 1, k] +
                    cumsum[i + 1, j, k] -
                    cumsum[i, j, k]
                )

    result = np.empty_like(arr)

    for i in prange(rows):  # Parallelize across rows
        for j in range(cols):
            dx = j - center_x
            dy = i - center_y
            distance = np.sqrt(dx**2 + dy**2)

            # Calculate blur factor
            blur_factor = int(strength * (distance / max_distance))
            blur_factor = max(1, blur_factor)  # Minimum kernel size of 1

            # Determine kernel bounds
            start_x = max(0, j - blur_factor)
            end_x = min(cols, j + blur_factor + 1)
            start_y = max(0, i - blur_factor)
            end_y = min(rows, i + blur_factor + 1)

            # Use cumulative sum to calculate region average
            for k in range(channels):
                region_sum = (
                    cumsum[end_y, end_x, k]
                    - cumsum[start_y, end_x, k]
                    - cumsum[end_y, start_x, k]
                    + cumsum[start_y, start_x, k]
                )
                count = (end_y - start_y) * (end_x - start_x)
                result[i, j, k] = region_sum / count

    return result.astype(np.uint8)
  

@njit(fastmath=True)
def apply_pixelation(arr, pixel_size=10):
    """Optimized pixelation effect using block average."""
    rows, cols, channels = arr.shape

    # Result array to store the pixelated image
    result = np.empty_like(arr)

    # Iterate over the image in blocks of size pixel_size
    for i in range(0, rows, pixel_size):  # Non-parallelized loop
        for j in range(0, cols, pixel_size):
            # Define the block boundaries
            end_i = min(i + pixel_size, rows)
            end_j = min(j + pixel_size, cols)

            # Calculate the average color of the block
            block_sum = np.zeros(channels, dtype=np.float64)
            count = 0
            for x in range(i, end_i):
                for y in range(j, end_j):
                    for c in range(channels):
                        block_sum[c] += arr[x, y, c]
                    count += 1

            # Assign the average color to the entire block
            avg_color = block_sum / count
            for x in range(i, end_i):
                for y in range(j, end_j):
                    result[x, y] = avg_color

    return result.astype(np.uint8)





@njit(fastmath=True)
def apply_lens_circle(arr, radius=100, strength=1.5):
    """Apply vignette effect to dim the edges of the image."""
    rows, cols, channels = arr.shape
    center_x, center_y = cols // 2, rows // 2

    # Precompute maximum distance for normalization
    max_distance = np.sqrt(center_x**2 + center_y**2)

    result = arr.copy()
    for i in range(rows):
        for j in range(cols):
            dx = j - center_x
            dy = i - center_y
            distance = np.sqrt(dx**2 + dy**2)

            # Compute attenuation factor
            attenuation = 1 - (strength * (distance / max_distance))
            attenuation = max(0, attenuation)  # Ensure it's non-negative

            # Apply attenuation to each channel
            for k in range(channels):
                result[i, j, k] = arr[i, j, k] * attenuation

    return result

def apply_gaussian_blur(image, blur_radius):
    """
    Apply Gaussian blur to the given image using OpenCV.
    """
    kernel_size = blur_radius * 2 + 1  # Ensure kernel size is odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

@njit(fastmath=True)
def bloom_effect_core(image, threshold, intensity):
    """
    Core bloom effect logic compatible with Numba.
    """
    # Calculate grayscale manually (average across the 3 color channels)
    rows, cols, channels = image.shape
    grayscale = np.empty((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            grayscale[i, j] = (image[i, j, 0] + image[i, j, 1] + image[i, j, 2]) / 3

    # Create a mask for bright areas
    bright_mask = (grayscale > threshold).astype(np.float32)

    # Multiply bright mask with the original image to isolate bright regions
    bright_areas = np.zeros_like(image, dtype=np.float32)
    for c in range(3):  # Process each channel
        for i in range(rows):
            for j in range(cols):
                bright_areas[i, j, c] = image[i, j, c] * bright_mask[i, j]

    return bright_areas


def apply_bloom_effect(image, threshold=200, intensity=1.5, blur_radius=15):
    """
    Apply a bloom effect to an image by creating a blurred glow around bright areas.
    """
    # Ensure the image is valid and preprocess if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack([image] * 3, axis=-1)

    # Perform the core bloom effect operations
    bright_areas = bloom_effect_core(image, threshold, intensity)

    # Apply Gaussian blur to the bright areas
    blurred_bright_areas = apply_gaussian_blur(bright_areas, blur_radius)

    # Add the blurred bright areas back to the original image with intensity scaling
    result = image.astype(np.float32) + blurred_bright_areas * intensity

    # Clip values to ensure they remain within valid range (0-255)
    return np.clip(result, 0, 255).astype(np.uint8)


class MangoShading:
    def __init__(self):
        self.cached_array = None

    def apply_effect(self, surface, effect_name, **kwargs):
        """Apply the selected effect to the given surface."""
        arr = pg.surfarray.pixels3d(surface)
        
        if effect_name == 'grayscale':
            arr = apply_grayscale(arr)
        elif effect_name == 'sepia':
            arr = apply_sepia(arr)
        elif effect_name == 'negate':
            arr = apply_negate(arr)
        elif effect_name == 'water_ripple':
            arr = apply_water_ripple(arr, **kwargs)
        elif effect_name == 'bulge':
            arr = apply_bulge(arr, **kwargs)
        elif effect_name == 'pinch':
            arr = apply_pinch(arr, **kwargs)
        elif effect_name == 'lens_circle':
            arr = apply_lens_circle(arr, **kwargs)
        elif effect_name == 'blur':
            arr = apply_blur(arr, **kwargs)
        elif effect_name == "radial_blur":
            arr = apply_radial_blur(arr, **kwargs)
        elif effect_name == "pixelization":
            arr = apply_pixelation(arr, **kwargs)
        elif effect_name == "chromatic_aberration":
            arr = apply_chromatic_aberration(arr, **kwargs)
        elif effect_name == "bloom":
            arr = apply_bloom_effect(arr, **kwargs)
        
        pg.surfarray.blit_array(surface, arr)
        del arr
        return surface
