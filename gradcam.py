# gradcam.py
# Grad-CAM Visual Explainability Module
# For: Deepfake Detection Project — Ephraim, FUTA
# Purpose: Produces a heatmap showing which parts of an image
#          most strongly influenced the model's classification.

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    A class that implements Grad-CAM for a trained PyTorch model.
    Grad-CAM = Gradient-weighted Class Activation Mapping.
    It produces a heatmap showing which regions of an image
    most influenced the model's prediction for a given class.
    """

    def __init__(self, model, target_layer=None):
        """
        Constructor — runs once when the GradCAM object is created.
        It registers hooks on the target layer so that every time
        data flows through that layer, the output and its gradients
        are saved automatically.

        Args:
            model:        A trained PyTorch model (EfficientNet-B0)
            target_layer: The layer to inspect. If None, we use the
                          last layer of model.features, which is the
                          final convolutional block of EfficientNet-B0.
        """
        self.model = model
        self.gradients = None    # Will hold gradients from backward pass
        self.activations = None  # Will hold activations from forward pass

        # If no target layer is specified, use the last conv block
        if target_layer is None:
            target_layer = model.features[-1]

        # Forward hook: saves the OUTPUT of the target layer
        # whenever the model does a forward pass.
        target_layer.register_forward_hook(self._save_activation)

        # Backward hook: saves the GRADIENT flowing back through
        # the target layer during backpropagation.
        # We use register_full_backward_hook (the modern method)
        # because register_backward_hook is deprecated in PyTorch 2.x.
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """
        This function is called automatically by the forward hook.
        It saves the output (activations) of the target layer.

        Args:
            module:  The layer the hook is attached to
            input:   What went INTO the layer (we don't need this)
            output:  What came OUT of the layer (this is what we save)
        """
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        """
        This function is called automatically by the backward hook.
        It saves the gradient of the loss with respect to the
        output of the target layer.

        Args:
            module:      The layer the hook is attached to
            grad_input:  Gradient going INTO the layer (not needed)
            grad_output: Gradient coming OUT of the layer (we save this)
        """
        # grad_output is a tuple; we take the first element
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_idx):
        """
        Runs the full Grad-CAM algorithm and returns a heatmap.

        Args:
            image_tensor: A preprocessed image tensor of shape (1, 3, 224, 224)
            class_idx:    The class to explain. 0 = REAL, 1 = FAKE.
                          Usually the model's predicted class.

        Returns:
            heatmap: A numpy array of shape (224, 224, 3) in BGR format
                     (because OpenCV uses BGR, not RGB).
        """
        # Set model to evaluation mode (turns off dropout, batchnorm uses
        # running statistics). Note: we do NOT use torch.no_grad() here
        # because we NEED gradients to flow for Grad-CAM to work.
        self.model.eval()

        # ── STEP 1: FORWARD PASS ────────────────────────────────
        # Run the image through the model. The forward hook will
        # automatically save the activations of the target layer.
        output = self.model(image_tensor)

        # ── STEP 2: BACKWARD PASS ───────────────────────────────
        # Clear any existing gradients in the model
        self.model.zero_grad()

        # Backpropagate ONLY the score of the target class.
        # output[0, class_idx] picks the score for the chosen class
        # for the first (and only) image in the batch.
        # The backward hook will automatically save the gradients.
        output[0, class_idx].backward()

        # ── STEP 3: COMPUTE WEIGHTS ─────────────────────────────
        # Global Average Pooling of the gradients.
        # We average each feature map's gradient over its spatial
        # dimensions (height and width), producing one weight per
        # feature map. The result has shape (1, C, 1, 1) where C is
        # the number of channels.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # ── STEP 4: WEIGHTED COMBINATION OF ACTIVATIONS ─────────
        # Multiply each feature map (activation) by its corresponding
        # weight, then sum across all channels.
        # Result shape: (1, 1, H, W) where H, W are the spatial
        # dimensions of the last conv layer's output.
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Free gradient memory immediately to keep RAM usage low
        self.model.zero_grad(set_to_none=True)

        # Apply ReLU — keeps only POSITIVE contributions.
        # Why? Negative values mean a region argued AGAINST the
        # prediction. We only want to show regions that argued FOR it.
        cam = F.relu(cam)

        # ── STEP 5: RESIZE TO 224×224 ───────────────────────────
        # The last conv layer output is smaller than the input image
        # (typically 7×7 for a 224×224 input). We upsample using
        # bilinear interpolation so the heatmap matches the image size.
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        # Remove the batch and channel dimensions, move to CPU,
        # and convert to a NumPy array for OpenCV processing.
        cam = cam.squeeze().cpu().detach().numpy()

        # ── STEP 6: NORMALISE TO 0–255 ──────────────────────────
        # Shift so the minimum value is 0
        cam -= cam.min()

        # Scale so the maximum value is 1 (avoiding division by zero)
        if cam.max() > 0:
            cam /= cam.max()

        # Convert to 0–255 integer range for OpenCV
        cam = (cam * 255).astype(np.uint8)

        # ── STEP 7: APPLY COLOUR MAP ────────────────────────────
        # The JET colour map converts grayscale values to a colour
        # scale: blue (low) → green → yellow → red (high).
        # OpenCV returns the image in BGR format (not RGB).
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        return heatmap


def overlay_heatmap(original_pil_image, heatmap_bgr, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on top of the original image
    so the user can see both at the same time.

    Args:
        original_pil_image: A PIL Image (the original uploaded image)
        heatmap_bgr:        A numpy array (224, 224, 3) in BGR format
                            (the output of GradCAM.generate())
        alpha:              How strong the heatmap overlay is.
                            0.0 = only original, 1.0 = only heatmap.
                            0.4 = 40% heatmap + 60% original (good default)

    Returns:
        A PIL Image (RGB) of the overlay, 224×224 pixels.
    """
    # Resize the original image to 224×224 to match the heatmap
    orig = original_pil_image.resize((224, 224))

    # Convert the PIL image (RGB) to an OpenCV array (BGR).
    # We do this because OpenCV uses BGR ordering by default,
    # and our heatmap is already in BGR.
    orig_bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)

    # Blend the two images together using weighted addition:
    #   result = (1 - alpha) × original + alpha × heatmap
    # This produces a single image showing both the photo and the
    # coloured heatmap regions on top of it.
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_bgr, alpha, 0)

    # Convert back to RGB so PIL can display it correctly in the browser
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Return as a PIL Image (this is what Flask will save and serve)
    return Image.fromarray(overlay_rgb)