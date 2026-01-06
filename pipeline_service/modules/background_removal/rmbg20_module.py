import torch
from PIL import Image
from torchvision import transforms

from transformers import AutoModelForImageSegmentation

from modules.background_removal.rmbg_manager import BackgroundRemovalService

class RMBG2BackgroundRemovalService(BackgroundRemovalService):
    def _initialize_model_and_transforms(self) -> tuple[AutoModelForImageSegmentation, transforms.Compose]:
        """
        Initialize RMBG2.0 model and transforms.
        """
        model: AutoModelForImageSegmentation | None = None

        transform = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
            ]
        )

        return model, transform
    
    def _load_model(self) -> AutoModelForImageSegmentation:
        """
        Load the RMBG2.0 background removal model.
        """
        model = AutoModelForImageSegmentation.from_pretrained(
            self.settings.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
                )
        return model.to(self.device)
    
    def _remove_background(self, image: Image) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the background from the image.
        """
        rgb_image = image.convert('RGB')
        rgb_tensor = self.transforms(rgb_image).to(self.device)
        input_tensor = self.normalize(rgb_tensor).unsqueeze(0)
                
        with torch.no_grad():
            # Get mask from model (1, 1, H, W)
            preds = self.model(input_tensor)[-1].sigmoid()
            # Reshape and quantize mask values: (1, 1, H, W) -> (1, H, W) -> (H, W)
            mask = preds[0].squeeze().mul_(255).int().div(255).float()
        return rgb_tensor,mask