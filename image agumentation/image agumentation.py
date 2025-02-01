from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

image_path =r"/content/background.jpg"
image= Image.open(image_path)

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomRotation (degrees=40),
    transforms.RandomResizedCrop(size=(128,128), scale=(0.5,1.2)),
    transforms.ColorJitter(brightness=0.4,contrast=0.5, saturation=0.6),
    ])

augmented_images = [augmentations(image) for _ in range(7)]

plt.figure(figsize=(13,7))
plt.subplot(1, 8, 1)
plt.title("original")
plt.imshow(image)
plt.axis("off")

for i,aug_img in enumerate(augmented_images,start=2):
   plt.subplot(1,8,i)
   plt.title(f"augmented (i-1)")
   plt.imshow(aug_img)
   plt.axis("off")

plt.tight_layout()
plt.show()