import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Utility function to load images from URLs
def load_image_from_url(url, max_size=400):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Scale the image to the specified size while maintaining the aspect ratio
    max_dim = max(img.size)
    scale_factor = max_size / max_dim
    new_size = tuple([int(x * scale_factor) for x in img.size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Convert image to tensor
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img_tensor

# Resize the images to match the content image size
def resize_image_to_target(content_img, style_img):
    style_img = transforms.Resize(content_img.shape[2:])(style_img)
    return style_img

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load content and style images
content_url = "https://cdn.wccftech.com/wp-content/uploads/2017/01/Zelda-Breath-of-the-Wild-screenshots6-1480x802.jpg"  # Replace with your content image URL
style_url = "https://m.media-amazon.com/images/I/71pzSa3-euL._AC_SL1000_.jpg"      # Replace with your style image URL

content_img = load_image_from_url(content_url).to(device)
style_img = load_image_from_url(style_url).to(device)

# Resize style image to match content image size
style_img = resize_image_to_target(content_img, style_img)

# Normalize images
def image_loader(image):
    return image.clone().detach().requires_grad_(True).to(device)

content_img = image_loader(content_img)
style_img = image_loader(style_img)

# Define the model, loss functions, and optimizer
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Define normalization values (same as used in the VGG model)
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Normalization layer
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, img):
        return (img - self.mean[None, :, None, None]) / self.std[None, :, None, None]

# Define the model with content and style losses
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = target_feature.detach()
        self.loss = None

    def forward(self, x):
        gram_x = self.gram_matrix(x)
        gram_target = self.gram_matrix(self.target)
        self.loss = nn.functional.mse_loss(gram_x, gram_target)
        return x

    def gram_matrix(self, x):
        _, c, h, w = x.size()
        x = x.view(c, h * w)
        gram = torch.mm(x, x.t())
        return gram / (c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target = target_feature.detach()
        self.loss = None

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

# Get model with style and content losses
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers_default, style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"

        model.add_module(name, layer)

        if name in style_layers_default:
            target_feature = model(style_img)
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        if name in content_layers_default:
            target_feature = model(content_img)
            content_loss = ContentLoss(target_feature)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

    return model, style_losses, content_losses

# Set parameters for style transfer
content_layers_default = ['conv_4']  # Content layer
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # Style layers
num_steps = 500  # Number of optimization steps
style_weight = 1000000  # Weight for style loss
content_weight = 1  # Weight for content loss

# Get the model and losses
model, style_losses, content_losses = get_style_model_and_losses(
    cnn, normalization_mean, normalization_std, style_img, content_img, content_layers_default, style_layers_default)

# Optimizer for the input image
optimizer = optim.LBFGS([content_img])

# Perform the optimization process
style_score_list = []
content_score_list = []
run = [0]

while run[0] <= num_steps:
    def closure():
        content_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(content_img)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss

        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        style_score_list.append(style_score.item())
        content_score_list.append(content_score.item())

        loss = style_score + content_score
        loss.backward()

        run[0] += 1

        return style_score + content_score

    optimizer.step(closure)

# Final image
final_img = content_img.detach().cpu().squeeze(0).permute(1, 2, 0)
plt.figure(figsize=(10, 10))
plt.imshow(final_img)
plt.axis("off")
plt.title("Final Style-Transferred Image")
plt.show()
