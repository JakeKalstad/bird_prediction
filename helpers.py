import io
import torch

import torchvision.transforms as transforms
from PIL import Image

device = "cpu"


def get_prediction(app, image_bytes):
    scaled_img = transform_image(image_bytes=image_bytes)
    torch_images = scaled_img.unsqueeze(0)
    model = app.model.to(device)

    top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(
        torch_images)
    _, predict = torch.max(concat_logits, 1)
    pred_id = predict.item()
    predicted = model.bird_classes[pred_id]
    return pred_id, predicted


def transform_image(image_bytes):
    preprocess = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.CenterCrop((448, 448)),
        # transforms.RandomHorizontalFlip(),  # only if train
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image)
