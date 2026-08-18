import torch
from PIL import Image
from torchvision import transforms

def predict_image(image_path, model):
    target_image = Image.open(image_path)

    trans = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # 사이즈 통일
            transforms.ToTensor(),  # 텐서 변환
            transforms.Normalize((0.5), (0.5)),  # 정규화
        ]
    )

    image_tensor = trans(target_image)
    image_tensor = image_tensor[:3, :, :]  # 차원 자름(3차원 RGB)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor.size()

    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        _, predict = torch.max(output, dim=1)

    print(predict)
