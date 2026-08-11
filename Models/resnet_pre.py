
from torchvision.models import resnet34, ResNet34_Weights

def get_resnet_model():
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    
    #모델의 모든 파라미터를 '훈련 불가'로 동결
    for params in model.parameters():
        params.requires_grad = False
        
    #모델의 마지막 FC 파라미터만 '훈련 가능'으로 동결 해제
    for params in model.fc.parameters():
        params.requires_grad = True
    
    return model