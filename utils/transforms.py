import torchvision.transforms as T

def get_transforms():
    return T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
