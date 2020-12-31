import torch
import torchvision
import torch.nn as nn
from torchvision import models, transforms
from typing import List
from PIL import Image
import numpy as np

"""
画像からライバー名を推定する
"""

device = 'cpu'
def build_model() -> models.resnet18:
  """ 推定モデルを構築する
  
  Returns:
    models.resnet18: 構築済み推定モデル
  """
  model = models.resnet18(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False

  infeature_count = model.fc.in_features
  model.fc = nn.Linear(infeature_count, 5)

  model.load_state_dict(torch.load('./model4cpu.pth'))
  type(model)
  return model

def predict_image(img: str, model: models.resnet18) -> torch.Tensor:
  """ 画像からライバー名を推定する

  Args:
    img (str): 推定対象画像のパス
    model (models.resnet18): 推定モデル

  Returns:
    torch.Tensor: 推定結果のラベル番号
  """
  model.eval()
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
  ])
  img = Image.open(img)
  img = transform(img)
  eval_img = img.unsqueeze(0)
  output = model(eval_img.to(device))
  result = output.argmax()

  return result

def main():
  model: models.resnet18 = build_model()
  print('分類対象の画像パスを入力')
  img_path = input()
  LABELS = ['大空スバル', '百鬼あやめ', '湊あくあ', '癒月ちょこ', '紫咲シオン']
  result: str = predict_image(img_path, LABELS, model)
  print(result.numpy())
  print(result.shape)

if __name__ == "__main__":
    main()