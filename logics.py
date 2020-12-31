from flask import request
import os
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy
import shutil

from werkzeug.datastructures import FileStorage

"""
推定ロジック
"""
def save_file(file: FileStorage, path: str, name: str=None) -> str:
  """ ファイルをストレージに保存する

  Args:
    file (FileStorage): 保存するファイル
    path (str): 保存先のパス
    name (str): 保存時にファイルにつける名前

  Returns:
    str: 保存先のフルパス
  """
  print(type(file))
  if not os.path.exists(path):
    os.mkdir(path)
    
  name = f"{ datetime.now().strftime('%Y%m%d%H%M%S') }.png" if name == None else name
  savepath = os.path.join(path, name)
  file.save(savepath)

  return savepath

LABELS = ['大空スバル', '百鬼あやめ', '湊あくあ', '癒月ちょこ', '紫咲シオン']

def attach_device() -> str:
  """ pytorchの使用リソースの確認を行う

  Returns:
    str: 使用リソースの種類 (cpu or cuda)
  """
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model() -> models.resnet18:
  """ 推論モデルを構築する

  Returns:
    models.resnet18: 構築済み推論モデル
  """
  model = models.resnet18(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False

  infeature_count = model.fc.in_features
  model.fc = nn.Linear(infeature_count, len(LABELS))

  model.load_state_dict(torch.load('./model4cpu.pth'))
  print(type(model))
  return model

def predict_img(imgpath: str, model: models.resnet18, device: str) -> str:
  """ 画像からライバー名を推定する

  Args:
    imgpath (str): 推定対象画像のパス
    model (models.resnet18): 推論モデル
    device (str): 推論に使用するリソース

  Returns:
    str: ライバー名
  """
  model.eval()
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
  ])
  img = Image.open(imgpath)
  img = transform(img)
  expand_img = img.unsqueeze(0)
  results = model(expand_img.to(device))
  predicted_index = results.argmax().numpy()

  return LABELS[predicted_index]

def delete_imgs_in_dir(path: str):
  """ 指定ディレクトリのファイルを全削除する
  
  Args:
    path (str): ファイル全削除を行うパス
  """
  shutil.rmtree(path)
