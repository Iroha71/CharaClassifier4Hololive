
from flask import Flask, render_template, request
import os
import logics

app = Flask(__name__, static_folder='tmp')
TMP_FILE = './tmp'
DEVICE = 'cpu'
global model

"""
画像アップロード処理を行い、ライバー推定を行う
"""
@app.route("/", methods=["GET", "POST"])
def upload_img():
  if request.method != "POST":
    return render_template("index.html")

  filepath: str = logics.save_file(request.files['img'], TMP_FILE)
  predict_name: str = logics.predict_img(filepath, model, DEVICE)

  return render_template('result.html', name=predict_name, imgpath=filepath)

"""
トップページを表示する
"""
@app.route('/', methods=["GET"])
def index():
  return render_template("index.html")

"""
tmpファイル内の画像をすべて削除する
"""
@app.route('/delete_imgs', methods=['DELETE', 'POST'])
def delete_imgs():
  if request.method != "POST":
    return render_template("index.html")

  logics.delete_imgs_in_dir(TMP_FILE)

  return render_template("index.html", img_deleted='done')

if __name__ == "__main__":
  model = logics.build_model()
  DEVICE = logics.attach_device()
  app.run(host='0.0.0.0', port=5500, debug=True)
  