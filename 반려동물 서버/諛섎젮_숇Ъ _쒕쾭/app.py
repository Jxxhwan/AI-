from flask import Flask, render_template
from flask import request, redirect, url_for, session
# from flaskext.mysql import MySQL
from werkzeug.utils import secure_filename
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home1():
   return render_template('index.html')

@app.route('/index.html')
def home2():
   return render_template('index.html')

@app.route('/demo.html', methods=['GET', 'POST'])
def demo():
   return render_template('demo.html')

@app.route('/report.html', methods=['GET', 'POST'])
def report():
   return render_template('report.html')

def select_best_circle(circles, img_area, image_size):
   if circles is None:
      return None
   best_circle = None
   min_area = 0.4 * img_area  # 원하는 최소 면적: 이미지 면적의 40%
   max_area = 0.75 * img_area  # 최대 면적: 이미지 면적의 75%

   # 중심점 조건 계산
   min_x, max_x = int(image_size / 4), int(3 * image_size / 4)
   min_y, max_y = int(image_size / 4), int(3 * image_size / 4)
   
   for circle in circles[0, :]:
      x, y, radius = circle
      circle_area = np.pi * (radius ** 2)
      # 면적 조건과 중심점 위치 조건 확인
      if (circle_area >= min_area and circle_area <= max_area and 
         min_x <= x <= max_x and min_y <= y <= max_y):
         if best_circle is None or circle_area > np.pi * (best_circle[2] ** 2):
               best_circle = circle
   return best_circle

def process_image(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
   img_area = image.shape[0] * image.shape[1]  # 이미지의 전체 면적 계산
   image_size = image.shape[0]  # 정사각형 이미지의 한 변의 길이

   detected_circles = cv2.HoughCircles(
      gray_blurred,
      cv2.HOUGH_GRADIENT,
      1,
      20,
      param1=50,
      param2=30,
      minRadius=0,  # 최소 반지름 제한 없음
      maxRadius=int(np.sqrt((0.75 * img_area) / np.pi))  # 가능한 최대 반지름 계산
   )

   best_circle = select_best_circle(detected_circles, img_area, image_size)
   if best_circle is not None:
      x, y, r = best_circle
      # 중심 좌표와 반지름을 정수형으로 변환
      x = int(x)
      y = int(y)
      r = int(r)
      
      # 원 마스크 생성
      mask = np.zeros_like(image, dtype=np.uint8)
      cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
      
      # 원 외부를 검정색으로 채우기
      masked_image = cv2.bitwise_and(image, mask)
      
   else:
      masked_image = image
      
   return masked_image



def cam(model, image, label):
   image = tf.convert_to_tensor(image)
   cam_model = tf.keras.models.Model(inputs=[model.input], outputs=[model.layers[-3].output, model.output])
   
   # 이미지에 배치 차원을 추가합니다.
   image_with_batch_dim = tf.expand_dims(image, axis=0)
   
   # CAM 모델로 예측을 수행합니다.
   conv_outputs, predictions = cam_model(image_with_batch_dim)
   
   # 특성 맵의 형태를 조정합니다.
   conv_outputs = conv_outputs[0]  # (H, W, C) 형태로 변환
   class_weights = model.layers[-1].get_weights()[0]
   cam_image = np.zeros(shape=conv_outputs.shape[:2], dtype=np.float32)
   
   for i, w in enumerate(class_weights[:, label]): 
      cam_image += w * conv_outputs[:, :, i]
   
   cam_image = np.maximum(cam_image, 0)  # ReLU 적용 (음수 값 제거)
   cam_image /= np.max(cam_image)  # [0, 1] 범위로 정규화   
   cam_image = (cam_image * 255).astype(np.uint8)  # [0, 255] 범위로 변환

   return cam_image

@app.route("/model", methods=['POST'])
def model():
   if request.method == 'POST':
      file = request.files['file']
      if not file:
         return "No file uploaded.", 400
      
      filename = file.filename
      file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      
      # Ensure the upload folder exists
      if not os.path.exists(app.config['UPLOAD_FOLDER']):
         os.makedirs(app.config['UPLOAD_FOLDER'])
      
      file.save(file_path)
      
      if not os.path.isfile(file_path):
         return "File not saved correctly.", 500
      
      image_src = url_for('static', filename='uploads/' + filename)
      image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Adjusted to correct path
      
      # 이미지 읽기 및 전처리
      try:
         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
         if image is None:
               return "Error loading image.", 400
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식이므로 RGB로 변환
      except Exception as e:
         return str(e), 500
         
      # crop
      crop_img = process_image(image)
      original_image = cv2.resize(crop_img, (112, 112)) # 이미지 크기 조정
      crop_img = original_image.astype(np.float32) / 255.0  # 이미지를 [0, 1] 범위로 정규화
      crop_img = np.expand_dims(crop_img, axis=0)  # 배치 차원을 추가하여 shape을 (1, 112, 112, 3)으로 만듦
      
      # 모델 로드
      model = tf.keras.models.load_model('./src/preprocessing_not_pretrained_best_model_again_disease_5.h5')
      
      # 진단명 분류 예측
      prediction = model.predict(crop_img)
      predicted_class = np.argmax(prediction)
      
      # cam
      cam_image = cam(model, crop_img[0], label=predicted_class)
      
      # cam 후처리
      if np.max(original_image) > 1:
         original_image = original_image / 255.0
         
      cam_image_resized = cv2.resize(cam_image, (original_image.shape[1], original_image.shape[0]))
      cam_image_normalized = (cam_image_resized - np.min(cam_image_resized)) / (np.max(cam_image_resized) - np.min(cam_image_resized))
      
      threshold = 0.7  # 임계값 설정 (0과 1 사이)
      mask = cam_image_normalized >= threshold
      
      # Ensure that only non-black pixels (or pixels close to black) in the original image are considered
      color_threshold = 10 / 255.0 # Adjust this threshold as needed
      non_black_mask = np.any(original_image > color_threshold, axis=-1)
      mask = mask & non_black_mask
      
      highlighted_heatmap = np.zeros_like(original_image)
      highlighted_heatmap[mask] = [1, 0, 0]  # 빨간색 (RGB)

      # 히트맵을 원본 이미지에 더하고 정규화
      superimposed_img = original_image.copy()
      superimposed_img[mask] = 0.5 * original_image[mask] + 0.5 * highlighted_heatmap[mask]
      
      superimposed_img_pil = Image.fromarray((superimposed_img * 255).astype(np.uint8))
      superimposed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], "superimposed_img.png")
      superimposed_img_pil.save(superimposed_img_path)
      
   # return jsonify(superimposed_img), 200
   return render_template('report.html', filename=image_src, label=predicted_class, cam_img=superimposed_img_path, ratio=prediction)


@app.route('/history.html', methods=['GET', 'POST'])
def history():
   return render_template('history.html')

@app.route('/history_1.html', methods=['GET', 'POST'])
def history_1():
   return render_template('history_1.html')

@app.route('/compare.html', methods=['GET', 'POST'])
def compare():
   return render_template('compare.html')

@app.route('/tables.html', methods=['GET', 'POST'])
def tables():
   return render_template('tables.html')

@app.route('/charts.html', methods=['GET', 'POST'])
def charts():
   return render_template('charts.html')

@app.route('/charts_1.html', methods=['GET', 'POST'])
def charts_1():
   return render_template('charts_1.html')

@app.route('/charts_2-1.html', methods=['GET', 'POST'])
def charts_2_1():
   return render_template('charts_2-1.html')

@app.route('/charts_2-2.html', methods=['GET', 'POST'])
def charts_2_2():
   return render_template('charts_2-2.html')

@app.route('/charts_2-3.html', methods=['GET', 'POST'])
def charts_2_3():
   return render_template('charts_2-3.html')

@app.route('/schedule.html', methods=['GET', 'POST'])
def schedule():
   return render_template('schedule.html')

if __name__ == '__main__':  
   app.run('0.0.0.0',port=5050,debug=True) 
