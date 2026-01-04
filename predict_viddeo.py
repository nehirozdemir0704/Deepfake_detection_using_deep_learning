import os
import sys
import cv2
import torch
from torchvision import transforms, models
import numpy as np

# 1) Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Sınıf isimleri
class_names = ['fake', 'real']

# 3) Yüz Tanıma (OpenCV)
# Hata almamak için try-except içine aldık
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print("UYARI: Yüz tanıma modülü yüklenemedi!", e)
    face_cascade = None

# 4) Transformlar (Senin kodundaki ile aynı)
img_size = 224
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 5) Modeli yükle
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to(device)

    model_path = os.path.join("saved_models", "deepfake_frame_resnet18.pth")
    # Yol kontrolü
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası burada yok: {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model başarıyla yüklendi.")
    return model

# 6) Tek bir frame için tahmin
def predict_frame(model, frame):
    try:
        # Önce griye çevir (yüz bulma için)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüzleri ara
        faces = []
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        target_face = None

        if len(faces) > 0:
            # Yüz bulduysak en büyüğünü al
            # (x, y, w, h)
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Yüzü kes (Crop)
            target_face = frame[y:y+h, x:x+w]
            durum = "YUZ_BULUNDU"
        else:
            # Yüz bulamadıysak MERKEZE ODAKLAN (Center Crop)
            # Bu, modelin duvara bakıp "Real" demesini azaltır
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            # Ekranın ortasından 224x224'lük bir alan alalım
            crop_size = 224
            start_x = max(center_x - crop_size//2, 0)
            start_y = max(center_y - crop_size//2, 0)
            end_x = min(center_x + crop_size//2, w)
            end_y = min(center_y + crop_size//2, h)
            
            target_face = frame[start_y:end_y, start_x:end_x]
            durum = "YUZ_YOK_MERKEZ"

        # Eğer kesilen parça çok küçükse veya boşsa işlem yapma
        if target_face is None or target_face.size == 0:
            return None, None, "BOS_GORUNTU"

        # Modele hazırlık (BGR -> RGB)
        frame_rgb = cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB)
        img = data_transforms(frame_rgb)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

        pred_class = class_names[preds.item()]
        confidence = conf.item()
        
        return pred_class, confidence, durum

    except Exception as e:
        # Hata olursa program durmasın, None döndürsün
        # print("Bir karede hata oluştu:", e) 
        return None, None, "HATA"

# 7) Video Döngüsü
def predict_video(model, video_path, frame_step=10):
    if not os.path.exists(video_path):
        print(f"Video bulunamadı: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı.")
        return

    frame_idx = 0
    predictions = []
    
    print(f"\nİşleniyor: {video_path}")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            pred_class, conf, durum = predict_frame(model, frame)
            
            if pred_class is not None:
                predictions.append(pred_class)
                # Durumu da yazdıralım ki ne yaptığını görelim
                print(f"Frame {frame_idx:4d} | {durum:15s} -> {pred_class.upper()} (Guven: {conf:.2f})")

        frame_idx += 1

    cap.release()

    if not predictions:
        print("Hiçbir tahmin yapılamadı.")
        return

    fake_count = predictions.count('fake')
    real_count = predictions.count('real')
    total = len(predictions)

    print("-" * 50)
    print(f"TOPLAM KARE: {total}")
    print(f"FAKE Sayısı: {fake_count}")
    print(f"REAL Sayısı: {real_count}")

    # Eşik değeri %50 (Yarısından fazlası fake ise fake de)
    if fake_count > real_count:
        final_decision = "FAKE"
    else:
        final_decision = "REAL"

    print(f"\n>>> SONUÇ: {final_decision}")

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python predict_video.py <video_yolu>")
        sys.exit(1)

    video_path = sys.argv[1]
    
    # Modeli yükle
    model = load_model()
    
    # Tahmin yap
    predict_video(model, video_path, frame_step=10)

if __name__ == "__main__":
    main()
