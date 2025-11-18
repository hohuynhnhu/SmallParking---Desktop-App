import tkinter as tk
import threading
import time
from datetime import datetime
from xeoto.nhanDien import detect_license_plate
from firebase_service import FirebaseService
from firebase_admin import firestore
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import cv2
from ultralytics import YOLO
from xeoto.ketQuaXeOtoRa import process_car_image
import torch.nn as nn
import xeoto.utils as utils
import requests
import numpy as np
import traceback


# =====================
# Hàm xử lý quét biển số
# =====================
def run_license_scan(label_status, root, label_bsx):
    firebase_service = FirebaseService()
    db = firestore.client()

    # khởi tạo model siamese
    import torch

    # === Transform y như khi train ===
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # === Load YOLO để crop xe ===
    yolo_model = YOLO("yolov8n.pt")

    def load_image(img_path):
        """Đọc ảnh từ local path, URL (http/https) hoặc numpy array."""
        # Nếu truyền vào là numpy array (ảnh OpenCV)
        if isinstance(img_path, np.ndarray):
            img_cv = img_path
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            return img_cv, img_pil

        # Nếu là string (đường dẫn hoặc URL)
        if isinstance(img_path, str):
            if img_path.startswith(("http://", "https://")):  # Link URL
                resp = requests.get(img_path, stream=True).content
                img_array = np.asarray(bytearray(resp), dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_pil = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
                return img_cv, img_pil
            else:  # File local
                img_cv = cv2.imread(img_path)
                img_pil = Image.open(img_path).convert("RGB")
                return img_cv, img_pil

        raise TypeError(f"Không hỗ trợ kiểu dữ liệu {type(img_path)} trong load_image")

    def crop_car_largest(img_path):
        """Crop xe lớn nhất từ ảnh (nếu không detect được thì trả ảnh gốc)."""
        results = yolo_model(img_path, verbose=False)
        max_area = 0
        best_crop = None

        # Đọc ảnh trước để tránh đọc nhiều lần
        img_cv, img_pil = load_image(img_path)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if yolo_model.names[cls] == "car":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        crop = img_cv[y1:y2, x1:x2]
                        if crop is not None and crop.size > 0:
                            best_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            max_area = area

        if best_crop is not None:
            return best_crop
        else:
            return img_pil

    # === Mạng Siamese giống lúc train ===
    class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

        def forward_once(self, x):
            return self.backbone(x)

        def forward(self, x1, x2):
            return self.forward_once(x1), self.forward_once(x2)

    while True:
        try:
            # 1. Quét biển số đuôi xe
            bien_so, url_image_detected, img_path, best_plate = detect_license_plate()
            if not bien_so:
                label_status.config(text="Không quét được biển số ", bg="red")
                utils.update_label_content(label_bsx, "Không quét được biển số", bg="red")
                utils.update_label_content(label_status, "Không quét được biển số", bg="red")
                time.sleep(2)
                continue

            bien_so_quet = bien_so.replace(".", "").upper()
            utils.update_label_content(label_bsx, bien_so_quet, bg="green")
            utils.update_label_content(label_status, "Biển số quét được: " + bien_so_quet)
            print("Biển số quét được:", bien_so_quet)

            # 2. Kiểm tra hợp lệ với Firebase (biển số từ detect_license_plate)
            ds_bien_so = firebase_service.get_all_license_plates()
            if bien_so_quet not in ds_bien_so:
                utils.update_label_content(label_status, f"Biển số {bien_so_quet} không có trong bãi xe", bg="red")
                time.sleep(1)
                continue

            # 3. Lấy dữ liệu biển số từ Firebase
            bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
            if not bien_so_data:
                utils.update_label_content(label_status, f"Không lấy được dữ liệu {bien_so_quet} ", bg="red")
                time.sleep(1)
                continue

            # Sau khi qua bước trên mới tới process_car_image
            link_goc, link_crops, mau, image_dau_xe, bsx_dau, image_logo = process_car_image()

            # 4. Lấy timeline gần nhất
            from datetime import datetime

            today = datetime.today().strftime("%d%m%Y")
            xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xeoto").document(bien_so_quet)

            # Lấy tất cả document trong timeline
            timeline_docs = xe_doc_ref.collection("timeline").list_documents()
            max_index = -1
            for tdoc in timeline_docs:
                name = tdoc.id
                if name.startswith("timeline"):
                    try:
                        index = int(name.replace("timeline", ""))
                        if index > max_index:
                            max_index = index
                    except ValueError:
                        continue

            timeline_data = None
            if max_index >= 0:
                timeline_doc_id = f"timeline{max_index}"
                timeline_ref = xe_doc_ref.collection("timeline").document(timeline_doc_id)

                # 🔹 Lấy dữ liệu từ timeline gần nhất
                doc_snapshot = timeline_ref.get()
                if doc_snapshot.exists:
                    timeline_data = doc_snapshot.to_dict()
                    hinhdauxevao = timeline_data.get("hinhxevao")
                    logovao = timeline_data.get("logovao")
                    logovao = logovao[0] if logovao else None
                    hinhduoixevao = timeline_data.get("biensoxevao")

                    print("Hình đầu xe vào:", hinhdauxevao)
                    print("Hình đuôi xe vào:", hinhduoixevao)
                    print("Lô gô vào:", logovao)
            else:
                timeline_doc_id = None
                timeline_ref = None
                hinhdauxevao, logovao, hinhduoixevao = None, None, None

            bsx_dau_vao = None
            bsx_duoi_vao = None
            _, _, _, bsx_dau_vao = detect_license_plate(hinhdauxevao)
            _, _, _, bsx_duoi_vao = detect_license_plate(hinhduoixevao)

            # === Hàm so sánh ảnh ===
            def compare_images(img_path1, img_path2, model_path="siamese_model.pth", threshold=0.5):
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # Load model
                    model = SiameseNetwork().to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()

                    # Crop + transform ảnh 1
                    img1 = crop_car_largest(img_path1)
                    img1 = transform(img1).unsqueeze(0).to(device)

                    # Crop + transform ảnh 2
                    img2 = crop_car_largest(img_path2)
                    img2 = transform(img2).unsqueeze(0).to(device)

                    # Forward
                    with torch.no_grad():
                        out1, out2 = model(img1, img2)
                        distance = torch.nn.functional.pairwise_distance(out1, out2).item()

                    print(f"Khoảng cách giữa 2 ảnh: {distance:.4f}")
                    if distance < threshold:
                        print("Cùng xe")
                        return True, distance
                    else:
                        print("Khác xe")
                        return False, distance
                except Exception as e:
                    print(f" Lỗi so sánh ảnh: {e}")
                    return False, 1.0  # Trả về False nếu có lỗi

            # ================================
            # So sánh link_goc và hinhxevao
            # ================================
            same_car = False
            same_logo = False

            if link_goc and hinhdauxevao:
                same_car, distance = compare_images(link_goc, hinhdauxevao, model_path="siamese_model.pth",
                                                    threshold=0.5)
                if same_car:
                    print("Ảnh cùng xe")
                else:
                    print(" Ảnh khác xe")
            else:
                same_car = False
                print("Không có đủ ảnh để so sánh xe")

            if logovao is not None and image_logo is not None:
                same_logo, distance_logo = compare_images(logovao, image_logo, model_path="siamese_model.pth",
                                                          threshold=0.5)
                if same_logo:
                    print(" Logo cùng xe")
                else:
                    print(" Logo khác xe")
            else:
                same_logo = False
                print("Không có đủ ảnh logo để so sánh")

            # 5. Cập nhật trạng thái xe
            utils.update_label_content(label_status, f"Biển số {bien_so_quet} hợp lệ ", bg="green")

            trangthai = bien_so_data.get('trangthai')

            # Debug: In ra các điều kiện
            # print(f"🔍 DEBUG ĐIỀU KIỆN:")
            # print(f"  - same_car: {same_car}")
            # print(f"  - same_logo: {same_logo}")
            # print(f"  - trangthai: {trangthai} (type: {type(trangthai)})")
            # print(f"  - Điều kiện tổng: {same_car and same_logo and (trangthai is False)}")

            # Chuyển đổi trangthai sang boolean nếu cần
            if isinstance(trangthai, str):
                trangthai = trangthai.lower() == 'true'
            elif trangthai is None:
                trangthai = False

            if same_car and same_logo and (trangthai is False):
                print("Tất cả điều kiện hợp lệ - Cho xe ra")
                firebase_service.update_license_plate_field(bien_so_quet, True)
                firebase_service.delete_license_plate(bien_so_quet)
                utils.update_label_content(label_status, f"Xe biển số {bien_so_quet} hợp lệ. Được phép ra", bg="green")
                time.sleep(1)
                # Lấy document xe
                doc = xe_doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    solanra = data.get("solanra", 0)
                else:
                    solanra = 0

                solanra += 1
                xe_doc_ref.set({"solanra": solanra}, merge=True)

                # Thời gian hiện tại
                time_now = datetime.now().strftime("%H:%M:%S")

                # Ghi vào timeline gần nhất
                if timeline_ref:
                    timeline_ref.set({
                        "timeout": time_now,
                        "biensoxera": url_image_detected,
                        "hinhxera": link_goc,
                        "logora": link_crops,
                    }, merge=True)
                    print(f"Đã cập nhật timeline {timeline_doc_id}")
                else:
                    print("Không tìm thấy timeline để cập nhật.")
                time.sleep(2)  # delay để người dùng thấy thông báo
            else:
                print(" Có điều kiện không hợp lệ - Không cho xe ra")
                utils.update_label_content(label_status, f"Xe biển số {bien_so_quet} không hợp lệ!!!", bg="red")
                time.sleep(1)
            break

        except Exception as e:
            print(f"Lỗi tổng thể: {e}")
            print(traceback.format_exc())
            utils.update_label_content(label_status, f"Lỗi hệ thống: {e}", bg="red")
            break

    data_xe_vao = utils.DataXeVao(
        hinh_dau_xe=hinhdauxevao,
        hinh_duoi_xe=hinhduoixevao,
        bsx_dau=bsx_dau_vao,
        bsx_duoi=bsx_duoi_vao,
        logo=logovao
    )
    sameImage = utils.SameImage(
        same_car=same_car,
        same_logo=same_logo
    )
    return bien_so_quet, img_path, best_plate, image_dau_xe, bsx_dau, image_logo, data_xe_vao, sameImage