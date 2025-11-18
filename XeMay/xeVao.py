import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from firebase_hander import create_time_expired
import pytz
import tkinter as tk
import time

from XeMay.nhanDien import detect_license_plate
from face_detection.train_face import capture_face_and_upload
from tkinter import messagebox
from PIL import Image, ImageTk
from io import BytesIO
import requests
from XeMay.laybienso import get_all_license_plates
FIREBASE_REALTIME_URL = 'https://smallparking-41c54-default-rtdb.firebaseio.com/'
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()


def normalize_plate(plate):
    return plate.replace(".", "").upper() if plate else None

def firebase_put(path, data, include_timestamp=True):
    vn_time = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
    timestamp = vn_time.strftime('%Y-%m-%d %H:%M:%S')
    json_data = {"value": data, "timestamp": timestamp} if not isinstance(data, dict) and include_timestamp else data
    if isinstance(data, dict) and include_timestamp:
        json_data["timestamp"] = timestamp

    url = f"{FIREBASE_REALTIME_URL}/{path}.json"
    response = requests.put(url, json=json_data)
    print(f"[{timestamp}] Ghi {path}: {response.status_code}, {response.text}")



def open_result_window(bien_so_quet, plate_url, face_url,parent):
    # result = {}
    win = tk.Toplevel(parent)
    win.title("Kết quả quét")

    # Label hiển thị biển số
    label_bienso = tk.Label(win, text=f"Biển số quét được: {bien_so_quet}", font=("Arial", 14))
    label_bienso.pack(pady=10)

    # Khung chứa ảnh
    frame_images = tk.Frame(win)
    frame_images.pack(pady=10)

    # Hiện ảnh biển số
    if plate_url:
        try:
            if plate_url.startswith("http"):
                response = requests.get(plate_url)
                img_data = response.content
            else:
                with open(plate_url, "rb") as f:
                    img_data = f.read()

            img = Image.open(BytesIO(img_data)).resize((300, 200))
            tk_img_plate = ImageTk.PhotoImage(img)

            label_plate = tk.Label(frame_images, image=tk_img_plate)
            label_plate.image = tk_img_plate
            label_plate.pack(side="left", padx=10)
        except Exception as e:
            print("Lỗi load ảnh biển số:", e)

    # Hiện ảnh khuôn mặt
    if face_url:
        try:
            if face_url.startswith("http"):
                response = requests.get(face_url)
                img_data = response.content
            else:
                with open(face_url, "rb") as f:
                    img_data = f.read()

            img = Image.open(BytesIO(img_data)).resize((200, 200))
            tk_img_face = ImageTk.PhotoImage(img)

            label_face = tk.Label(frame_images, image=tk_img_face)
            label_face.image = tk_img_face
            label_face.pack(side="right", padx=10)
        except Exception as e:
            print("Lỗi load ảnh khuôn mặt:", e)



def run_license_scan( root):
    while True:
        # lấy từ nhận diện biển số xe
        bien_so, url_image_detected = detect_license_plate()
        bien_so_quet = normalize_plate(bien_so)
        print("Biển số quét được:", bien_so_quet)

        ds_bien_so_raw = get_all_license_plates()
        ds_bien_so = [normalize_plate(val) for val in ds_bien_so_raw if val]
        hop_le = bien_so_quet in ds_bien_so


        if hop_le:

            # Ghi Firestore
            today = datetime.today().strftime("%d%m%Y")
            xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xemay").document(bien_so_quet)
            db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

            # Tăng số lần vào
            doc = xe_doc_ref.get()
            solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
            xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

            # Chụp ảnh khuôn mặt
            image_url_vao = capture_face_and_upload()


            # Ghi timeline
            time_now = datetime.now().strftime("%H:%M:%S")
            timeline_data = {
                "timein": time_now,
                "biensoxevao": url_image_detected,
                "khuonmatvao": image_url_vao,
                "timeout": None,
                "biensoxera": None,
                "khuonmatra": None
            }
            xe_doc_ref.collection("timeline").document("timeline" + str(solanvao)).set(timeline_data)

            # Realtime DB
            firebase_put("trangthaicong", True, include_timestamp=False)
            if hop_le:
                firebase_put(f"biensotrongbai/{bien_so_quet}", {"trangthai": True, "canhbao": False})
            else:
                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                firebase_put(f"biensotrongbai/{bien_so_quet}", {
                    "trangthai": True,
                    "canhbao": False,
                    "timestamp": datetime_str,
                    "timeExpired": create_time_expired(datetime_str)
                })

            return url_image_detected, image_url_vao, time_now,bien_so_quet

        else:
            messagebox.showinfo("Thông báo", "Người dùng chưa đăng ký tài khoản")
            # window.destroy()


        time.sleep(1)




