import firebase_admin  # Thêm import này
from firebase_admin import credentials, firestore
from datetime import datetime
import sys, os
import traceback  # Thêm để debug

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from firebase_hander import create_time_expired
import requests
import pytz
import threading
import tkinter as tk
from tkinter import messagebox  # Thêm import
import time
import json
import uuid
from google.cloud.firestore import Client
from xeoto.nhanDien import detect_license_plate
from firebase_hander import get_field_from_all_docs
from xeoto.ketQuaXeOtoVao import process_car_image
import xeoto.utils as utils
from XeMay.laybienso import get_all_license_plates

FIREBASE_REALTIME_URL = 'https://smallparking-41c54-default-rtdb.firebaseio.com/'

# Khởi tạo Firebase
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print(" Firebase initialized successfully")
except Exception as e:
    print(f"Lỗi khởi tạo Firebase: {e}")
    db = None


def normalize_plate(plate):
    return plate.replace(".", "").upper() if plate else None


def firebase_put(path, data, include_timestamp=True):
    try:
        vn_time = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
        timestamp = vn_time.strftime('%Y-%m-%d %H:%M:%S')
        json_data = {"value": data, "timestamp": timestamp} if not isinstance(data,
                                                                              dict) and include_timestamp else data
        if isinstance(data, dict) and include_timestamp:
            json_data["timestamp"] = timestamp

        url = f"{FIREBASE_REALTIME_URL}/{path}.json"
        response = requests.put(url, json=json_data)
        print(f"[{timestamp}] Ghi {path}: {response.status_code}, {response.text}")
        return True
    except Exception as e:
        print(f"Lỗi ghi Firebase Realtime: {e}")
        return False


def run_license_scan(label_status, label_bsx):
    if db is None:
        utils.update_label_content(label_status, "Lỗi kết nối Firebase", bg="red")
        return None, None, None, None, None, None

    while True:
        try:
            # 1. Quét biển số
            bien_so, url_image_detected, img_path, best_plate = detect_license_plate()
            if not bien_so:
                utils.update_label_content(label_bsx, "Không quét được biển số", bg="red")
                utils.update_label_content(label_status, "Không quét được biển số", bg="red")
                time.sleep(2)
                continue

            bien_so_quet = normalize_plate(bien_so)
            utils.update_label_content(label_bsx, bien_so_quet, bg="green")
            utils.update_label_content(label_status, "Biển số quét được: " + bien_so_quet)
            print("Biển số quét được:", bien_so_quet)

            # 2. Kiểm tra hợp lệ
            ds_bien_so_raw = get_all_license_plates()
            ds_bien_so = [normalize_plate(val) for val in ds_bien_so_raw if val]
            hop_le = bien_so_quet in ds_bien_so

            print(f"Danh sách biển số hợp lệ: {ds_bien_so}")
            print(f"Biển số {bien_so_quet} hợp lệ: {hop_le}")

            if hop_le:
                utils.update_label_content(label_status, f"Biển số {bien_so_quet} hợp lệ", bg="green")

                # Ghi Firestore với try-catch
                try:
                    today = datetime.today().strftime("%d%m%Y")
                    print(f" Ngày: {today}")
                    print(f" Biển số: {bien_so_quet}")

                    # Tạo document reference
                    xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xeoto").document(
                        bien_so_quet)

                    # Ghi document ngày
                    db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)
                    print(" Đã ghi document ngày")

                    # Lấy và tăng số lần vào
                    doc = xe_doc_ref.get()
                    if doc.exists:
                        solanvao = doc.to_dict().get("solanvao", 0)
                        print(f" Số lần vào hiện tại: {solanvao}")
                    else:
                        solanvao = 0
                        print(" Xe mới, số lần vào: 0")

                    solanvao += 1
                    xe_doc_ref.set({"solanvao": solanvao}, merge=True)
                    print(f" Đã cập nhật số lần vào: {solanvao}")

                    # Lấy dữ liệu từ process_car_image
                    link_goc, link_crops, mau, image_dau_xe, bsx_dau, image_logo = process_car_image()
                    print(" Đã xử lý ảnh xe")

                    # Ghi timeline
                    time_now = datetime.now().strftime("%H:%M:%S")
                    timeline_data = {
                        "timein": time_now,
                        "biensoxevao": url_image_detected,
                        "hinhxevao": link_goc,
                        "logovao": link_crops,
                        "logora": None,
                        "timeout": None,
                        "biensoxera": None,
                        "hinhxera": None
                    }

                    timeline_doc_ref = xe_doc_ref.collection("timeline").document("timeline" + str(solanvao))
                    timeline_doc_ref.set(timeline_data)
                    print(f"Đã ghi timeline: timeline{solanvao}")

                    # Realtime DB
                    firebase_put("trangthaicong", True, include_timestamp=False)
                    firebase_put(f"biensotrongbai/{bien_so_quet}", {
                        "trangthai": True,
                        "canhbao": False
                    })
                    print("Đã ghi Realtime Database")

                    time.sleep(1)
                    break

                except Exception as e:
                    print(f" Lỗi ghi Firestore: {e}")
                    print(traceback.format_exc())
                    utils.update_label_content(label_status, f"Lỗi ghi dữ liệu: {e}", bg="red")
                    break

            else:
                utils.update_label_content(label_status, f"Biển số {bien_so_quet} không hợp lệ", bg="red")
                firebase_put("trangthaicong", False, include_timestamp=False)
                image_dau_xe = None
                bsx_dau = None
                image_logo = None
                print("Biển số không hợp lệ.")

                # Hiển thị thông báo
                try:
                    messagebox.showinfo("Thông báo", f"Biển số {bien_so_quet} không có trong danh sách đăng ký")
                except:
                    pass
                break

        except Exception as e:
            print(f" Lỗi tổng thể: {e}")
            print(traceback.format_exc())
            utils.update_label_content(label_status, f"Lỗi hệ thống: {e}", bg="red")
            break

    label_status.update()
    time.sleep(1)
    print("🏁 Kết thúc quá trình quét")
    return bien_so_quet, img_path, best_plate, image_dau_xe, bsx_dau, image_logo
