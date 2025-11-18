
import sys
import os
from datetime import datetime
from XeMay.nhanDien import detect_license_plate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firebase_service import FirebaseService
from face_detection.train_face import capture_face_and_upload
from deepface import DeepFace
from firebase_admin import firestore
import time


db = firestore.client()
def run_license_scan_ra(label_status, canvas_old, canvas_new,root):
    firebase_service = FirebaseService()
    db = firestore.client()

    while True:
        # 1. Quét biển số
        bien_so, url_image_detected = detect_license_plate()
        check_plate = False
        if not bien_so:
            check_plate=False
            return check_plate
            continue
        bien_so_quet = bien_so.replace(".", "").upper()
        print("Biển số quét được:", bien_so_quet)

        # 2. Kiểm tra hợp lệ
        ds_bien_so = firebase_service.get_all_license_plates()
        if bien_so_quet not in ds_bien_so:
            label_status.config(text=f"Biển số {bien_so_quet} không hợp lệ ", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        # 3. Lấy dữ liệu biển số
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if not bien_so_data:

            time.sleep(2)
            continue

        # Kiểm tra có phải khách ưu tiên ko
        is_khach_uu_tien = is_khach_uutien(bien_so_quet)
        if not is_khach_uu_tien:
            # Kiểm tra số lượt có hợp lệ ko
            is_hople = update_soluot_khira(bien_so_quet)
            if not is_hople:
                label_status.config(text=f"Số lượt ra còn lại của biển số {bien_so_quet} không đủ ", bg="yellow")
                label_status.update()
                time.sleep(2)
                break

        # 4. Lấy timeline gần nhất
        today = datetime.today().strftime("%d%m%Y")
        xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xemay").document(bien_so_quet)
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

        if max_index >= 0:
            timeline_doc_id = f"timeline{max_index}"
            timeline_ref = xe_doc_ref.collection("timeline").document(timeline_doc_id)
            timeline_data = timeline_ref.get().to_dict()
            url_khuonmatvao = timeline_data.get("khuonmatvao")
            print("URL khuôn mặt vào gần nhất:", url_khuonmatvao)

        else:

            timeline_doc_id = None
            timeline_ref = None
            url_khuonmatvao = None

        # 5. Chụp khuôn mặt mới
        image_url_new_face = capture_face_and_upload()




        same_person = False
        CUSTOM_THRESHOLD = 0.35  # ngưỡng tùy chỉnh (càng thấp càng khắt khe)

        if url_khuonmatvao and image_url_new_face:
            try:
                result = DeepFace.verify(
                    img1_path=image_url_new_face,
                    img2_path=url_khuonmatvao,
                    model_name="ArcFace",  # model chính xác
                    detector_backend="retinaface",  # backend dò khuôn mặt tốt
                    distance_metric="cosine",  # metric phù hợp ArcFace
                    align=True,
                    enforce_detection=True
                )

                dist = float(result.get("distance", 1.0))
                thr = float(result.get("threshold", 0.0))
                verified_default = result.get("verified", False)

                # So sánh bằng ngưỡng custom
                verified_custom = dist <= CUSTOM_THRESHOLD
                same_person = verified_default and verified_custom

                print(f"Khoảng cách = {dist:.4f}, Ngưỡng mặc định = {thr:.4f}, Ngưỡng custom = {CUSTOM_THRESHOLD}")
                if same_person:

                    print("Kết quả: CÙNG 1 NGƯỜI")
                else:
                    print(" Kết quả: KHÁC NGƯỜI")

            except Exception as e:
                print("Lỗi khi so khớp khuôn mặt:", e)

        if same_person:
            check_plate = True
            print("=" * 50)
            print("✓ Xác nhận cùng 1 người")

            trangthai = bien_so_data.get('trangthai')
            print(f"Trạng thái: {trangthai}")

            if trangthai is False:
                print("→ Bắt đầu update Firestore...")

                # Update license plate
                try:
                    firebase_service.update_license_plate_field(bien_so_quet, True)
                    print("✓ Update license plate OK")
                except Exception as e:
                    print(f"✗ Lỗi update license plate: {e}")

                # Update solanra
                try:
                    doc = xe_doc_ref.get()
                    solanra = doc.to_dict().get("solanra", 0) if doc.exists else 0
                    solanra += 1
                    xe_doc_ref.set({"solanra": solanra}, merge=True)
                    print(f"✓ Update solanra = {solanra} OK")
                except Exception as e:
                    print(f"✗ Lỗi update solanra: {e}")

                # Update timeline
                try:
                    if timeline_ref:
                        time_now = datetime.now().strftime("%H:%M:%S")
                        timeline_ref.set({
                            "timeout": time_now,
                            "biensoxera": url_image_detected,
                            "khuonmatra": image_url_new_face
                        }, merge=True)
                        print(f"✓ Update timeline {timeline_doc_id} OK")
                    else:
                        print("⚠ Không có timeline để update")
                except Exception as e:
                    print(f"✗ Lỗi update timeline: {e}")

                print("=" * 50)
                return check_plate
        else:

            label_status.config(text=f" Không phải cùng người", bg="red")

        label_status.update()
        time.sleep(2)  # delay giữa các lần quét




