from firebase_admin import credentials, db
import firebase_admin
import os

class FirebaseService:
    def __init__(self):
        print("Đã chạy __init__")
        if not firebase_admin._apps:
            print("Đang khởi tạo Firebase...")

            # Đường dẫn tới serviceAccountKey.json
            cred_path = "D:/thuc_tap_tot_nghiep/smallPaking-Destop/serviceAccountKey.json"
            cred = credentials.Certificate(cred_path)

            # ✅ Khởi tạo CHO CẢ Realtime Database VÀ Firestore
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://smallparking-41c54-default-rtdb.firebaseio.com/'
            })
            print("✓ Firebase đã được khởi tạo")
    def get_all_license_plates(self):
        ref = db.reference("biensotrongbai")
        data = ref.get()
        return list(data.keys()) if data else []

    def get_license_plate_data(self, plate):
        """
        Lấy toàn bộ dữ liệu của một biển số từ nhánh 'license_plates'
        """
        ref = db.reference(f'biensotrongbai/{plate}')
        return ref.get()

    def update_license_plate_field(self, plate,value):

        ref = db.reference(f'trangthaicong')  # ✅ Cùng cấp với biensotrongbai
        ref.set(value)
    def update_canhbao(self,plate,value):
        ref = db.reference(f'/biensotrongbai/{plate}/canhbao')
        ref.set(value)

    def delete_license_plate(self, plate):
        ref = db.reference(f'/biensotrongbai/{plate}')
        ref.delete()
    def has_khach(self, plate) -> bool:
        """
        Kiểm tra trong biensotrongbai/{plate} có tồn tại trường 'khach' hay không
        Trả về True nếu có, False nếu không.
        """
        ref = db.reference(f'biensotrongbai/{plate}/khach')
        data = ref.get()
        return data is not None