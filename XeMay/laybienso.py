
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Lấy thông tin kết nối
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")


def get_all_license_plates():

    try:
        # Kết nối MongoDB
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DATABASE_NAME]
        users_collection = db['users']

        # Query lấy license_plate
        cursor = users_collection.find(
            {"license_plate": {"$exists": True, "$ne": None, "$ne": ""}},
            {"license_plate": 1, "_id": 0}
        )

        # Trích xuất license_plate thành list
        license_plates = [doc["license_plate"] for doc in cursor]

        # Đóng kết nối
        client.close()

        return license_plates

    except Exception as e:
        print(f"Lỗi khi lấy license_plate: {e}")
        return []


# # ===== TEST =====
# if __name__ == "__main__":
#     plates = get_all_license_plates()
#     print(f"✅ Lấy được {len(plates)} license_plate:")
#     for plate in plates[:10]:  # Hiển thị 10 biển đầu
#         print(f"  - {plate}")
#     if len(plates) > 10:
#         print(f"  ... và {len(plates) - 10} biển số khác")