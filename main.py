import pickle
import CalorieDetection
from ultralytics import YOLO

# Modeli yükle ki class isimlerini alabilelim
model = YOLO("yolov8n_trained.pt")

with open("results.pkl", "rb") as f:
    results = pickle.load(f)

calorieDetector = CalorieDetection.CalorieDetector()
totalCalories = 0
images_with_detection = 0

for r in results:
    # Eğer en az bir obje varsa say
    if len(r.boxes) > 0:
        images_with_detection += 1

    # cls indexlerini isimlere çevirip calorie_dict ile toplam hesapla
    for box in r.boxes:
        class_name = model.names[int(box.cls)]
        totalCalories += calorieDetector.calorie_dict.get(class_name, 0)

total_images = len(results)
percentage_detected = (images_with_detection / total_images) * 100

print(f"Total Calories: {totalCalories} kcal")
print(f"Images with at least one detection: {images_with_detection}/{total_images} ({percentage_detected:.2f}%)")
