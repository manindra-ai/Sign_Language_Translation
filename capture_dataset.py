import cv2
import os

# Ask for cluster and subcategory
cluster = input("Enter main category (travelling / new_place / food_place): ").strip()
sub = input("Enter subcategory (A/B/C/D): ").strip().upper()

# Folder path
base_dir = "Dataset"
path = os.path.join(base_dir, cluster, sub)
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

count = 0
print("\n🎬 Capturing images... Press 'q' to stop.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow(f"Capturing: {cluster}/{sub}", frame)

    # Save every 3rd frame to control image count
    if count % 3 == 0:
        img_name = os.path.join(path, f"{sub}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"📸 Saved: {img_name}")

    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n✅ Saved {count//3} images to {path}")
        break

cap.release()
cv2.destroyAllWindows()
