import os
import cv2
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import load_model


class CombinedSystem:
    def __init__(self, yolo_model_path, classifier_path, class_labels):
        print("Initializing combined system...")
        self.detector = YOLO(yolo_model_path)
        self.classifier = load_model(classifier_path)
        self.class_labels = class_labels
        print("Models loaded successfully.")

    def detect_and_classify(self, image_path, conf_threshold=0.5):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            return None

        results = self.detector(image)
        detections = []

        for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                             results[0].boxes.conf.cpu().numpy()):
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (32, 32)) / 255.0
            crop_resized = np.expand_dims(crop_resized, axis=0)

            preds = self.classifier.predict(crop_resized, verbose=0)
            class_id = int(np.argmax(preds))
            class_conf = float(np.max(preds))
            class_name = self.class_labels[class_id]

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "detection_confidence": float(conf),
                "classification_result": {
                    "class_name": class_name,
                    "confidence": class_conf
                }
            })

        return {"image_path": image_path, "detections": detections}

    def visualize_results(self, results, save_path=None):
        image = cv2.imread(results['image_path'])
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            cls = det['classification_result']['class_name']
            c_cls = det['classification_result']['confidence']
            c_det = det['detection_confidence']

            label = f"{cls} ({c_cls:.2f}/{c_det:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, image)
        else:
            cv2.imshow("Results", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class_labels = {
    'Speed limit (20km/h)',
    'Speed limit (30km/h)',
    'Speed limit (50km/h)',
    'Speed limit (60km/h)',
    'Speed limit (70km/h)',
    'Speed limit (80km/h)',
    'End of speed limit (80km/h)',
    'Speed limit (100km/h)',
    'Speed limit (120km/h)',
    'No passing',
    'No passing veh over 3.5 tons',
    'Right-of-way at intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vehicles',
    'Veh > 3.5 tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve left',
    'Dangerous curve right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work',
    'Traffic signals',
    'Pedestrians',
    'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End speed + passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End no passing veh > 3.5 tons'
}
combined_system = CombinedSystem(
    yolo_model_path="traffic_sign_detector.pt",
    classifier_path="traffic_sign_classifier.h5",
    class_labels=class_labels
)


def batch_test_combined_system(images_dir, output_dir="results_combined"):
    if combined_system is None:
        print("System not initialized.")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    print(f"Processing {len(image_files)} images...")

    summary = []

    for i, image_file in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {os.path.basename(image_file)}")

        results = combined_system.detect_and_classify(image_file, conf_threshold=0.5)
        output_path = os.path.join(output_dir, f"result_{os.path.basename(image_file)}")
        combined_system.visualize_results(results, save_path=output_path)

        for det in results['detections']:
            summary.append({
                'image': os.path.basename(image_file),
                'detection_conf': det['detection_confidence'],
                'class_name': det['classification_result']['class_name'],
                'classification_conf': det['classification_result']['confidence']
            })

    if summary:
        df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, "summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Summary saved: {summary_path}")
        print(f"Images processed: {len(image_files)}")
        print(f"Total detections: {len(summary)}")
        print(f"Avg detection conf: {df['detection_conf'].mean():.3f}")
        print(f"Avg classification conf: {df['classification_conf'].mean():.3f}")

    return summary

batch_results = batch_test_combined_system("dataset/images/test")
