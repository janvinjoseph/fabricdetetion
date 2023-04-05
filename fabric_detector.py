import cv2
import os

class FabricDefectDetector:

    def __init__(self):
        # Load Network
        weights_path = "dnn_model/custom.weights"
        cfg_path = "dnn_model/Custom.cfg"
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255)

        # Allow classes containing Fabric Defects only
        self.classes_allowed = [0]

    def detect_defects(self, img):
        # Detect Objects
        defect_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                # Skip detection with low confidence
                continue

            if class_id in self.classes_allowed:
                defect_boxes.append(box)

        # Print total count
        print("Total Defects Detected:", len(defect_boxes))

        return defect_boxes


if __name__ == '__main__':
    # Create Fabric Defect Detector Object
    detector = FabricDefectDetector()

    # Define image folder path
    images_folder_path = "images/"

    # Iterate through images in folder and detect defects
    for image_file in os.listdir(images_folder_path):
        # Read image
        image_path = os.path.join(images_folder_path, image_file)
        img = cv2.imread(image_path)

        # Detect defects
        defect_boxes = detector.detect_defects(img)

        # Draw bounding boxes on image
        for box in defect_boxes:
            cv2.rectangle(img, box, (0, 0, 255), 2)

        # Display image
        cv2.imshow("Image", img)
        cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()

