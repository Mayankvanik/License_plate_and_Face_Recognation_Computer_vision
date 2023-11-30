from ultralytics import YOLO
import cv2
from util import get_car
from sort.sort import Sort
import numpy as np

mot_tracker = Sort()

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best_plate03.pt')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('my3.mp4')

# Update class indices based on your model
vehicles_and_faces = [0, 1, 2, 3, 4, 5, 6, 8, 11]  # Include both classes (0 for Vehicle_id, 1 for Face)

frame_nrm = -1
ret = True
# Get video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video8.avi', fourcc, fps, (width, height))

while ret:
    frame_nrm += 1
    ret, frame = cap.read()
    if ret and frame_nrm < 10000:
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(50, 50),
                                              maxSize=(600,600))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_crop = frame[y:y + h, x:x + w]
            img_blur = cv2.GaussianBlur(face_crop, (71, 71), 0)
            frame[y:y + h, x:x + w] = img_blur

        # Vehicle and face detection
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles_and_faces:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates_result = license_plate_detector(frame)
        for class_id, license_plates in enumerate(license_plates_result):
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = license_plate

                # assign license plate to car or face based on class_id
                x_coord, y_coord, _, _, entity_id = get_car(license_plate, track_ids)

                # crop license plate
                print(f"Detected entity with class_id: {class_id}, entity_id: {entity_id} ")
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # process license plate
                license_plate_blur = cv2.GaussianBlur(license_plate_crop, (111, 111), 0)

                # Replace the original license plate region with the blurred one
                frame[int(y1):int(y2), int(x1):int(x2), :] = license_plate_blur

        output_video.write(frame)

        cv2.imshow('Blurred License Plates', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# # Load the smiley image with an alpha channel (transparency)
# smiley_img = cv2.imread('smile.png', cv2.IMREAD_UNCHANGED)
#
# cap = cv2.VideoCapture('my1.mp4')
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(50, 50),
#                                           maxSize=(600, 600))
#
#     for (x, y, w, h) in faces:
#         # Resize the smiley image to match the size of the detected face
#         resized_smiley = cv2.resize(smiley_img, (w, h))
#
#         # Extract the alpha channel from the resized smiley image
#         mask = resized_smiley[:, :, 0] > 0
#
#         # Overlay the resized smiley image onto the frame
#         frame[y:y + h, x:x + w][mask] = resized_smiley[mask]
#
#     cv2.imshow('Smiley Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()