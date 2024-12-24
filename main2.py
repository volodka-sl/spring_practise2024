import cv2
from ultralytics import YOLO

video_source = "output2.avi"
cap = cv2.VideoCapture(video_source)

model = YOLO('yolov8s.pt')

left_cars_count = 0
rigth_cars_count = 0

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
horizontal_line_position = frame_height // 2
vertical_line_position = frame_width // 2

lmdb = 1.2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.line(frame, (0, horizontal_line_position), (frame_width, horizontal_line_position), (0, 255, 255), 2)

    results = model.predict(frame, classes=[2, 5, 7], conf=0.2, iou=0.2)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_y = (y1 + y2) / 2  # центра автомобиля по Y
            center_x = (x1 + x2) / 2  # по X

            if horizontal_line_position - lmdb <= center_y <= horizontal_line_position + lmdb:
                if center_x > vertical_line_position:
                    rigth_cars_count += 1
                    print(box.id)
                else:
                    left_cars_count += 1

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.putText(frame, f'Cars from left counted: {left_cars_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Cars from right counted: {rigth_cars_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
