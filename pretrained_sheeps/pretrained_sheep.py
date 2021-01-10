import cv2
import numpy as np

img = cv2.imread("C://Users//Merve//Desktop//imona_case//images//456.jpg")
# print(img.shape)

img_width = img.shape[1]
img_height = img.shape[0]

img_blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
# print(img_blob.shape)

labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
          "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
          "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
          "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

colors = ["0,255,255", "0,0,255", "0,255,0"]
colors = [np.array(c.split(",")).astype("int") for c in colors]
colors = np.array(colors)
colors = np.tile(colors, (18, 1))
# print(colors)

model = cv2.dnn.readNetFromDarknet("C://Users//Merve//Desktop//imona_case//models//yolov3.cfg",
                                   "C://Users//Merve//Desktop//imona_case//models//yolov3.weights")
layers = model.getLayerNames()
output_layer = [layers[l[0] - 1] for l in model.getUnconnectedOutLayers()]

model.setInput(img_blob)
detection_layers = model.forward(output_layer)

ids_list = []
boxes_list = []
confidences_list = []

for detection in detection_layers:
    for object in detection:
        scores = object[5:]
        predicted = np.argmax(scores)
        confidence = scores[predicted]
        if confidence >= 0.70:
            label = labels[predicted]
            bounding_box = object[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))

            ids_list.append(predicted)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

            max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.6, 0.7)
            for max_id in max_ids:
                max_class_id = max_id[0]
                box = boxes_list[max_class_id]
                start_x = box[0]
                start_y = box[1]
                box_width = box[2]
                box_height = box[3]
                predicted = ids_list[max_class_id]
                label = labels[predicted]
                confidence = confidences_list[max_class_id]
                end_x = start_x + box_width
                end_y = start_y + box_height
                box_color = colors[predicted]
                box_color = [int(c) for c in box_color]
                label = "{}: {:.2f}%".format(label, confidence * 100)
                print("predicted object {}".format(label))
                cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 5)
                cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, box_color, 5)

cv2.imshow("detection", img)
cv2.waitKey(0)
