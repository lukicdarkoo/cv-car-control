import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# Load YOLO
net = cv2.dnn.readNet(
    '/home/lukic/Downloads/yolov3-tiny_train.backup', 'yolov3-tiny.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

while True:
    ret, img = cap.read()
    image_height, image_width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Exporting detected fists
    objects = []
    for out in outs:
        for detection in out:
            confidence = detection[5]
            if confidence > 0.5:
                center_x = detection[0] * image_width
                center_y = detection[1] * image_height
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                objects.append({
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'confidence': confidence
                })

    # Delete overlaped
    filtered_objects = []
    for object_1 in objects:
        accept_object_1 = True
        for object_2 in objects:
            if object_1['x'] < object_2['x'] + object_2['width'] and \
                object_1['x'] + object_1['width'] > object_2['x'] and \
                object_1['y'] < object_2['y'] + object_2['height'] and \
                object_1['y'] + object_1['height'] > object_2['y'] and \
                object_1['confidence'] < object_2['confidence']:
                accept_object_1 = False
        if accept_object_1:
            filtered_objects.append(object_1)

    # Take 2 objects with the highest confidence
    filtered_objects = sorted(filtered_objects, key=lambda x: x['confidence'], reverse=True)
    filtered_objects = filtered_objects[:2]

    # Visualize
    for object in filtered_objects:
        x = object['x']
        y = object['y']
        w = object['width']
        h = object['height']
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
