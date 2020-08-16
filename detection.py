import math
import cv2
import numpy as np


def draw_overlay(l_img, s_img, x_offset, y_offset):
    # Reference: https://stackoverflow.com/a/14102014/1983050

    y1 = y_offset 
    y2 = y_offset + s_img.shape[0]
    x1 = x_offset
    x2 = x_offset + s_img.shape[1]
    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                alpha_l * l_img[y1:y2, x1:x2, c])

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, -angle * 180 / math.pi, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


cap = cv2.VideoCapture(0)

# Load YOLO
net = cv2.dnn.readNet(
    '/home/lukic/Downloads/yolov3-tiny_train (3).backup', 'yolov3-tiny.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

wheel_image = cv2.imread('steering_wheel.png', -1)

while True:
    _, img = cap.read()
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
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                objects.append({
                    'x': x,
                    'y': y,
                    'center_x': center_x,
                    'center_y': center_y,
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
    filtered_objects = sorted(
        filtered_objects, key=lambda x: x['confidence'], reverse=True)
    filtered_objects = filtered_objects[:2]

    # Visualize
    for object in filtered_objects:
        x = object['x']
        y = object['y']
        w = object['width']
        h = object['height']
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(filtered_objects) == 2:
        left_object = filtered_objects[1]
        right_object = filtered_objects[0]
        if filtered_objects[0]['center_x'] < filtered_objects[1]['center_x']:
            left_object = filtered_objects[0]
            right_object = filtered_objects[1]

        #print(right_object['center_x'] - left_object['center_x'])
        width_diff = right_object['center_x'] - left_object['center_x']
        height_diff = right_object['center_y'] - left_object['center_y']
        wheel_width = int(math.sqrt(width_diff**2 + height_diff**2))
        wheel_x = (right_object['center_x'] + left_object['center_x']) / 2 - wheel_width / 2
        wheel_x = max(min(wheel_x, img.shape[0] - wheel_width), 0)
        wheel_y = (right_object['center_y'] + left_object['center_y']) / 2 - wheel_width / 2
        wheel_y = max(min(wheel_y, img.shape[1] - wheel_width), 0)
        wheel_angle = math.atan2(height_diff, width_diff)
        wheel_transformed = cv2.resize(wheel_image, (wheel_width, wheel_width))
        wheel_transformed = rotate_image(wheel_transformed, wheel_angle)
        draw_overlay(img, wheel_transformed, int(wheel_x), int(wheel_y))

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
