import io
import math
import os

import numpy
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from yolo5face.get_model import get_model


def match_faces(unknown_image, known_image, tolerance=0.8):
    """
    Matching faces of the two images if the images are of a same person then the function will return true

    unknown_image: the image with known person or unknown persons,
    Known_image: The reference image of the person.
    Known_image_Name: The name of the reference image of the person,
    tolerance: distance threshold for images defualt 0.8"""
    mtcnn = MTCNN()

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained="casia-webface").eval()

    model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)

    boxes, _ = model(numpy.array(unknown_image)[:, :, ::-1])

    result = []
    # # If faces are detected, 'boxes' will contain the bounding box coordinates
    if boxes is not None:
        for _, ref_box in enumerate(boxes[0]):
            try:
                x1, y1, x2, y2 = ref_box
                cropped_face = unknown_image.crop((x1, y1, x2, y2))
                aligned_test = mtcnn(cropped_face)
                aligned = mtcnn(known_image)
                test_embeddings = resnet(aligned_test.unsqueeze(0)).detach()
                embeddings = resnet(aligned.unsqueeze(0)).detach()
                distance = torch.norm(embeddings - test_embeddings, p=2)
                if distance.item() <= tolerance:
                    result.append(True)
                else:
                    result.append(False)
            except:
                pass
    return result


def show_detections(unknown_image, known_image, known_image_name, tolerance=0.8):
    """
    Show detections draws bounding boxes arround the detections
    -----------------------------------------------------------
    unknown_image: the image with known person or unknown persons,
    Known_image: The reference image of the person.
    Known_image_Name: The name of the reference image of the person,
    tolerance: distance threshold for images defualt 0.8
    """
    mtcnn = MTCNN()

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained="casia-webface").eval()

    model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)

    boxes, _ = model(numpy.array(unknown_image)[:, :, ::-1])
    font = ImageFont.truetype(
        "Arial.ttf",
        size=28,
    )
    # # If faces are detected, 'boxes' will contain the bounding box coordinates
    if boxes is not None:
        for _, ref_box in enumerate(boxes[0]):
            try:
                x1, y1, x2, y2 = ref_box
                cropped_face = unknown_image.crop((x1, y1, x2, y2))
                aligned_test = mtcnn(cropped_face)
                aligned = mtcnn(known_image)
                test_embeddings = resnet(aligned_test.unsqueeze(0)).detach()
                embeddings = resnet(aligned.unsqueeze(0)).detach()
                distance = torch.norm(embeddings - test_embeddings, p=2)
                if distance.item() <= tolerance:
                    draw = ImageDraw.Draw(unknown_image)
                    draw.text(
                        ref_box,
                        f"{known_image_name}",
                        font=font,
                        fill=(255, 255, 255, 128),
                    )
                    draw.rectangle(ref_box, outline="red", width=3)
            except:
                pass
    return unknown_image
