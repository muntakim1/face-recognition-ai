import io
import math

import numpy
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from yolo5face.get_model import get_model


def match_faces(unknown_image, reference_image, tolerance):
    """If required, create a face detection pipeline using MTCNN"""
    mtcnn = MTCNN()

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained="casia-webface").eval()

    model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)

    half = 0.5
    img_ref = Image.open(reference_image)
    img_ref = img_ref.resize([int(half * s) for s in img_ref.size])

    img_test = Image.open(io.BytesIO(unknown_image))
    img_test = img_test.resize([int(half * s) for s in img_test.size])

    boxes, _ = model(numpy.array(img_test)[:, :, ::-1])

    results = []

    # # If faces are detected, 'boxes' will contain the bounding box coordinates
    if boxes is not None:
        for _, ref_box in enumerate(boxes[0]):
            x1, y1, x2, y2 = ref_box
            cropped_face = img_test.crop((x1, y1, x2, y2))
            aligned_test = mtcnn(cropped_face)
            aligned = mtcnn(img_ref)
            test_embeddings = resnet(aligned_test.unsqueeze(0)).detach()
            embeddings = resnet(aligned.unsqueeze(0)).detach()
            distance = torch.norm(embeddings - test_embeddings, p=2)
            if distance.item() <= tolerance:
                results.append(True)
            else:
                results.append(False)

    return results
