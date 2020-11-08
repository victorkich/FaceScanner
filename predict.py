from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.preprocessing.image import img_to_array
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from data import cfg_mnet, cfg_re50
from torch.autograd import Variable
from torchvision import transforms
from scipy.signal import lfilter
from itertools import chain
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import torchvision
import argparse
import torch
import cv2
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--img', dest='img', action='store', )
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch CUDA available: {}".format(torch.cuda.is_available()))
cudnn.benchmark = True


def free_cache(model=None):
    print('Cleaning CUDA cache...')
    if model: del model
    gc.collect()
    torch.cuda.empty_cache()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_retina_model():
    print("Loading face detector model...")
    torch.set_grad_enabled(False)
    cfg = cfg_re50
    model = RetinaFace(cfg=cfg, phase='test')
    if not torch.cuda.is_available():
        pretrained_dict = torch.load('models/Resnet50_Final.pth', map_location=lambda storage, loc: storage)
    else:
        dev = torch.cuda.current_device()
        pretrained_dict = torch.load('models/Resnet50_Final.pth', map_location=lambda storage, loc: storage.cuda(dev))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_mask_model():
    print("Loading mask detector model...")
    model = tf_load_model("models/mask_detector.model")
    return model


def face_detect(model, image):
    print("Detecting faces...")
    cfg = cfg_re50
    resize = 1

    img_raw = image
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    
    loc, conf, _ = model(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)

    # ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:750, :]

    # show image
    bb = []
    for b in dets:
        if b[4] < 0.6:
            continue
        b = list(map(int, b))
        bb.append(b)
    return bb


def mask_detect(model, image, bboxes):
    print('Detecting masks...')
    mask_list = []
    for bb in tqdm(bboxes):
        for e, b in enumerate(bb):
            if b < 0:
                bb[e] = 0
        raw_face = image[bb[1]:bb[3], bb[0]:bb[2]]
        img_UMat = cv2.UMat(raw_face)
        face = cv2.cvtColor(img_UMat, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = cv2.UMat.get(face)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        mask_list.append(label)
    return mask_list


def load_fairface_model():
    print("Loading age, gender, and race models...")
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load('models/res34_fair_align_multi_7_20190809.pt'))
    model = model.to(device)
    model.eval()
    return model


def predict_age_gender_race(model, img, bboxes):
    print('Classifying genders, ages, and races...')
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    face_names = []
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []

    for i, bb in enumerate(tqdm(bboxes)):
        for e, b in enumerate(bb):
            if b < 0:
                bb[e] = 0
        raw_face = img[bb[1]:bb[3], bb[0]:bb[2]]
        img_UMat = cv2.UMat(raw_face)
        raw_face = cv2.cvtColor(img_UMat, cv2.COLOR_BGR2RGB)
        face = cv2.UMat.get(raw_face)

        image = trans(face)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        face_names.append(i)
        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair,
                           gender_scores_fair,
                           age_scores_fair,
                           bboxes]).T
    result.columns = ['index',
                      'race_preds_fair',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'gender_scores_fair',
                      'age_scores_fair',
                      "bbox"]

    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '7'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '15'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '25'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '35'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '45'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '55'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '65'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '75'
    return result


def create_log_and_image(id_, img, bboxes, agr_table, mask_list):
    print("Creating a bounding box image...")
    shape = img.shape
    new_img = cv2.UMat(img)
    for i, bb in enumerate(tqdm(bboxes)):
        color = (0, 255, 0) if mask_list[i] == "Mask" else (0, 0, 255)
        cv2.rectangle(new_img, (bb[0], bb[1]), (bb[2], bb[3]), color, thickness=2)
        contour_top, contour_bot = get_contours(bb, shape)
        age = get_age(agr_table['age_scores_fair'][i])
        cv2.putText(img=new_img, text='{} {}'.format(agr_table['gender'][i], age), org=(contour_top[0], contour_top[1]),
                    fontScale=1, color=color, thickness=2, lineType=cv2.LINE_4, fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.putText(img=new_img, text=agr_table['race'][i], org=(contour_bot[0], contour_bot[1]), fontScale=1, 
                    color=color, thickness=2, lineType=cv2.LINE_4, fontFace=cv2.FONT_HERSHEY_PLAIN)

    final_img = cv2.UMat.get(new_img)
    cv2.imwrite("output/output_{}.jpg".format(id_), final_img)
    print("Saved image at output/output_{}".format(id_))

    print("Saving log...")
    mask_table = pd.DataFrame(mask_list, columns=['mask'])
    result = pd.concat([agr_table, mask_table], axis=1) 
    result[['index', 'race', 'gender', 'age', 'bbox', 'mask']].to_csv("output/log_{}.csv".format(id_), index=False)
    print("Saved results at log_{}.csv".format(id_))


def get_age(results):
    res = [np.linspace(results[0], results[1], 5), np.linspace(results[1], results[2], 10),
           np.linspace(results[2], results[3], 10), np.linspace(results[3], results[4], 10),
           np.linspace(results[4], results[5], 10), np.linspace(results[5], results[6], 10),
           np.linspace(results[6], results[7], 10), np.linspace(results[7], results[8], 10)]
    arr = np.array(list(chain.from_iterable(res)))
    
    n = 2
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, arr)
    age = np.argmax(yy)
    #x = np.arange(0, 75)
    #plt.plot(x, yy)
    #plt.show()
    if age < 30 and age > 20: age = int(age - (age - 20) / 2)
    return age


def get_contours(bbox, shape):
    x = bbox[0] + 5
    y = bbox[3] - 10
    contourTop = [x, y]
    y = bbox[1] + 20
    contourBot = [x, y]
    return contourTop, contourBot


def open_image():
    print('Opening image...')
    dir_ = os.path.abspath(os.path.dirname(__file__))
    img = cv2.imread("{}/{}".format(dir_, args.img))
    outputs = os.listdir("{}/output".format(dir_))
    id_ = int(len(outputs)/2+1)
    return id_, img


if __name__ == "__main__":
    free_cache()
    id_, img = open_image()

    model = load_retina_model()
    bboxes = face_detect(model, img)
    print("Faces number: {}".format(len(bboxes)))
    free_cache(model)
    
    model = load_mask_model()
    mask_list = mask_detect(model, img, bboxes)
    free_cache(model)

    model = load_fairface_model()
    agr_table = predict_age_gender_race(model, img, bboxes)
    free_cache(model)

    create_log_and_image(id_, img, bboxes, agr_table, mask_list)
    print("All done!")
