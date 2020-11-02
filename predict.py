from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model as tf_load_model
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import torch.backends.cudnn as cudnn
import pandas as pd
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import argparse
import cv2
from tqdm import tqdm
from scipy.signal import lfilter
from itertools import chain
from matplotlib import pyplot as plt
import gc


def free_cache():
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


def load_model(model, pretrained_path, load_to_cpu):
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def face_detect(image, device):
    torch.set_grad_enabled(False)
    cfg = cfg_re50

    # net and model
    print("Loading face detector model...")
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, 'models/Resnet50_Final.pth', True)
    net = net.to(device)
    net.eval()

    #cudnn.benchmark = True

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

    print("Detecting faces...")
    loc, conf, _ = net(img)  # forward pass

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
    
    del net
    return len(bb), bb


def mask_detect(image, model):
    img_UMat = cv2.UMat(image)
    face = cv2.resize(img_UMat, (224, 224))
    face = cv2.UMat.get(face)
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    (mask, withoutMask) = model.predict(face)[0]
    label = "Mask" if mask > withoutMask else "No Mask"
    return label


def predidct_age_gender_race(img, num_faces, bboxes, id_, device):
    print("Loading age, gender, and race models...")

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('models/res34_fair_align_multi_7_20190809.pt'))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    #model_fair_4 = torchvision.models.resnet34(pretrained=True)
    #model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    #model_fair_4.load_state_dict(torch.load('models/fairface_alldata_4race_20191111.pt'))
    #model_fair_4 = model_fair_4.to(device)
    #model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading face mask detector model...")
    model = tf_load_model("models/mask_detector.model")

    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    #race_scores_fair_4 = []
    #race_preds_fair_4 = []
    
    face_mask = []

    print('Classifying genders, ages, and races...')
    for i in tqdm(range(num_faces)):
        face_names.append(i)
        bb = bboxes[i]
        image = img[bb[1]:bb[3], bb[0]:bb[2]]
        label = mask_detect(image, model)
        face_mask.append(label)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        #outputs = model_fair_4(image)
        #outputs = outputs.cpu().detach().numpy()
        #outputs = np.squeeze(outputs)

        #race_outputs = outputs[:4]
        #race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        #race_pred = np.argmax(race_score)

        #race_scores_fair_4.append(race_score)
        #race_preds_fair_4.append(race_pred)

    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           #race_preds_fair_4,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair, #race_scores_fair_4,
                           gender_scores_fair,
                           age_scores_fair,
                           bboxes]).T
    result.columns = ['index',
                      'race_preds_fair',
                      #'race_preds_fair_4',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      #'race_scores_fair_4',
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

    # race fair 4
    #result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    #result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    #result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    #result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

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

    result[['index', 'race', 'gender', 'age', 'bbox', 'age_preds_fair', 'age_scores_fair']].to_csv("output/log_{}.csv".format(id_), index=False)
    print("Saved results at log_{}.csv".format(id_))

    del model
    del model_fair_7

    print("Creating a bounding box image...")
    shape = img.shape
    new_img = cv2.UMat(img)
    for i in tqdm(range(num_faces)):
        color = color = (0, 255, 0) if face_mask[i] == "Mask" else (0, 0, 255)
        bb = bboxes[i]
        cv2.rectangle(new_img, (bb[0], bb[1]), (bb[2], bb[3]), color, thickness=2)
        contour = get_contours(bb, shape)
        age = get_age(result['age_scores_fair'][i])
        cv2.putText(img=new_img, text='{} {} {}'.format(result['gender'][i], age, result['race'][i]),
                    org=(contour[0], contour[1]), fontScale=1, color=color, thickness=2, lineType=cv2.LINE_4,
                    fontFace=cv2.FONT_HERSHEY_PLAIN)

    final_img = cv2.UMat.get(new_img)
    return final_img


def get_age(results):
    res = [np.linspace(results[0], results[1], 5), np.linspace(results[1], results[2], 10),
           np.linspace(results[2], results[3], 10), np.linspace(results[3], results[4], 10),
           np.linspace(results[4], results[5], 10), np.linspace(results[5], results[6], 10),
           np.linspace(results[6], results[7], 10), np.linspace(results[7], results[8], 10)]
    arr = np.array(list(chain.from_iterable(res)))
    
    n = 7
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, arr)
    age = np.argmax(yy)
    #x = np.arange(0, 75)
    #plt.plot(x, yy)
    #plt.show()
    return age


def get_contours(bbox, shape):
    contour = []
    x = bbox[0]-20
    if x < 0:
        contour.append(0)
    else:
        contour.append(x)
    y = bbox[3]+20
    if y > shape[1]:
        contour.append(shape[1])
    else:
        contour.append(y)
    return contour


if __name__ == "__main__":
    free_cache()

    id_ = np.random.randint(0, 1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest='img', action='store', )
    print("Torch CUDA available: {}".format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    dir_ = os.path.abspath(os.path.dirname(__file__))
    img = cv2.imread("{}/{}".format(dir_, args.img))

    free_cache()

    num_faces, bboxes = face_detect(img, device)
    print("Faces number: {}".format(num_faces))

    free_cache()

    final_img = predidct_age_gender_race(img, num_faces, bboxes, id_, device)
    cv2.imwrite("output/output_{}.jpg".format(id_), final_img)

    free_cache()
    print("All done!")
