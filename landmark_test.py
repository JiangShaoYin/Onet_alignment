from mtcnn.core.imagedb import ImageDB
from mtcnn.core.image_reader import TrainImageReader
from mtcnn_test import test_onet
import datetime
import cv2
import os
from mtcnn.core.models import PNet,RNet,ONet,LossFn
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import mtcnn.core.image_tools as image_tools
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
import numpy as np
from mtcnn.core.vision import vis_face,vis_face_test, vis_face_raw, vis_face_landmark_img_label

annotation_file = "./data_set/face_landmark/1200.txt"

class LandmarkTestor(object):
    def __init__(self, annotation, output_dir):
        self.output_dir = output_dir
        self.imagedb = ImageDB(annotation)
        self.gt_imdb = self.imagedb.load_imdb()


    def test_face_alignment(self, test_moudel, savePic):
        pnet, rnet, onet_jiang = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                                  r_model_path="./original_model/rnet_epoch.pt",
                                                  o_model_path="./original_model/" + test_moudel + ".pt", use_cuda=False)
        mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet_jiang, min_face_size=24)


        train_data = TrainImageReader(self.gt_imdb, 48, batch_size = 100, shuffle=False)  # 读入1个batch的数据
        # train_data.reset()
        total_errors = 0

        cnt = 0
        for i,(images,(gt_labels,gt_bboxes,gt_landmarks))in enumerate(train_data):  #取1个batch
            # if i == 12:
            # print(i)
            list_imgs = [images[i,:,:,:] for i in range(images.shape[0]) ] # 100张图片

            list_bboxes = [gt_bboxes[i,:] for i in range(gt_bboxes.shape[0]) ]
            list_gt_landmarks = [gt_landmarks[i,:] for i in range(gt_landmarks.shape[0]) ]
            mix = list(zip(list_imgs, list_bboxes,  list_gt_landmarks))
            batch_errors = []

            for img, gt_bbox, gt_landmark in mix:   # 取1个图片

                bboxs, landmarks = mtcnn_detector.detect_face(img)  # 原始的图片用原始网络检测
                if landmarks.size:
                    cnt += 1
                    bboxs = bboxs[:1]  # 多个检测框保留第一个
                    landmarks = landmarks[:1]
                    if savePic:
                        vis_face(img, bboxs, landmarks, self.output_dir + str(cnt) + ".jpg")  # 保存图片
                    gt_landmark = np.array(gt_landmark).reshape(5, 2)
                    landmarks = np.array(landmarks).reshape(5,2)

                    normDist = np.linalg.norm(gt_landmark[1] - gt_landmark[0])  # 左右眼距离
                    error = np.mean(np.sqrt(np.sum((landmarks - gt_landmark) ** 2, axis=1))) / normDist

                    # print("the %sth pic error is : %s"%(cnt, error))
                    batch_errors.append(error)

            batch_errors = np.array(batch_errors).sum()
            total_errors += batch_errors
            print("%s:   %s pics mean error is %s" % (datetime.datetime.now(), cnt,  total_errors / cnt))

        f = open("landmark_test.txt", "a+")
        f.write("%s, moudel_name:%s.pt, %s pics mean error is %s\n" % (datetime.datetime.now(), test_moudel, cnt, np.array(total_errors).reshape(1,-1).sum()/cnt))
        f.close()

        print("%s:%s pics mean error is %s" % (datetime.datetime.now(), cnt, total_errors / cnt))

    def test_Onet_without_PRnet(self,annotation, outputDir, test_moudel, xxyy, savePic):
        imagedb = ImageDB(annotation)
        gt_imdb = imagedb.load_imdb()
        pnet, rnet, onet_jiang = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                                  r_model_path="./original_model/rnet_epoch.pt",
                                                  o_model_path="./original_model/" + test_moudel + ".pt", use_cuda=False)
        mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet_jiang, min_face_size=24)


        test_data = TrainImageReader(gt_imdb, 48, batch_size = 100, shuffle=False)  # 读入1个batch的数据
        # train_data.reset()
        total_errors = 0

        cnt = 0
        for i,(images,(gt_labels,gt_bboxes,gt_landmarks))in enumerate(test_data):  # 取1个batch
            list_imgs = [images[i,:,:,:] for i in range(images.shape[0]) ] # 100张图片

            list_bboxes = [gt_bboxes[i,:] for i in range(gt_bboxes.shape[0]) ]
            list_gt_landmarks = [gt_landmarks[i,:] for i in range(gt_landmarks.shape[0]) ]
            mix = list(zip(list_imgs, list_bboxes,  list_gt_landmarks))
            batch_errors = []

            for img, gt_bbox, gt_landmark in mix:   # 取1个图片
                if xxyy:
                    bboxs, landmarks = mtcnn_detector.detect_onet_xxyy(img, gt_bbox)  # 原始的图片用原始网络检测,xxyy
                else:
                    bboxs, landmarks = mtcnn_detector.detect_onet(img, gt_bbox)  # 原始的图片用原始网络检测,xxyy

                if landmarks.size:
                    cnt += 1
                    bboxs = bboxs[:1]  # 多个检测框保留第一个
                    landmarks = landmarks[:1]
                    if savePic:
                        vis_face(img, bboxs, landmarks, self.output_dir + str(cnt) + ".jpg")  # 保存图片
                    gt_landmark = np.array(gt_landmark).reshape(5, 2)
                    landmarks = np.array(landmarks).reshape(5,2)

                    normDist = np.linalg.norm(gt_landmark[1] - gt_landmark[0])  # 左右眼距离
                    error = np.mean(np.sqrt(np.sum((landmarks - gt_landmark) ** 2, axis=1))) / normDist

                    batch_errors.append(error)

            batch_errors = np.array(batch_errors).sum()
            total_errors += batch_errors
            print("%s:   %s pics mean error is %s" % (datetime.datetime.now(), cnt,  total_errors / cnt))
            if cnt > 999:
                print("%s:%s pics mean error is %s" % (datetime.datetime.now(), cnt, total_errors / cnt))
                f = open("landmark_test.txt", "a+")
                f.write("%s, moudel_name:%s.pt, %s pics mean error is %s\n" % (datetime.datetime.now(), test_moudel, cnt, np.array(total_errors).reshape(1, -1).sum() / cnt))
                f.close()
                return


        print("%s:%s pics mean error is %s" % (datetime.datetime.now(), cnt, total_errors / cnt))

    def test_Onet_without_PRnet1(inputDir,outputDir,model):
        pnet, rnet, onet_jiang = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                                  r_model_path="./original_model/rnet_epoch.pt",
                                                  o_model_path="./original_model/" + model + ".pt", use_cuda=False)
        mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet_jiang, min_face_size=24)

        files = os.listdir(inputDir)
        i = 0
        for image in files:
            i += 1
            image = os.path.join(inputDir, image)

            img = cv2.imread(image)
            img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            landmarks2_jiang = mtcnn_detector.detect_onet_raw(img)

            vis_face_test(img_bg, landmarks2_jiang,  outputDir + model + "-" + str(i) + ".jpg")
            if i == 50:
                break

    def test_img_and_label_48(self, label_path, store = True, store_path = "./landmark_show/"):
        if store and not os.path.exists(store_path):
                os.makedirs(store_path)
        with open(label_path, 'r') as f:
            annotations = f.readlines()

        im_idx_list = list()
        num_of_images = len(annotations)
        print("processing %d images in total" % num_of_images)
        i = 0
        for annotation in annotations:
            i += 1
            annotation = annotation.strip().split(' ')

            im_idx = annotation[0]
            landmark = list(map(float, annotation[6:]))
            landmark = np.array(landmark, dtype=np.float)

            img_test = cv2.imread(im_idx)
            img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
            vis_face_landmark_img_label(img_test, landmark, "./landmark_show/" + str(i) + ".jpg")






testor = LandmarkTestor("./data_set/face_landmark/1200.txt", "./test_result/")


# v4
testor.test_face_alignment("onet_epoch",savePic = False)  # 原数据集
testor.test_face_alignment("onet_epoch_50",savePic = False) # bbox偏移50%情况下的模型

testor.test_Onet_without_PRnet("./data_set/face_landmark/testList.txt", "Onet_result/jiang_onet_result_v4/", "onet_epoch", xxyy = True,savePic = False)  # # 原数据集
testor.test_Onet_without_PRnet("./data_set/face_landmark/testList.txt", "Onet_result/jiang_onet_result_v4/", "onet_epoch_50", xxyy = True,savePic = False)  # bbox偏移50%情况下的模型

# v9
# testor.test_face_alignment("onet_v9_epoch_50",savePic = False)  # V9 bbox偏移50%情况下的模型
# testor.test_face_alignment("onet_epoch_81", savePic = False)      # 原数据集
# testor.test_Onet_without_PRnet("./data_set/face_landmark/testList.txt", "Onet_result/jiang_onet_result_v4/", "onet_v9_epoch_50", xxyy = True,savePic = False)  # V9 bbox偏移50%情况下的模型
# testor.test_Onet_without_PRnet("./data_set/face_landmark/testList.txt", "Onet_result/jiang_onet_result_v4/", "onet_epoch_81", xxyy = True,savePic = False)  # 原数据集






# testor.test_img_and_label_48(label_path = "./data_set/train/48/landmark_large_offset.txt", store = True, store_path = "./landmark_show/")