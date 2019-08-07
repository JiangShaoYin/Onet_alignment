import cv2,os
import numpy as np
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face,vis_face_test, vis_face_raw, vis_face_landmark_img_label

annotation_file = "./data_set/face_landmark/testImageList.txt"




def test(inoutDir,outputDir,model):  # 原模型的P，R，net + 自行训练后的Onet，展示并保存检测后的图片
    pnet, rnet, onet_jiang = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                              r_model_path="./original_model/rnet_epoch.pt",
                                              o_model_path="./original_model/" + model + ".pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet_jiang, min_face_size=24)
    files = os.listdir(inoutDir)
    i = 0
    for image in files:
        i += 1
        image = os.path.join("./lfpw_test/", image)

        img = cv2.imread(image)
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxs, landmarks1 = mtcnn_detector.detect_face(img)  # 原始的图片用原始网络检测

        vis_face(img_bg, bboxs, landmarks1, outputDir + model + "-" + str(i) + ".jpg") # 保存图片

def test_onet(inoutDir,outputDir,model):
    pnet, rnet, onet_jiang = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                              r_model_path="./original_model/rnet_epoch.pt",
                                              o_model_path="./original_model/" + model + ".pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet_jiang, min_face_size=24)

    files = os.listdir(inoutDir)
    i = 0
    for image in files:
        i += 1
        image = os.path.join(inoutDir, image)

        img = cv2.imread(image)
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        landmarks2_jiang = mtcnn_detector.detect_onet_raw(img)

        vis_face_test(img_bg, landmarks2_jiang,  outputDir + model + "-" + str(i) + ".jpg")
        if i == 50:
            break


if __name__ == '__main__':
    # 检测P R O net
    # test("./lfpw_test/", "Onet_result/jiang_onet_result_v4/", "onet_epoch_99")
    # test("./lfpw_test/", "Onet_result/jiang_onet_result_v4/", "onet_epoch_46")
    # test("./lfpw_test/", "Onet_result/jiang_onet_result_v4/", "onet_epoch_20")
    # test("./lfpw_test/", "Onet_result/jiang_onet_result_v4/", "onet_epoch_10")
    # test("./lfpw_test/", "Onet_result/jiang_onet_result_v4/", "onet_epoch_50")
    # test("./lfpw_test/", "Onet_result/origin_onet_result_v4/", "onet_epoch")
    # test("./lfpw_test/", "Onet_result/jiang_onet_result_v9/-PROnet-", "onet_epoch_81")

    # 检测Onet
    ## 测试集在4层30epoch结果
    # test_onet("./Onet_result/Onet_input/", "Onet_result/jiang_onet_result_v4/", "onet_epoch_50")
    # test_onet("./Onet_result/Onet_input/", "Onet_result/origin_onet_result_v4/", "onet_epoch")

    # 测试集在9层81epoch结果
    test_onet("./Onet_result/Onet_input/", "Onet_result/jiang_onet_result_v9/test-", "onet_epoch_81")
    ## 训练集在4层30epoch结果
    # test_onet("./data_set/train/48/train_ONet_landmark_aug/", "Onet_result/jiang_onet_result_v4/train-", "onet_epoch_30")
    # test_onet("./data_set/train/48/train_ONet_landmark_aug/", "Onet_result/origin_onet_result_v4/train-", "onet_epoch")
    # test_onet("./data_set/train/48/train_ONet_landmark_aug/", "Onet_result/jiang_onet_result_v4/train-", "onet_epoch_30")
    ## 测试集在6层100epoch结果
    # test_onet("./Onet_result/Onet_input/", "Onet_result/jiang_onet_result_v6/", "onet_epoch_100")










    #测试原始的图片landmark信息
    # with open("./data_set/train/48/landmark_large_offset.txt", 'r') as f:
    #     annotations = f.readlines()
    #
    # im_idx_list = list()
    # num_of_images = len(annotations)
    # print("processing %d images in total" % num_of_images)
    # i = 0
    # for annotation in annotations:
    #     i += 1
    #     annotation = annotation.strip().split(' ')
    #
    #     im_idx = annotation[0]
    #     landmark = list(map(float, annotation[6:]))
    #     landmark = np.array(landmark, dtype=np.float)
    #
    #     img_test = cv2.imread(im_idx)
    #     img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    #     vis_face_landmark_img_label(img_test, landmark, "./landmark_show/" + str(i)+".jpg")
