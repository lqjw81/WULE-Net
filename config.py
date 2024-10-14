from os.path import join

class FLAGES(object):


    lr = 4e-4
    # lr = 1e-3
    decay_rate = 0.99
    decay_step = 10000
    num_workers = 4
    # train_batch_size = 8
    train_batch_size = 4
    test_batch_size = 1
    val_batch_size = 1

    # 集美数据集路径
    trainA_path = r"/home/tr708/data/data3/zlq/lowlight2024/myenlight-542-24.1309-0.8798/datasets/train/low/"  #raw
    trainB_path = r"/home/tr708/data/data3/zlq/lowlight2024/myenlight-542-24.1309-0.8798/datasets/train/high/"  #reference

    valA_path = r"/home/tr708/data/data3/zlq/lowlight2024/myenlight-542-24.1309-0.8798/datasets/val/low/"  #raw
    valB_path = r"/home/tr708/data/data3/zlq/lowlight2024/myenlight-542-24.1309-0.8798/datasets/val/high/"  #reference
    testA_path = r"/home/tr708/data/data3/zlq/lowlight2024/myenlight-542-24.1309-0.8798/datasets/val/low/"  #raw

    # EUVP数据集路径
    # trainA_path = r"D:/Desk/PycharmProject/ModifiedByMyself/Myfive/dataset_euvp/train_data/input/"  #raw
    # trainB_path = r"D:/Desk/PycharmProject/ModifiedByMyself/Myfive/dataset_euvp/train_data/target/"  #reference
    #
    # valA_path = r"D:/Desk/PycharmProject/ModifiedByMyself/Myfive/dataset_euvp/val_data/input/"  #raw
    # valB_path = r"D:/Desk/PycharmProject/ModifiedByMyself/Myfive/dataset_euvp/val_data/target/"  #reference

    # testA_path = r"D:/Desk/PycharmProject/ModifiedByMyself/Myfive/dataset_jimei/val_data/input/"  #raw
    #
    #
    out_path = r"/home/tr708/data/data3/zlq/lowlight2024/myenlight-542-24.1309-0.8798/results/output_test/"  #output

    model_dir = r'/home/tr708/data/data3/zlq/lowlight2024/myenlight-542-24.1309-0.8798/results/model_save/'

    backup_model_dir = join(model_dir)
