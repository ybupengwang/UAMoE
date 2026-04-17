class Config():
    img_dir = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\TrainingData'
    gt_dir = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Training'
    test_img_dir1 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test1Data'
    test_gt_dir1 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test1'
    test_img_dir2 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test2Data'
    test_gt_dir2 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test2'
    GPU = 0
    optimizer = 'adam'
    base_number = 40
    resize_h = 512
    resize_w = 512
    sigma = 10
    point_num = 19
    num_epochs = 300
    lr = 1e-3
    trans = True
    struct_biaozhi = True
    save_model_path = ''
    save_results_path = ''
    handimg_dir = r"D:\dataset\yachi\shou-biaozhu\set1\trainimage"
    handgt_dir = r"D:\dataset\yachi\shou-biaozhu\set1\traindata"
    testhandimg_dir = r"D:\dataset\yachi\shou-biaozhu\set1\testimage"
    testhandgt_dir = r"D:\dataset\yachi\shou-biaozhu\set1\testdata"
    handresize_h = 512
    handresize_w = 512



