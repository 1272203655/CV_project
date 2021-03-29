"""
分类模型的数据集划分 同时生成train.txt 文件
适用于根据txt文件读取 图片数据和标签
标准的图片分类 划分方式 
"""
import codecs
import os
import random
import shutil
from PIL import Image

train_ratio = 0.8
all_file_dir = r'paddle\InterpretDL\wood_defect'
class_list = [c for c in os.listdir(all_file_dir) if os.path.isdir(os.path.join(all_file_dir,c)) 
                and not c.endswith('Set') and not c.startswith('.')]
class_list.sort()
print(class_list)

train_image_dir = os.path.join(all_file_dir,"trainImageSet")
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
eval_image_dir = os.path.join(all_file_dir, "evalImageSet")
if not os.path.exists(eval_image_dir):
    os.makedirs(eval_image_dir)

train_file = codecs.open(os.path.join(all_file_dir,'train.txt'),'w')
eval_file = codecs.open(os.path.join(all_file_dir, "eval.txt"), 'w')

with codecs.open(os.path.join(all_file_dir,"label_list.txt"),'w') as label_list:
    label_id = 0
    for class_dir in class_list:
        label_list.write("{0}\t{1}\n".format(label_id,class_dir))
        image_path_pre = os.path.join(all_file_dir,class_dir)
        for file in os.listdir(image_path_pre):
            try:
                img = Image.open(os.path.join(image_path_pre,file))
                if random.uniform(0,1) <= train_ratio:
                    shutil.copyfile(os.path.join(image_path_pre,file),os.path.join(train_image_dir,file))
                    train_file.write("{0}\t{1}\n".format(os.path.join(train_image_dir,file),label_id))
                else:
                    shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(eval_image_dir, file))
                    eval_file.write("{0}\t{1}\n".format(os.path.join(eval_image_dir, file), label_id))
            except Exception as e:
                pass
        label_id += 1
train_file.close()
eval_file.close()