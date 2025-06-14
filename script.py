import os
import random
import shutil
import cv2
import os

SOURCE_DIR = 'yolo_plate_dataset'     
DEST_DIR = 'dataset'        
VAL_RATIO = 0.4                     
LIMIT = 300                         

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def split_dataset():
    all_images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png'))]
    print(f'Tổng số ảnh có sẵn: {len(all_images)}')

    # Giới hạn số lượng ảnh
    if LIMIT and LIMIT < len(all_images):
        all_images = random.sample(all_images, LIMIT)
        print(f'Số lượng: {LIMIT} ảnh')

    random.shuffle(all_images)

    val_count = int(len(all_images) * VAL_RATIO)
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    print(f'Train: {len(train_images)}, Val: {len(val_images)}')

    for split in ['train', 'val']:
        create_dir(os.path.join(DEST_DIR, split, 'images'))
        create_dir(os.path.join(DEST_DIR, split, 'labels'))

    def copy_files(images, split):
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + '.txt'

            shutil.copy2(os.path.join(SOURCE_DIR, img_name),
                         os.path.join(DEST_DIR, split, 'images', img_name))

            label_path = os.path.join(SOURCE_DIR, label_name)
            if os.path.exists(label_path):
                shutil.copy2(label_path,
                             os.path.join(DEST_DIR, split, 'labels', label_name))
            else:
                open(os.path.join(DEST_DIR, split, 'labels', label_name), 'w').close()

    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    print('Chia xong.')

if __name__ == '__main__':
    split_dataset()