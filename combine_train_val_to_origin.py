# train 데이터셋과 val 데이터셋을 하나의 original 데이터셋으로 병합

import os
import shutil

src_root = 'resources'
dst_root = 'resources/original'

for split in ['train', 'val']:
    split_path = os.path.join(src_root, split)
    if not os.path.exists(split_path):
        continue

    for class_name in os.listdir(split_path):
        class_src = os.path.join(split_path, class_name)
        class_dst = os.path.join(dst_root, class_name)
        if not os.path.isdir(class_src):
            continue

        os.makedirs(class_dst, exist_ok=True)

        for file_name in os.listdir(class_src):
            src_file = os.path.join(class_src, file_name)
            dst_file = os.path.join(class_dst, file_name)
            if os.path.exists(dst_file):
                base, ext = os.path.splitext(file_name)
                new_file_name = f"{split}_{base}{ext}"
                dst_file = os.path.join(class_dst, new_file_name)
            shutil.copy2(src_file, dst_file)

print("복사가 완료되었습니다.")
