import os
import shutil

src_root = 'resource'
dst_root = 'resource/original'

# train, val 폴더 각각 순회
for split in ['train', 'val']:
    split_path = os.path.join(src_root, split)
    if not os.path.exists(split_path):
        continue  # 폴더가 없으면 건너뜀

    # 각 클래스 폴더 순회
    for class_name in os.listdir(split_path):
        class_src = os.path.join(split_path, class_name)
        class_dst = os.path.join(dst_root, class_name)
        if not os.path.isdir(class_src):
            continue  # 폴더가 아니면 건너뜀

        os.makedirs(class_dst, exist_ok=True)  # 대상 폴더 없으면 생성

        # 파일 하나씩 복사
        for file_name in os.listdir(class_src):
            src_file = os.path.join(class_src, file_name)
            dst_file = os.path.join(class_dst, file_name)
            # 이름이 겹칠 수 있으니, 겹칠 때는 파일명에 prefix 추가 (선택)
            if os.path.exists(dst_file):
                base, ext = os.path.splitext(file_name)
                new_file_name = f"{split}_{base}{ext}"
                dst_file = os.path.join(class_dst, new_file_name)
            shutil.copy2(src_file, dst_file)

print("복사가 완료되었습니다.")
