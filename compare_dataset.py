# 두개의 데이터셋을 비교하여 각 클래스 별로 데이터가 추가(또는 축소)되었는지 확인

import os

dir1 = 'resource/val'   # 확인하려는 폴더 경로 1
dir2 = 'resource/train' # 확인하려는 폴더 경로 2

def count_files_in_class_folders(root_dir):
    class_counts = {}
    if not os.path.exists(root_dir):
        print(f"{root_dir} 경로가 존재하지 않습니다.")
        return class_counts

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            file_count = len([f for f in os.listdir(class_path)
                              if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = file_count
    return class_counts

counts1 = count_files_in_class_folders(dir1)
counts2 = count_files_in_class_folders(dir2)

print(f'counts1: {counts1.__len__()} / counts2: {counts2.__len__()}')

all_classes = set(counts1.keys()).union(counts2.keys())
print(f"{'클래스':<20}{'dataset/original':<20}{'backup/dataset/original':<25}")
print('-' * 65)

print(f"총 클래스 개수: {len(all_classes)}")

num_add_class = 0

for class_name in sorted(all_classes):
    n1 = counts1.get(class_name, 0)
    n2 = counts2.get(class_name, 0)
    if(n1 != n2):
        num_add_class += 1
        print(f"{class_name:<30}{n1:<10}{n2:<10}")

print(f"추가 된 클래스 수: {num_add_class}")
