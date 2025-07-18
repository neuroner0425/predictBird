import os
import random
import shutil
import sys
from PIL import Image
import matplotlib.pyplot as plt

outlier_dir = 'resources/flagged_outlier'
root_dir = 'resources/only_crawl'

outlier_files = [f for f in os.listdir(outlier_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def get_classname(filename):
    return filename.split('_')[0]

def get_class_sample_images(class_name, exclude_file=None, num_samples=3):
    class_path = os.path.join(root_dir, class_name)
    all_files = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f != exclude_file]
    if len(all_files) < num_samples:
        samples = all_files
    else:
        samples = random.sample(all_files, num_samples)
    return [os.path.join(class_path, f) for f in samples]

print("'-' : 삭제 | '=' : 복구 | '`' : 프로그램 종료 | 기타 키 : 건너뜀 (창에서 바로 키 입력, 항상 전체화면)\n")

for file in outlier_files:
    class_name = get_classname(file)
    outlier_path = os.path.join(outlier_dir, file)
    if not os.path.isdir(os.path.join(root_dir, class_name)):
        print(f"폴더 없음: {class_name}")
        continue
    left_imgs = get_class_sample_images(class_name, num_samples=3)

    user_action = {"key": None}

    def on_key(event):
        if event.key == '-':
            user_action["key"] = '-'
            plt.close()
        elif event.key == '=':
            user_action["key"] = '='
            plt.close()
        else:
            user_action["key"] = 'skip'
            plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    for idx, img_path in enumerate(left_imgs):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f'Sample {idx+1}')
        axes[idx].axis('off')
    img = Image.open(outlier_path)
    axes[3].imshow(img)
    axes[3].set_title('Flagged\nOutlier')
    axes[3].axis('off')
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', on_key)

    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()
    except AttributeError:
        try:
            manager.window.state('zoomed')
        except Exception:
            pass

    plt.show()   # 키 누르면 plt.close()됨

    key = user_action["key"]
    if key == '-':
        os.remove(outlier_path)
        print(f"{file} 삭제됨.\n")
    elif key == '=':
        dst_path = os.path.join(root_dir, class_name, file.replace(f"{class_name}_", ""))
        shutil.move(outlier_path, dst_path)
        print(f"{file} 복구됨. (기존 폴더로 이동)\n")
    else:
        print("프로그램을 종료합니다.")
        sys.exit(0)
