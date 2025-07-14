from icrawler.builtin import BingImageCrawler
import os
from concurrent.futures import ThreadPoolExecutor

# 1. 부족한 클래스 확인
root_dir = 'resource/crawl'
min_count = 200
few_data_classes = []

for class_name in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_name)
    if os.path.isdir(class_path):
        num_files = len([f for f in os.listdir(class_path)
                         if os.path.isfile(os.path.join(class_path, f))])
        if num_files < min_count:
            few_data_classes.append((class_name, num_files))

print(f"이미지 개수가 {min_count}개 미만인 클래스:")
for name, cnt in few_data_classes:
    print(f"{name}: {cnt}개")

# 2. 병렬 크롤링 함수
def crawl_images_for_class(args):
    class_name, current_count = args
    save_path = os.path.join(root_dir, class_name)
    need = min_count - current_count
    if need <= 0:
        return f"{class_name}: 필요 없음"
    print(f"{class_name}: {need}장 추가 크롤링 중...")
    crawler = BingImageCrawler(storage={'root_dir': save_path})
    crawler.crawl(keyword=class_name, max_num=need)
    return f"{class_name}: {need}장 크롤링 완료"

# 3. 병렬 실행
with ThreadPoolExecutor(max_workers=96) as executor:  # worker 개수 조정 가능
    results = list(executor.map(crawl_images_for_class, few_data_classes))

print("\n=== 결과 ===")
for r in results:
    print(r)
