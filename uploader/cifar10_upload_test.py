from datasets import load_dataset
from minio import Minio
import io
from concurrent.futures import ThreadPoolExecutor

# 1. MinIO 클라이언트 연결
client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="admin12345",
    secure=False
)

bucket = "dblab-dataset"

# 버킷 없으면 생성
if not client.bucket_exists(bucket):
    client.make_bucket(bucket)
    print(f"📦 버킷 생성: {bucket}")
else:
    print(f"✅ 버킷 존재 확인: {bucket}")

# 2. CIFAR-10 데이터셋 불러오기
dataset = load_dataset("uoft-cs/cifar10")

# 3. 업로드 함수
def upload_image(args):
    split_name, idx, item = args
    img = item["img"]   # PIL.Image
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    object_name = f"uoft_cs_cifar10/{split_name}/img_{idx}.png"
    client.put_object(
        bucket,
        object_name,
        buffer,
        length=buffer.getbuffer().nbytes,
        content_type="image/png"
    )
    return idx

# 4. 병렬 업로드 실행
def upload_split_parallel(split_name, split_ds, max_workers=8):
    print(f"🚀 {split_name} 업로드 시작 (총 {len(split_ds)}장)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, _ in enumerate(executor.map(upload_image,
                                           [(split_name, idx, item) for idx, item in enumerate(split_ds)])):
            if i % 1000 == 0 and i > 0:
                print(f"[{split_name}] {i}개 업로드 완료")
    print(f"🎉 {split_name} 업로드 완료! (총 {len(split_ds)}장)")

# 5. 검증 함수
def verify_count(split_name):
    count = sum(1 for _ in client.list_objects(bucket, prefix=f"uoft_cs_cifar10/{split_name}/", recursive=True))
    print(f"✅ MinIO에 저장된 {split_name} 개수: {count}")

# 실행 (train: 50,000장, test: 10,000장)
upload_split_parallel("train", dataset["train"], max_workers=8)
upload_split_parallel("test", dataset["test"], max_workers=8)

# 업로드 검증
verify_count("train")
verify_count("test")

print("🏁 CIFAR-10 업로드 및 검증 완료!")
