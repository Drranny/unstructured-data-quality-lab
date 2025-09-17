from minio import Minio

client = Minio(
    "localhost:9000",
    access_key=os.getenv("ADMIN_ID"),
    secret_key=os.getenv("ADMIN_PASSWORD"),
    secure=False
)

bucket = "dblab-dataset"

# train 전체 삭제
for obj in client.list_objects(bucket, prefix="uoft_cs_cifar10/train/", recursive=True):
    client.remove_object(bucket, obj.object_name)
print("🗑 train 전체 삭제 완료!")

# test 전체 삭제
for obj in client.list_objects(bucket, prefix="uoft_cs_cifar10/test/", recursive=True):
    client.remove_object(bucket, obj.object_name)
print("🗑 test 전체 삭제 완료!")