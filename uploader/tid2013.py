from datasets import load_dataset
from minio import Minio
import io
import pandas as pd

# MinIO 연결
client = Minio(
    "localhost:9000",
    access_key=os.getenv("ADMIN_ID"),
    secret_key=os.getenv("ADMIN_PASSWORD"),
    secure=False
)
bucket_name = "dblab-dataset"
prefix = "tid2013"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

# 데이터 로드
ds = load_dataset("xmba15/TID2013", split="train")

# 이미지 업로드
for idx, item in enumerate(ds):
    img = item["image"]
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    object_name = f"{prefix}/images/img_{idx}.png"
    client.put_object(
        bucket_name,
        object_name,
        buffer,
        length=buffer.getbuffer().nbytes,
        content_type="image/png"
    )
    if idx % 500 == 0:
        print(f"{idx} images uploaded...")

print("✅ All images uploaded.")

# 메타데이터 추출
df = ds.to_pandas()
print("Available columns:", df.columns)

metadata = df[["score", "image width (px)"]].copy()
metadata["filename"] = [f"img_{i}.png" for i in range(len(df))]

# Parquet 형식으로 저장 & 업로드
buffer = io.BytesIO()
metadata.to_parquet(buffer, index=False)
buffer.seek(0)

client.put_object(
    bucket_name,
    f"{prefix}/metadata.parquet",
    buffer,
    length=len(buffer.getvalue()),
    content_type="application/octet-stream"
)

print("🎉 Metadata uploaded (score + image width).")
