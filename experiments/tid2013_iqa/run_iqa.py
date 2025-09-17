import os
import pandas as pd
from skimage import io
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from imquality import brisque
from image_quality import niqe, piqe
from tqdm import tqdm

# TID2013 데이터 경로 (원본/왜곡 구조라 가정)
REF_DIR = "../../data/dblab-dataset/tid2013/reference"
DIST_DIR = "../../data/dblab-dataset/tid2013/distorted"

results = []

for f in tqdm(os.listdir(DIST_DIR)[:100]):  # 우선 100개만 테스트
    ref_path = os.path.join(REF_DIR, f)       # 파일명 동일하다고 가정
    dist_path = os.path.join(DIST_DIR, f)

    if not os.path.exists(ref_path):
        continue

    ref_img = io.imread(ref_path)
    dist_img = io.imread(dist_path)

    # FR-IQA
    ssim_val = ssim(ref_img, dist_img, channel_axis=-1)
    psnr_val = psnr(ref_img, dist_img)

    # NR-IQA
    brisque_val = brisque.score(dist_img)
    niqe_val = niqe(dist_img)
    piqe_val = piqe(dist_img)

    results.append({
        "file": f,
        "SSIM": ssim_val,
        "PSNR": psnr_val,
        "BRISQUE": brisque_val,
        "NIQE": niqe_val,
        "PIQE": piqe_val,
    })

df = pd.DataFrame(results)
df.to_csv("tid2013_iqa_results.csv", index=False)
print(" TID2013 결과 저장 완료: tid2013_iqa_results.csv")
