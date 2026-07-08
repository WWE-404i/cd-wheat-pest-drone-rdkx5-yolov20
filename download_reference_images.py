"""
从 Wikimedia Commons 下载小麦病变参考图片
"""
import requests
import json
from pathlib import Path
from PIL import Image
import io

OUT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\reference_crops")
OUT.mkdir(exist_ok=True)

# 四个病变的 Wikimedia 搜索词
QUERIES = {
    "Brown_Rust": "Puccinia triticina wheat leaf rust symptoms",
    "Yellow_Rust": "Puccinia striiformis wheat stripe rust symptoms",
    "Black_Rust": "Puccinia graminis wheat stem rust symptoms",
    "Septoria": "Zymoseptoria tritici wheat septoria blotch symptoms",
}

HEADERS = {"User-Agent": "WheatDiseaseRef/1.0"}

for name, query in QUERIES.items():
    print(f"\n搜索: {name} ({query})")

    # Wikimedia Commons API 搜索
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": "6",  # File namespace
        "srlimit": "10",
        "srprop": "snippet|titlesnippet",
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        data = resp.json()
    except Exception as e:
        print(f"  搜索失败: {e}")
        continue

    pages = data.get("query", {}).get("search", [])
    print(f"  找到 {len(pages)} 个结果")

    for i, page in enumerate(pages[:5]):
        title = page["title"]
        # 获取图片URL
        img_params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": "400",
        }
        try:
            img_resp = requests.get(url, params=img_params, headers=HEADERS, timeout=15)
            img_data = img_resp.json()
        except Exception as e:
            print(f"    {title}: 获取失败 {e}")
            continue

        pages_data = img_data.get("query", {}).get("pages", {})
        for page_id, page_info in pages_data.items():
            if page_id == "-1":
                continue
            imageinfo = page_info.get("imageinfo", [])
            if not imageinfo:
                continue
            thumb_url = imageinfo[0].get("thumburl")
            if not thumb_url:
                continue
            # 判断是否靠谱（排除显微镜图、图表等）
            mime = imageinfo[0].get("mime", "")
            size = imageinfo[0].get("size", 0)
            if mime not in ("image/jpeg", "image/png"):
                continue
            if size < 10000:  # 太小
                continue

            # 下载
            try:
                img_resp2 = requests.get(thumb_url, headers=HEADERS, timeout=30)
                img = Image.open(io.BytesIO(img_resp2.content))
                img.save(OUT / f"{name}_{i+1:02d}.jpg")
                print(f"    下载成功: {title} ({img.size})")
            except Exception as e:
                print(f"    下载失败: {title}: {e}")

print(f"\n完成 → {OUT}")
