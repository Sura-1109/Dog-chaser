import os, sys, requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

files = {
    "yolo4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

def fetch(name, url):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        print(f"[OK] {name} it already exists")
        return
    print(f"Downloading {name} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"[DONE] {name}")


if __name__ == "__main__":
    for name, url in files.items():
        try:
            fetch(name, url)
        except Exception as e:
            print(f"[ERR] {name}: {e}")
            sys.exist(1)
    print("All the model's files have been downloaded")
        