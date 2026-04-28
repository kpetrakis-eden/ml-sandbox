from ultralytics import YOLO

def main():
  print("Hello from ml-sandbox!")

  # model = YOLO("yolo11n.pt")
  model = YOLO("yolo26n.pt")
  # device=0 used by Xorg
  # results = model.train(data="coco8.yaml", epochs=3, device=1)
  results = model.train(data="data/data.yaml", epochs=30, device=1, val=True,
                        cls_pw=1.0, name="cls_pw")
  
if __name__ == "__main__":
  main()
