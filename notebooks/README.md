# First EDA

I downloaded `20260417@1113 - Hortifrut - Almonte` (UID:c834a7b7-4bbd-4791-98c0-9d2d8123d60a) dataset... 

- all_images :  not yet utilized images.. (can ignore for now)
- images/ : this is used for training
- labels/ :
- annotations/ : unzip YOLO.zip ()


## DETAILS

Classes: 
  - Bud
  - Flower 
  - Green
  - pink
  - blue

Each Dataset: ~ 300 images
  - Some background images in there too..

20260225@1359 - Hortifrut - Almonte


## EXCLUDE

Those datasets where annotated in champia and/or had different class seperation
- 20250610@1035 - Berry Plasma World - Varda
- 20251124@1725 - DataCrop - Loncoche

Old class names:
- 20250610@1035 - Berry Plasma World - Varda
  - UUID: 74519691-0c10-4cc5-9f2c-0e74bac63330
- 20251124@1725 - DataCrop - Loncoche 
  - UUID: c2450642-ea8f-448f-8066-c0610b192ef0
- 20250806@1012 - Sergio Panini - Tarantasca (blueberry class)
  - UUID: 92fbc3d1-824f-41d0-80d6-3e840b130a30

20260313@0700 - Hortifrut - Almonte


## DATASET HANDLING

- 20260409@0959-Hortitool-ChãsdeTavares(in Google sheet) or dataset-f2ac3a5a (in Dashboard) (UUID: f2ac3a5a-bc56-492e-8d9a-1a430d8fe53d)
  - it has 266/301 images with no correpsonding label.
  - All the images with no labels don't correspond to background -> THEY ARE JUST NOT ANNOTATED!! but contain fruits...
  - So don't use them as background in parsing..

- 20260410@0849 - Hortifrut - Almonte (in Google sheet) or dataset-c967718b (Dashboard) (UUID: c967718b-2031-49ca-aa14-080210cc3fc6)
  - 10/301 images with no corresponding label
  - parsing it through ML dashboard I see a lot of images which are just not annotated.. Some are actually background..

- (in google sheet) or dataset-c834a7b7 (in dashboard) (UUID: c834a7b7-4bbd-4791-98c0-9d2d8123d60a)
  - 144/445 images have no corresponding label..


- dataset-c967718b (UUID: c967718b-2031-49ca-aa14-080210cc3fc6)

To obj.names exei : flower, green, pink, blue. Alla sta .txt iparxei kai index 4. 
Mallon den iparxei i klasei 1 (bud) pouthena kai apla ta indexes einai 0, 2, 3, 4
