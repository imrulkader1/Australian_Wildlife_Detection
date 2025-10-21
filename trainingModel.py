# ---------------------------------------------
# Australian Animal Species Detection Training (YOLOv8n)
# ---------------------------------------------
"""
Dataset citation:
Qianqian Zhang and Khandakar Amed. Australia Animal Species Image Dataset (50). Kaggle. 2025.
DOI: 10.34740/KAGGLE/DSV/12990738
URL: https://www.kaggle.com/datasets/entenam/australia-animal-species-image-dataset-47/
License: CC BY-NC 4.0 (Attribution-NonCommercial)
"""
# -------------------------------
# 1: Import required libraries
# -------------------------------
from ultralytics import YOLO
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import shutil
import torch
torch.cuda.empty_cache()
# -------------------------------
# 2: Define full label_map (scientific name → common name)
# -------------------------------
label_map = {
    "Dromaius novaehollandiae": "Emu",
    "Macropus giganteus": "Eastern Grey Kangaroo",
    "Phascolarctos cinereus": "Koala",
    "Tursiops aduncus": "Indo-Pacific Bottlenose Dolphin",
    "Threskiornis molucca": "Australian White Ibis",
    "Gymnorhina tibicen": "Australian Magpie",
    "Platycercus elegans": "Crimson Rosella",
    "Cacatua galerita": "Sulphur-crested Cockatoo",
    "Chroicocephalus novaehollandiae": "Silver Gull",
    "Thalasseus bergii": "Crested Tern",
    "Vanellus miles": "Masked Lapwing",
    "Anas superciliosa": "Pacific Black Duck",
    "Chenonetta jubata": "Australian Wood Duck",
    "Cygnus atratus": "Black Swan",
    "Wallabia bicolor": "Swamp Wallaby",
    "Trichosurus vulpecula": "Common Brushtail Possum",
    "Pteropus poliocephalus": "Grey-headed Flying Fox",
    "Rattus fuscipes": "Bush Rat",
    "Vulpes vulpes": "Red Fox",
    "Canis lupus": "Dingo",
    "Mirounga leonina": "Southern Elephant Seal",
    "Tiliqua scincoides": "Eastern Blue-tongued Lizard",
    "Mus musculus": "House Mouse",
    "Ctenophorus nuchalis": "Central Netted Dragon",
    "Pogona barbata": "Eastern Bearded Dragon",
    "Chelonia mydas": "Green Sea Turtle",
    "Eretmochelys imbricata": "Hawksbill Sea Turtle",
    "Litoria peronii": "Peron's Tree Frog",
    "Ranoidea aurea": "Green and Golden Bell Frog",
    "Crinia signifera": "Common Eastern Froglet",
    "Carcharhinus melanopterus": "Blacktip Reef Shark",
    "Galeorhinus galeus": "School Shark",
    "Bathytoshia brevicaudata": "Smooth Stingray",
    "Dicathais orbita": "Cart-rut Shell",
    "Conus anemone": "Anemone Cone",
    "Heliocidaris erythrogramma": "Short-spined Sea Urchin",
    "Pseudonaja textilis": "Eastern Brown Snake",
    "Camelus dromedarius": "Dromedary Camel",
    "Sarcophilus harrisii": "Tasmanian Devil",
    "Ornithorhynchus anatinus": "Platypus",
    "Vombatus ursinus": "Common Wombat",
    "Dasyurus maculatus": "Spotted-tailed Quoll",
    "Dacelo novaeguineae": "Laughing Kookaburra",
    "Trichonephila edulis": "Golden Orb-weaving Spider",
    "Tachyglossus aculeatus": "Short-beaked Echidna",
    "Pastinachus ater": "Cowtail Stingray",
    "Casuarius casuarius": "Southern Cassowary",
    "Conus coronatus": "Crowned Cone",
    "Trichonephila plumipes": "Plumipes Orb-weaver",
    "Litoria fallax": "Eastern Dwarf Tree Frog"
}

# -------------------------------
# 3: Prepare Dataset YAML for YOLO training
# -------------------------------
yaml_content = """
train: dataset_yolo/train/images
val: dataset_yolo/val/images
nc: 50
names: [
  "Dromaius novaehollandiae",
  "Macropus giganteus",
  "Phascolarctos cinereus",
  "Tursiops aduncus",
  "Threskiornis molucca",
  "Gymnorhina tibicen",
  "Platycercus elegans",
  "Cacatua galerita",
  "Chroicocephalus novaehollandiae",
  "Thalasseus bergii",
  "Vanellus miles",
  "Anas superciliosa",
  "Chenonetta jubata",
  "Cygnus atratus",
  "Wallabia bicolor",
  "Trichosurus vulpecula",
  "Pteropus poliocephalus",
  "Rattus fuscipes",
  "Vulpes vulpes",
  "Canis lupus",
  "Mirounga leonina",
  "Tiliqua scincoides",
  "Mus musculus",
  "Ctenophorus nuchalis",
  "Pogona barbata",
  "Chelonia mydas",
  "Eretmochelys imbricata",
  "Litoria peronii",
  "Ranoidea aurea",
  "Crinia signifera",
  "Carcharhinus melanopterus",
  "Galeorhinus galeus",
  "Bathytoshia brevicaudata",
  "Dicathais orbita",
  "Conus anemone",
  "Heliocidaris erythrogramma",
  "Pseudonaja textilis",
  "Camelus dromedarius",
  "Sarcophilus harrisii",
  "Ornithorhynchus anatinus",
  "Vombatus ursinus",
  "Dasyurus maculatus",
  "Dacelo novaeguineae",
  "Trichonephila edulis",
  "Tachyglossus aculeatus",
  "Pastinachus ater",
  "Casuarius casuarius",
  "Conus coronatus",
  "Trichonephila plumipes",
  "Litoria fallax"
]
"""
with open("kaggle_animals.yaml", "w") as f:
    f.write(yaml_content)

# -------------------------------
# 4: Train YOLOv8n model on your dataset
# -------------------------------
def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="kaggle_animals.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4
    )

# -------------------------------
# 5: Validate trained model (shows performance metrics)
# -------------------------------
    results = model.val()
    print("Validation Results:", results)

# -------------------------------
# 6: Save trained model weights for later use
# -------------------------------
    shutil.copy("runs/detect/train/weights/best.pt", "trained_kaggle_yolo.pt")

# -------------------------------
# End of Pipeline
# -------------------------------
if __name__ == '__main__':
    main()