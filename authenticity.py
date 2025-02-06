import torch
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from numpy import uint8, exp

authenticity_models = [
    "Nahrawy/AIorNot",
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
    "otaku840726/autotrain-cupfg-ml9bd",
    "otaku840726/autotrain-realfake-swin-20"
]

def get_authenticity_extractor(model_id):
    return AutoFeatureExtractor.from_pretrained(model_id)

def get_authenticity_classification(model_id):
    return AutoModelForImageClassification.from_pretrained(model_id)

def softmax(vector):
    e = exp(vector - vector.max())
    return e / e.sum()

def run_authenticity(image, model_id=None):
    labels = ["Fake", "Real"]
    feature_extractor = get_authenticity_extractor(model_id)
    classofocation = get_authenticity_classification(model_id)
    input = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = classofocation(**input)
        logits = outputs.logits
        probability = softmax(logits)
        px = pd.DataFrame(probability.numpy())
    prediction = logits.argmax(-1).item()
    label = labels[prediction]
    results = {labels[idx]: float(px[idx][0]) for idx in range(len(px))}
    return results
