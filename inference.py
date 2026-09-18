"""
Inference pipeline for the product-detection dashboard.

Loads three things ONCE at process start (not per-request):
1. product_detector.pt  -> trained YOLO, finds product bounding boxes
2. mobile_sam.pt        -> MobileSAM, segments each box into a clean mask
3. vectors/*.pkl        -> DINOv2 catalog embeddings, used to identify
                           which product each isolated crop actually is
"""

from pathlib import Path
import pickle
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO, SAM

# ---- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
VECTORS_DIR = BASE_DIR / "vectors"

DETECTOR_PATH = MODELS_DIR / "product_detector.pt"
SAM_PATH = MODELS_DIR / "mobile_sam.pt"

ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]



class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        if self.device == "cuda":
            self.model.half() 
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            tensor = tensor.half()
            
        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=(self.device=='cuda')):
            vec = self.model(tensor).squeeze()
            
        vec_np = vec.cpu().numpy()
        norm = np.linalg.norm(vec_np)
        if norm > 1e-12:
            vec_np = vec_np / norm
        return vec_np


class ProductMatcher:
    """Loads vectors/*.pkl and ranks a query crop against the catalog."""
    
    def __init__(self, extractor: FeatureExtractor, vectors_dir: Path):
        self.extractor = extractor
        self.catalog = self._load_catalog(vectors_dir)

    @staticmethod
    def _load_catalog(vectors_dir: Path):
        catalog = {}
        for pkl_path in vectors_dir.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            features = data["features"]
            features = features / np.clip(
                np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None
            )
            catalog[data["item_id"]] = features
        if not catalog:
            raise FileNotFoundError(f"No *.pkl catalog vectors found in {vectors_dir}")
        return catalog

    def identify(self, crop_pil, min_similarity=0.75, min_margin=0.02):
        query_features = []
        for angle in ROTATION_ANGLES:
            rotated = crop_pil.rotate(
                angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white"
            )
            vec = self.extractor.embed(rotated)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            query_features.append(vec)
        query_features = np.array(query_features)

        scores = {}
        for item_id, ref_features in self.catalog.items():
            sims = query_features @ ref_features.T
            best_per_angle = sims.max(axis=1)
            keep = max(3, int(np.ceil(len(best_per_angle) * 0.75)))
            scores[item_id] = float(np.sort(best_per_angle)[-keep:].mean())

        ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_item, best_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        margin = best_score - second_score

        if best_score < min_similarity:
            return "unknown", best_score, scores
                
        return best_item.replace("_vector", ""), best_score, scores


class ProductPipeline:
    """Ties detector + SAM + matcher together."""
    
    def __init__(self):
        if not DETECTOR_PATH.exists():
            raise FileNotFoundError(f"Missing {DETECTOR_PATH}")
        if not SAM_PATH.exists():
            raise FileNotFoundError(f"Missing {SAM_PATH}")
            
        self.detector = YOLO(str(DETECTOR_PATH))
        self.sam = SAM(str(SAM_PATH))
        self.extractor = FeatureExtractor()
        self.matcher = ProductMatcher(self.extractor, VECTORS_DIR)
        self.device = self.extractor.device

    def predict(self, image_path, conf=0.40, iou=0.50):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img_h, img_w = image.shape[:2]

        detections = self.detector.predict(
            source=str(image_path), conf=conf, iou=iou, save=False, verbose=False
        )
        boxes = detections[0].boxes

        if len(boxes) > 0:
            keep_indices = boxes.xyxy.new_tensor(
                torch.ops.torchvision.nms(
                    boxes.xyxy.squeeze(1), 
                    boxes.conf, 
                    iou 
                )
            ).long()
            boxes = boxes[keep_indices]

        if len(boxes) == 0:
            return {"products": [], "count": 0}
            
        box_prompts = boxes.xyxy.cpu().numpy().tolist()
        
        seg_results = self.sam.predict(
            source=str(image_path), bboxes=box_prompts, device=self.device, verbose=False
        )
        masks = seg_results[0].masks.data.cpu().numpy()
        
        results = []
        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5
            isolated = np.full_like(image, 255)
            isolated[mask] = image[mask]
            
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
                
            crop = isolated[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            item_id, confidence, scores = self.matcher.identify(crop_pil)
            
            # ====== Dual Check System for Croissants (047 and 048) ======
            if item_id in ["047", "048"]:
                score_047 = scores.get("047_vector", 0.0)
                score_048 = scores.get("048_vector", 0.0)
                
                # 1. Base principle: Rely on DINOv2 Scores
                if score_048 > score_047:
                    temp_id = "048"
                    temp_conf = score_048
                else:
                    temp_id = "047"
                    temp_conf = score_047
                    
                # 2. Secondary principle: Area-based verification
                raw_area = (x2 - x1) * (y2 - y1)
                distance_factor = (img_h / max(y2, 1)) ** 2
                normalized_area = raw_area * distance_factor
                
                SIZE_THRESHOLD = 1080000
                
                if normalized_area > SIZE_THRESHOLD:
                    item_id = "047"
                    confidence = max(score_047, temp_conf)
                else:
                    item_id = "048"
                    confidence = max(score_048, temp_conf)
            
            results.append({
                "bbox": [x1, y1, x2, y2],
                "product": item_id,
                "confidence": round(confidence, 4),
            })
            
        return {"products": results, "count": len(results)}


pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ProductPipeline()
    return pipeline