import pickle
from pathlib import Path

vectors_dir = Path("vectors")
pkl_files = list(vectors_dir.glob("*.pkl"))

if not pkl_files:
    print("❌ No .pkl files found in the 'vectors' directory.")
else:
    print(f"✅ Found {len(pkl_files)} .pkl file(s). Inspecting them...\n")
    
    # Check up to 3 files to confirm consistency
    for pkl_path in pkl_files[:3]:
        print(f"📁 File: {pkl_path.name}")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            features = data.get("features")
            if features is not None:
                shape = features.shape
                print(f"   • Item ID: {data.get('item_id')}")
                print(f"   • Samples: {shape[0]}")
                print(f"   • Embedding Dimension: {shape[1]}")
                
                if shape[1] == 384:
                    print("   ✅ MATCH: This is a DINOv2 (ViT-S/14) embedding. You are good to go!")
                elif shape[1] == 2048:
                    print("   ❌ MISMATCH: This is a ResNet50 embedding. You MUST re-upload your catalog images.")
                elif shape[1] == 768:
                    print("   ⚠️ WARNING: This is DINOv2 (ViT-B/14). Update your code to use 'dinov2_vitb14' instead of 'vits14'.")
                else:
                    print(f"   ⚠️ UNKNOWN: Embedding dimension is {shape[1]}. Check your model configuration.")
            else:
                print("   ❌ ERROR: 'features' key not found in the pickle file.")
        except Exception as e:
            print(f"   ❌ ERROR reading file: {e}")
        print("-" * 60)