from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import face_recognition
import time
import cv2
import numpy as np
import logging

# Suppress transformers logging to avoid CLIP model warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Suppress Hugging Face Hub warnings and progress bars
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_WARNINGS'] = '1'

import warnings

# Suppress all Hugging Face warnings
warnings.filterwarnings("ignore", message=".*Hugging Face Hub.*")
warnings.filterwarnings("ignore", message=".*token.*")
warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests.*")

# Load CLIP model
print("Loading CLIP model...")
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
print("CLIP model loaded successfully!")

# Define the folder path
folder_path = "/Users/ofloericke/images"

def crop_to_face(image_path):
    """Crop image to face region using faster OpenCV face detection"""
    start_time = time.time()
    load_start = time.time()
    
    try:
        print(f"Processing: {image_path} - using OpenCV face detection")
        
        # Load image
        pil_image = Image.open(image_path)
        load_time = time.time() - load_start
        print(f"  Image loaded in {load_time:.3f} seconds")
        
        # Resize if too large
        resize_start = time.time()
        max_size = 800
        if max(pil_image.size) > max_size:
            resize_ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * resize_ratio), int(pil_image.height * resize_ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
            resize_time = time.time() - resize_start
            print(f"  Resized to {new_size} in {resize_time:.3f} seconds")
        else:
            resize_time = time.time() - resize_start
            print(f"  No resizing needed ({resize_time:.3f} seconds)")
        
        # Convert to OpenCV format
        convert_start = time.time()
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        convert_time = time.time() - convert_start
        print(f"  Image converted in {convert_time:.3f} seconds")
        
        # Load OpenCV's pre-trained face detection model
        try:
            # Try to use GPU (Metal) acceleration on M1 Mac
            net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt',
                'res10_300x300_ssd_iter_140000.caffemodel'
            )
            
            # Set preferable backend and target for M1 GPU
            try:
                # Try Metal backend first for M1 Mac
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Metal acceleration
                print("  Using OpenCV with M1 GPU acceleration (Metal)")
            except Exception as e:
                print(f"  Could not configure GPU: {e}")
                print("  Using OpenCV with CPU only")
                
        except Exception as e:
            print(f"  Could not load OpenCV models: {e}")
            print("  Falling back to face_recognition library")
            return crop_to_face_fallback(image_path)
        
        # Prepare image for detection
        face_start = time.time()
        
        # Optimize blob creation for M1
        blob = cv2.dnn.blobFromImage(
            image_cv, 
            scalefactor=1.0, 
            size=(300, 300), 
            mean=[104, 117, 123],
            swapRB=False,
            crop=False
        )
        net.setInput(blob)
        
        # Run face detection with warmup (helps on M1)
        # First run might be slower, but subsequent runs will be faster
        detections = net.forward()
        face_detect_time = time.time() - face_start
        print(f"  OpenCV face detection completed in {face_detect_time:.3f} seconds")
        
        # Add note about expected performance
        if face_detect_time > 0.5:
            print("  Note: First run may be slower. Subsequent runs should be faster.")

        # Process detections
        if len(detections) == 0 or detections.shape[2] == 0:
            print(f"  No faces found in {image_path}, using full image")
            return pil_image

        # Get the detection with highest confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:  # Confidence threshold
            print(f"  Low confidence ({confidence:.2f}) face detection, using full image")
            return pil_image

        # Extract face coordinates
        box = detections[0, 0, i, 3:7] * np.array([image_cv.shape[1], image_cv.shape[0],
                                               image_cv.shape[1], image_cv.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")

        # Add padding
        padding = 20
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(image_cv.shape[1], endX + padding)
        endY = min(image_cv.shape[0], endY + padding)

        # Crop and return
        crop_start = time.time()
        cropped = pil_image.crop((startX, startY, endX, endY))
        crop_time = time.time() - crop_start
        
        total_face_time = time.time() - start_time
        print(f"  Face cropping completed in {crop_time:.3f} seconds")
        print(f"Total face processing time: {total_face_time:.2f} seconds")
        
        return cropped, {
            'load_time': load_time,
            'resize_time': resize_time,
            'convert_time': convert_time,
            'face_detect_time': face_detect_time,
            'crop_time': crop_time,
            'total_face_time': total_face_time
        }

    except Exception as e:
        print(f"OpenCV face detection failed for {image_path}: {e}")
        print("Falling back to face_recognition library")
        return crop_to_face_fallback(image_path)

def crop_to_face_fallback(image_path):
    """Fallback to original face_recognition library"""
    start_time = time.time()
    load_start = time.time()
    try:
        print(f"Processing: {image_path} - using face_recognition fallback")
        
        # Load image and resize if too large
        pil_image = Image.open(image_path)
        load_time = time.time() - load_start
        print(f"  Image loaded in {load_time:.3f} seconds")
        
        # Resize large images for faster face detection
        resize_start = time.time()
        max_size = 800
        if max(pil_image.size) > max_size:
            resize_ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * resize_ratio), int(pil_image.height * resize_ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
            resize_time = time.time() - resize_start
            print(f"  Resized image to {new_size} in {resize_time:.3f} seconds")
        else:
            resize_time = time.time() - resize_start
            print(f"  No resizing needed ({resize_time:.3f} seconds)")
        
        # Convert to numpy array for face_recognition
        convert_start = time.time()
        image = face_recognition.load_image_file(image_path)
        convert_time = time.time() - convert_start
        print(f"  Image converted in {convert_time:.3f} seconds")
        
        # Use HOG model with reduced upsampling for maximum speed
        face_start = time.time()
        face_locations = face_recognition.face_locations(
            image, 
            number_of_times_to_upsample=0,
            model="hog"
        )
        face_detect_time = time.time() - face_start
        print(f"  Face detection completed in {face_detect_time:.3f} seconds")

        if not face_locations:
            print(f"No faces found in {image_path}, using full image")
            return pil_image

        # Get the first face found
        top, right, bottom, left = face_locations[0]

        # Add minimal padding around the face
        padding = 10
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(image.shape[0], bottom + padding)
        right = min(image.shape[1], right + padding)

        # Crop and return
        crop_start = time.time()
        cropped = pil_image.crop((left, top, right, bottom))
        crop_time = time.time() - crop_start
        
        total_face_time = time.time() - start_time
        print(f"  Face cropping completed in {crop_time:.3f} seconds")
        print(f"Total face processing time: {total_face_time:.2f} seconds")
        
        return cropped, {
            'load_time': load_time,
            'resize_time': resize_time,
            'convert_time': convert_time,
            'face_detect_time': face_detect_time,
            'crop_time': crop_time,
            'total_face_time': total_face_time
        }

    except Exception as e:
        print(f"Face detection failed for {image_path}: {e}")
        print("Using full image instead")
        return Image.open(image_path)

# Enhanced text prompts for better discrimination
text_prompts = [
    "A person with closed eyes",
    "A person with eyes wide open",
    "A face with eyelids completely covering the eyes",
    "Eyes that are shut",
    "A person blinking"
]

# Encode all text prompts once
text_embeddings = model.encode(text_prompts)

# Cache for face detection results to avoid reprocessing
face_cache = {}

if not os.path.exists(folder_path):
    print(f"Folder {folder_path} does not exist.")
else:
    print(f"Starting analysis of images in {folder_path}")
    print("=" * 60)
    
    # Statistics tracking
    total_images = 0
    total_face_time = 0.0
    total_load_time = 0.0
    total_resize_time = 0.0
    total_convert_time = 0.0
    total_face_detect_time = 0.0
    total_crop_time = 0.0
    total_processing_time = 0.0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            total_images += 1
            image_start_time = time.time()
            
            try:
                print(f"\nAnalyzing: {filename}")
                
                # Use cached face image if available
                if image_path in face_cache:
                    face_image = face_cache[image_path]
                    print("Using cached face image")
                else:
                    # Call crop_to_face and capture detailed timing
                    result = crop_to_face(image_path)
                    if isinstance(result, tuple):
                        face_image, timing_data = result
                        # Update statistics
                        total_load_time += timing_data['load_time']
                        total_resize_time += timing_data['resize_time']
                        total_convert_time += timing_data['convert_time']
                        total_face_detect_time += timing_data['face_detect_time']
                        total_crop_time += timing_data['crop_time']
                        total_face_time += timing_data['total_face_time']
                    else:
                        face_image = result
                    face_cache[image_path] = face_image

                # Encode the cropped image with timeout
                print("Encoding image with CLIP...")
                clip_start = time.time()
                img_emb = model.encode(face_image)
                clip_time = time.time() - clip_start
                print(f"CLIP encoding completed in {clip_time:.2f} seconds")

                # Compute similarities against all prompts
                print("Calculating similarities...")
                sim_start = time.time()
                similarity_scores = model.similarity(img_emb, text_embeddings)
                sim_time = time.time() - sim_start
                print(f"Similarity calculation completed in {sim_time:.2f} seconds")

                # Extract individual scores from the tensor
                closed_eyes_score = similarity_scores[0][0].item()
                open_eyes_score = similarity_scores[0][1].item()
                specific_closed_score = similarity_scores[0][2].item()

                # Calculate confidence metrics
                raw_closed_score = closed_eyes_score
                relative_confidence = closed_eyes_score - open_eyes_score
                combined_score = (closed_eyes_score + specific_closed_score) / 2 - open_eyes_score

                # Lower threshold for better sensitivity
                is_closed = relative_confidence > 0.02  # Changed from 0.05
                prediction = "closed" if is_closed else "open"

                image_time = time.time() - image_start_time
                total_processing_time += image_time
                print(f"Total processing time for {filename}: {image_time:.2f} seconds")
                
                print(f"Results for {filename}:")
                print(f"  Closed eyes score: {raw_closed_score:.4f}")
                print(f"  Open eyes score: {open_eyes_score:.4f}")
                print(f"  Relative confidence: {relative_confidence:.4f}")
                print(f"  Combined score: {combined_score:.4f}")
                print(f"  Prediction: Eyes likely {prediction}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                print("Continuing with next image...")
                continue
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY:")
    print(f"Total images processed: {total_images}")
    if total_images > 0:
        print(f"\nDETAILED FACE PROCESSING TIMES:")
        print(f"  Average load time: {total_load_time/total_images:.3f} seconds")
        print(f"  Average resize time: {total_resize_time/total_images:.3f} seconds")
        print(f"  Average convert time: {total_convert_time/total_images:.3f} seconds")
        print(f"  Average face detection time: {total_face_detect_time/total_images:.3f} seconds")
        print(f"  Average crop time: {total_crop_time/total_images:.3f} seconds")
        print(f"  Average total face processing time: {total_face_time/total_images:.2f} seconds")
        
        print(f"\nOVERALL PROCESSING:")
        print(f"  Average total processing time per image: {total_processing_time/total_images:.2f} seconds")
        print(f"  Total processing time: {total_processing_time:.2f} seconds")
    print("Analysis complete!")