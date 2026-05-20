import argparse
import io
import os
from urllib.parse import urlparse
import numpy as np
import requests
import tensorflow as tf
from PIL import Image

# MODEL CONFIGURATION (EfficientNetB0, 224x224)

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence required for plant detection

# Model file paths
MODEL_DETECTION_PATH = 'plant_detection_model_224_final.h5'
MODEL_SPECIES_PATH = 'species_classification_model_224_final.h5'
MODEL_ROSE_HEALTH_PATH = 'rose_health_classification_model_224_final.h5'
MODEL_POTHOS_HEALTH_PATH = 'pothos_classification_model_224_final.h5'

# Class label definitions
CLASS_DETECTION = ["leaf", "no_leaf"]

CLASS_SPECIES = [
    'African Violet (Saintpaulia ionantha)', 'Aloe Vera', 'Anthurium (Anthurium andraeanum)', 
    'Areca Palm (Dypsis lutescens)', 'Asparagus Fern (Asparagus setaceus)', 'Begonia (Begonia spp.)', 
    'Bird of Paradise (Strelitzia reginae)', 'Birds Nest Fern (Asplenium nidus)', 
    'Boston Fern (Nephrolepis exaltata)', 'Calathea', 'Cast Iron Plant (Aspidistra elatior)', 
    'Chinese Money Plant (Pilea peperomioides)', 'Chinese evergreen (Aglaonema)', 
    'Christmas Cactus (Schlumbergera bridgesii)', 'Chrysanthemum', 'Ctenanthe', 
    'Daffodils (Narcissus spp.)', 'Dracaena', 'Dumb Cane (Dieffenbachia spp.)', 
    'Elephant Ear (Alocasia spp.)', 'English Ivy (Hedera helix)', 'Hyacinth (Hyacinthus orientalis)', 
    'Iron Cross begonia (Begonia masoniana)', 'Jade plant (Crassula ovata)', 'Kalanchoe', 
    'Lilium (Hemerocallis)', 'Lily of the valley (Convallaria majalis)', 'Money Tree (Pachira aquatica)', 
    'Monstera Deliciosa (Monstera deliciosa)', 'Orchid', 'Parlor Palm (Chamaedorea elegans)', 
    'Peace lily', 'Poinsettia (Euphorbia pulcherrima)', 'Polka Dot Plant (Hypoestes phyllostachya)', 
    'Ponytail Palm (Beaucarnea recurvata)', 'Pothos (Ivy arum)', 'Prayer Plant (Maranta leuconeura)', 
    'Rattlesnake Plant (Calathea lancifolia)', 'Rose', 'Rubber Plant (Ficus elastica)', 
    'Sago Palm (Cycas revoluta)', 'Schefflera', 'Snake plant (Sanseviera)', 'Tradescantia', 
    'Tulip', 'Venus Flytrap', 'Yucca', 'ZZ Plant (Zamioculcas zamiifolia)'
]

CLASS_ROSE_HEALTH = ['Black spot', 'Downy mildew', 'Insects Infected', 'Mosaic', 'Pure']
CLASS_POTHOS_HEALTH = ['Bacterial wilt disease', 'Healthy', 'Manganese Toxicity']

TARGET_ROSE = "Rose"
TARGET_POTHOS = "Pothos (Ivy arum)"


class IntegratedPlantAnalyzer:
    def __init__(self):
        """Initializes the analyzer and handles conditional loading of all models."""
        print("INFO: Loading production models...")
        try:
            self.model_detection = tf.keras.models.load_model(MODEL_DETECTION_PATH)
            self.model_species = tf.keras.models.load_model(MODEL_SPECIES_PATH)
            
            # Health models are loaded conditionally if the files exist
            self.model_rose_health = tf.keras.models.load_model(MODEL_ROSE_HEALTH_PATH) if os.path.exists(MODEL_ROSE_HEALTH_PATH) else None
            self.model_pothos_health = tf.keras.models.load_model(MODEL_POTHOS_HEALTH_PATH) if os.path.exists(MODEL_POTHOS_HEALTH_PATH) else None
            
            print("INFO: Core models loaded successfully.")
            if self.model_rose_health: print("INFO: Dedicated health model loaded: Rose")
            if self.model_pothos_health: print("INFO: Dedicated health model loaded: Pothos")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load models: {e}")
            exit(1)

    def download_image(self, url):
        """Downloads an image file from the specified URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                raise ValueError(f"URL content is not a valid image ({content_type})")
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            raise Exception(f"Image download failed: {e}")

    def preprocess_image(self, image):
        """Preprocesses the input image according to EfficientNetB0 requirements."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        # Standard EfficientNet preprocessing scaling input to [0, 255]
        processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return processed_img

    def get_top_k_predictions(self, predictions, class_names, k=3):
        """Sorts the model predictions and returns the top K most probable classes."""
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        return [
            {'label': class_names[idx], 'confidence': float(predictions[idx])}
            for idx in top_k_indices
        ]

    def process_pipeline(self, image_url):
        """Executes the multi-stage inference pipeline on the input URL."""
        raw_image = self.download_image(image_url)
        processed_tensor = self.preprocess_image(raw_image)
        
        # Initialize the baseline output report structure
        output_report = {
            'url': image_url,
            'is_plant': False,
            'detection_confidence': 0.0,
            'identified_species': None,
            'species_confidence': 0.0,
            'alternative_species': [],
            'health_status': None,
            'health_confidence': 0.0
        }

        # STAGE 1: Plant Detection
        det_preds = self.model_detection.predict(processed_tensor, verbose=0)[0]
        leaf_score = float(det_preds[0])      # Index 0: leaf
        no_leaf_score = float(det_preds[1])   # Index 1: no_leaf

        output_report['is_plant'] = leaf_score > no_leaf_score
        output_report['detection_confidence'] = max(leaf_score, no_leaf_score)

        # Abort the pipeline if no plant is detected or confidence is below threshold
        if not output_report['is_plant'] or output_report['detection_confidence'] < CONFIDENCE_THRESHOLD:
            return output_report

        # STAGE 2: Species Identification
        species_preds = self.model_species.predict(processed_tensor, verbose=0)[0]
        top_species = self.get_top_k_predictions(species_preds, CLASS_SPECIES, k=3)
        
        output_report['identified_species'] = top_species[0]['label']
        output_report['species_confidence'] = top_species[0]['confidence']
        output_report['alternative_species'] = top_species[1:]

        # STAGE 3: Conditional Health Diagnostics
        chosen_species = output_report['identified_species']
        
        if chosen_species == TARGET_ROSE and self.model_rose_health:
            health_preds = self.model_rose_health.predict(processed_tensor, verbose=0)[0]
            top_health = self.get_top_k_predictions(health_preds, CLASS_ROSE_HEALTH, k=1)[0]
            output_report['health_status'] = top_health['label']
            output_report['health_confidence'] = top_health['confidence']
            
        elif chosen_species == TARGET_POTHOS and self.model_pothos_health:
            health_preds = self.model_pothos_health.predict(processed_tensor, verbose=0)[0]
            top_health = self.get_top_k_predictions(health_preds, CLASS_POTHOS_HEALTH, k=1)[0]
            output_report['health_status'] = top_health['label']
            output_report['health_confidence'] = top_health['confidence']

        return output_report


def print_formatted_results(report):
    """Outputs the pipeline results directly to the standard console."""
    print("INTEGRATED BOTANICAL ANALYSIS REPORT")
    
    if not report['is_plant']:
        print("STATUS: No plant target detected in the image.")
        print(f"Confidence: {report['detection_confidence']:.2%}")
        print("Suggestion: Ensure the target object is centered and clearly visible.")
        print("=======================================================")
        return

    print(f"STATUS: Plant object verified (Confidence: {report['detection_confidence']:.2%})")
    print(f"IDENTIFIED SPECIES: {report['identified_species']}")
    print(f"Classification Confidence: {report['species_confidence']:.2%}")
    
    print("\nAlternative matches:")
    for alt in report['alternative_species']:
        print(f"   * {alt['label']} ({alt['confidence']:.2%})")
        
    print("\nHEALTH DIAGNOSTIC ANALYSIS:")
    if report['health_status']:
        print(f"   * Classification: {report['health_status']} ({report['health_confidence']:.2%})")
    else:
        print("   * Status: No dedicated health evaluation model available for this species.")
        print("     (Diagnostic features are currently limited to: Rose, Pothos)")
    print("=======================================================")


def validate_url(url):
    try:
        res = urlparse(url)
        return all([res.scheme, res.netloc])
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description='Integrated multi-model plant image analyzer pipeline.')
    parser.add_argument('--url', type=str, help='Direct URL to the plant image asset')
    args = parser.parse_args()

    analyzer = IntegratedPlantAnalyzer()

    if args.url:
        if not validate_url(args.url):
            print("ERROR: Provided string argument is not a valid URL structure.")
            return
        try:
            report = analyzer.process_pipeline(args.url)
            print_formatted_results(report)
        except Exception as e:
            print(f"ERROR: Execution failed during pipeline processing: {e}")
    else:
        print("\n--- INTERACTIVE CLI MODE (Type 'q' or 'exit' to terminate) ---")
        while True:
            try:
                url_input = input("\nEnter image URL: ").strip()
                if url_input.lower() in ['q', 'quit', 'exit']:
                    print("Exiting pipeline environment.")
                    break
                if not url_input or not validate_url(url_input):
                    print("WARNING: Invalid input sequence. Please submit a valid URL.")
                    continue
                
                report = analyzer.process_pipeline(url_input)
                print_formatted_results(report)
            except KeyboardInterrupt:
                print("\nProcess interrupted by user exit sequence.")
                break
            except Exception as e:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    main()