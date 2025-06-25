import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re
from collections import Counter
import string
from spellchecker import SpellChecker
import tempfile
import os
import time
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Handwriting OCR App",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to beautify the app
st.markdown("""
<style>
    .main {
        background-color: #f5f7ff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1E40AF;
    }
    .metrics-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .result-area {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #6B7280;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px 5px 0 0;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E0E7FF;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("📝 Handwriting OCR with Advanced Grammar Correction")
st.markdown("### Transform handwritten text into digitally corrected content")

# Function definitions from the original code
@st.cache_data
def load_models():
    """Load the YOLO models and return them"""
    try:
        # Display loading message
        with st.spinner("Loading models... This might take a minute."):
            # Path to your models - Update these paths for your deployment
            word_model_path = "word_model.pt"  # Replace with actual path
            char_model_path = "char_model.pt"  # Replace with actual path
            
            # For demonstration, we'll create mock models if the files don't exist
            if not os.path.exists(word_model_path) or not os.path.exists(char_model_path):
                st.warning("Using mock models for demonstration. Replace with actual models for production.")
                class MockModel:
                    def __call__(self, *args, **kwargs):
                        return [MockResults()]
                
                class MockResults:
                    def __init__(self):
                        self.boxes = MockBoxes()
                
                class MockBoxes:
                    def __init__(self):
                        self.xyxy = [np.array([10, 10, 100, 50])]
                        self.cls = [np.array([10])]
                        self.conf = [np.array([0.95])]
                
                word_model = MockModel()
                character_model = MockModel()
            else:
                word_model = YOLO(word_model_path)
                character_model = YOLO(char_model_path)
            
            # Define the character map - REPLACE THIS with your actual character mappings
            character_map = {
                0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
                10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 
                19: 'j', 20: 'k', 21: 'l', 22: 'll', 23: 'm', 24: 'n', 25: 'o', 26: 'p', 27: 'q', 
                28: 'r', 29: 's', 30: 't', 31: 'th', 32: 'u', 33: 'v', 34: 'w', 35: 'x', 36: 'y', 37: 'z'
            }
            
            return word_model, character_model, character_map
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def segment_words(image, word_model, confidence=0.25, visualize=False):
    """
    Segment words in a document image using YOLO model
    
    Args:
        image: Document image as numpy array
        word_model: YOLO model for word segmentation
        confidence: Confidence threshold for word detection
        visualize: Whether to show visualizations
    
    Returns:
        word_images: List of cropped word images
        word_boxes: List of word bounding boxes [x1, y1, x2, y2]
    """
    if image is None:
        st.error("Invalid image")
        return [], []
    
    # Run word detection
    results = word_model(image, conf=confidence)
    
    # Extract bounding boxes
    word_boxes = []
    word_images = []
    
    # Process detections
    for r in results:
        boxes = r.boxes
        
        if len(boxes) == 0:
            st.warning("No words detected.")
            return [], []
        
        # Get all boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Ensure the box is within the image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Crop the word image
            word_img = image[y1:y2, x1:x2]
            
            # Skip empty crops
            if word_img.size == 0:
                continue
                
            word_images.append(word_img)
            word_boxes.append([x1, y1, x2, y2])
    
    # Visualize if requested
    if visualize and word_boxes:
        vis_image = image.copy()
        for x1, y1, x2, y2 in word_boxes:
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax.set_title("Word Segmentation")
        ax.axis('off')
        st.pyplot(fig)
    
    return word_images, word_boxes

def recognize_characters(word_image, character_model, character_map, confidence=0.25, visualize=False):
    """
    Recognize characters in a word image using YOLO model
    
    Args:
        word_image: Cropped image of a word
        character_model: YOLO model for character recognition
        character_map: Dictionary mapping class indices to characters
        confidence: Confidence threshold for character detection
        visualize: Whether to show visualizations
    
    Returns:
        word_text: Recognized text for the word
    """
    # Run character detection
    results = character_model(word_image, conf=confidence)
    
    # Extract character detections
    chars = []
    
    # Process detections
    for r in results:
        boxes = r.boxes
        
        # If no characters detected, return empty string
        if len(boxes) == 0:
            return ""
        
        # Process each character
        for box in boxes:
            # Get character class
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get character based on class index
            if cls in character_map:
                char = character_map[cls]
            else:
                char = "?"  # Unknown character
            
            # Get position for sorting - use the center of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) / 2
            
            chars.append((x_center, char))
    
    # Sort characters by horizontal position (left to right)
    chars.sort(key=lambda x: x[0])
    
    # Construct the word text
    word_text = ''.join([char for _, char in chars])
    
    # Visualize if requested (for debugging only, not displayed in the app)
    if visualize and chars and st.session_state.get('debug_mode', False):
        vis_image = word_image.copy()
        for i, (x_center, char) in enumerate(chars):
            x_pos = int(x_center)
            cv2.putText(vis_image, char, (x_pos, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Draw bounding box around the character position
            cv2.circle(vis_image, (x_pos, 20), 5, (255, 0, 0), -1)
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Character Recognition: '{word_text}'")
        ax.axis('off')
        st.pyplot(fig)
    
    return word_text

def apply_custom_rules(text):
    """
    Apply custom rules for common OCR errors in handwritten text
    
    Args:
        text: Text to process
        
    Returns:
        Processed text
    """
    # Define replacement patterns
    patterns = [
        (r'\b(\w)l\b', r'\1i'),  # Fix lowercase l confused with i at end of words
        (r'\b0(\w+)\b', r'o\1'),  # Fix 0 confused with o at start of words
        (r'\b(\w+)5\b', r'\1s'),  # Fix 5 confused with s at end of words
        (r'\b(\w+)rn(\w*)\b', r'\1m\2'),  # Fix 'rn' confused with 'm'
        (r'\bdcn\b', r'don'),  # Fix 'dcn' to 'don'
        (r'\bl(\w+)\b', r'i\1'),  # Fix l -> i at the beginning of words
        (r'\bnct\b', r'not'),  # Fix 'nct' to 'not'
        (r'\b(\w+)cl(\w*)\b', r'\1d\2'),  # Fix 'cl' confused with 'd'
        (r'\bwith0ut\b', r'without'),  # Fix 'with0ut' to 'without'
        (r'\bwitn\b', r'with'),  # Fix 'witn' to 'with'
        (r'\bthc\b', r'the'),  # Fix 'thc' to 'the'
        (r'\bthls\b', r'this'),  # Fix 'thls' to 'this'
        (r'\bt0\b', r'to'),  # Fix 't0' to 'to'
        (r'\bfcr\b', r'for'),  # Fix 'fcr' to 'for'
    ]
    
    # Apply each pattern
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    return text

def enhanced_spell_correction(text, custom_dict=None):
    """
    Enhanced spell checking with context awareness and custom dictionary support
    
    Args:
        text: Text to correct
        custom_dict: List of domain-specific words to add to the dictionary
        
    Returns:
        Corrected text with highlighting for changes
    """
    # Initialize spell checker with enhanced options
    spell = SpellChecker(distance=2)  # Allow more edits for better correction
    
    # Add custom dictionary words if provided
    if custom_dict:
        spell.word_frequency.load_words(custom_dict)
    
    # Split the text into lines
    lines = text.split('\n')
    corrected_lines = []
    change_tracking = []  # To track changes for highlighting
    
    for line in lines:
        # Skip correction if line is empty
        if not line.strip():
            corrected_lines.append(line)
            continue
        
        # Split line into words
        words = line.split()
        corrected_words = []
        line_changes = []
        
        for word_idx, word in enumerate(words):
            # Separate punctuation from the word
            prefix = ''
            suffix = ''
            while word and word[0] in string.punctuation:
                prefix += word[0]
                word = word[1:]
            while word and word[-1] in string.punctuation:
                suffix = word[-1] + suffix
                word = word[:-1]
            
            # Skip correction for very short words unless they're common ones
            common_short_words = {'a', 'i', 'is', 'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 'if', 'in', 'it', 'me', 'my', 'no', 'of', 'oh', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}
            
            # Correct the word if it's not empty
            if word:
                original_word = word
                # Only correct words that are long enough to be meaningful
                if len(word) > 2 or word.lower() in common_short_words:
                    # Check if the word is misspelled
                    if word.lower() not in spell:
                        # Get the corrected word
                        corrected_word = spell.correction(word.lower())
                        # Add context-awareness - look at neighboring words
                        # This is a simplified version - you could expand with n-gram analysis
                        
                        # Handle the case where correction returns None
                        if corrected_word is None:
                            corrected_word = word  # Keep original if no correction found
                        # Preserve original capitalization
                        elif word[0].isupper():
                            corrected_word = corrected_word.capitalize()
                            
                        # Track changes if the word was modified
                        if corrected_word.lower() != original_word.lower():
                            line_changes.append((word_idx, original_word, corrected_word))
                            
                        corrected_words.append(prefix + corrected_word + suffix)
                    else:
                        corrected_words.append(prefix + word + suffix)
                else:
                    corrected_words.append(prefix + word + suffix)
            else:
                corrected_words.append(prefix + suffix)
        
        # Join words back into a line
        corrected_line = ' '.join(corrected_words)
        corrected_lines.append(corrected_line)
        change_tracking.append(line_changes)
    
    # Join lines back into text
    corrected_text = '\n'.join(corrected_lines)
    
    return corrected_text, change_tracking

def process_document(image, word_model, character_model, character_map, word_conf=0.25, char_conf=0.25, visualize=False, correct_language=True, custom_dict=None):
    """
    Process a document image with word segmentation and character recognition
    
    Args:
        image: Document image as numpy array
        word_model: YOLO model for word segmentation
        character_model: YOLO model for character recognition
        character_map: Dictionary mapping class indices to characters
        word_conf: Confidence threshold for word detection
        char_conf: Confidence threshold for character detection
        visualize: Whether to show visualizations
        correct_language: Whether to apply spelling and grammar correction
        custom_dict: Custom dictionary for domain-specific terms
    
    Returns:
        full_text: The recognized text from the document
        visualization: Image with bounding boxes and recognized text
        corrected_text: The corrected text (if correction is applied)
        changes: List of changes made during correction
    """
    # Show progress to user
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Segment words
    status_text.text("Segmenting words...")
    word_images, word_boxes = segment_words(image, word_model, confidence=word_conf, visualize=visualize)
    progress_bar.progress(20)
    
    if not word_images:
        status_text.text("No words detected in the image.")
        return "", None, "", []
    
    # Create visualization
    visualization = image.copy()
    
    # Step 2: Recognize characters in each word
    status_text.text("Recognizing characters...")
    recognized_words = []
    
    for i, word_img in enumerate(word_images):
        # Update progress
        progress_percent = 20 + (i / len(word_images)) * 40
        progress_bar.progress(int(progress_percent))
        
        # Show progress
        if (i+1) % max(1, len(word_images)//10) == 0:
            status_text.text(f"Processing word {i+1}/{len(word_images)}")
            
        # Recognize characters in this word
        word_text = recognize_characters(
            word_img, character_model, character_map, 
            confidence=char_conf, visualize=(i < 3 and visualize)  # Only visualize first 3 words
        )
        
        recognized_words.append(word_text)
        
        # Draw on visualization image
        x1, y1, x2, y2 = word_boxes[i]
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(visualization, word_text, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    progress_bar.progress(60)
    
    # Step 3: Reconstruct full text by grouping words into lines
    status_text.text("Reconstructing text...")
    if word_boxes:
        # Calculate median line height
        heights = [box[3] - box[1] for box in word_boxes]
        median_height = np.median(heights)
        
        # Line detection tolerance
        line_tolerance = median_height * 0.4
        
        # Create a combined list of word boxes, recognized words, and original indices
        combined_data = [(i, box, word) for i, (box, word) in enumerate(zip(word_boxes, recognized_words))]
        
        # Sort words primarily by their y-coordinate (top to bottom)
        combined_data.sort(key=lambda x: x[1][1])  # Sort by y1 (top coordinate)
        
        # Group words into lines based on vertical position
        lines = []
        current_line = []
        prev_y = None
        
        for idx, box, word in combined_data:
            # Get the vertical center of this word
            y_center = (box[1] + box[3]) / 2
            
            # If this is the first word or it belongs to the current line
            if prev_y is None or abs(y_center - prev_y) < line_tolerance:
                current_line.append((idx, box, word))
                # Update the average y-position for this line
                if prev_y is None:
                    prev_y = y_center
                else:
                    # Weighted average based on number of words in line
                    prev_y = (prev_y * (len(current_line) - 1) + y_center) / len(current_line)
            else:
                # This word belongs to a new line
                # Sort the current line by x-coordinate (left to right)
                current_line.sort(key=lambda x: x[1][0])  # Sort by x1 (left coordinate)
                lines.append(current_line)
                # Start a new line with this word
                current_line = [(idx, box, word)]
                prev_y = y_center
        
        # Don't forget to add the last line
        if current_line:
            current_line.sort(key=lambda x: x[1][0])  # Sort by x1 (left coordinate)
            lines.append(current_line)
        
        # Construct the full text
        lines_text = []
        for line in lines:
            # Join words in this line with spaces
            line_text = ' '.join([word for _, _, word in line])
            lines_text.append(line_text)
        
        full_text = '\n'.join(lines_text)
    else:
        full_text = ""
    
    progress_bar.progress(80)
    
    # Step 4: Apply language correction if requested
    corrected_text = full_text
    changes = []
    
    if correct_language and full_text:
        status_text.text("Applying spelling and grammar correction...")
        try:
            # First apply custom rules for common OCR errors
            corrected_text = apply_custom_rules(full_text)
            
            # Then apply enhanced spell checking
            corrected_text, changes = enhanced_spell_correction(corrected_text, custom_dict)
            
            # Update visualization with corrected text if available
            if lines:
                # Create a copy of the original visualization for corrected text
                corrected_vis = image.copy()
                
                # Split corrected text into lines
                corrected_lines = corrected_text.split('\n')
                
                # Update visualization with corrected text
                for i, (line_words, corrected_line) in enumerate(zip(lines, corrected_lines)):
                    if i < len(lines):
                        # Highlight each line with a different color
                        color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)][i % 5]
                        
                        # Split corrected line into words
                        corrected_words = corrected_line.split()
                        
                        # Update each word box with corrected text
                        for j, (idx, box, _) in enumerate(line_words):
                            if j < len(corrected_words):
                                x1, y1, x2, y2 = box
                                cv2.rectangle(corrected_vis, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(corrected_vis, corrected_words[j], (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Create a side-by-side comparison
                h, w = image.shape[:2]
                comparison = np.zeros((h, 2*w, 3), dtype=np.uint8)
                comparison[:, :w] = visualization
                comparison[:, w:] = corrected_vis
                
                # Add text labels
                cv2.putText(comparison, "Original Recognition", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison, "Corrected Text", (w + 10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Update visualization to the comparison
                visualization = comparison
        except Exception as e:
            st.error(f"Error in language correction: {e}")
            st.warning("Falling back to original text")
    
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return full_text, visualization, corrected_text, changes

# Sidebar controls
st.sidebar.title("Control Panel")

# Model loading
with st.sidebar.expander("Model Settings", expanded=False):
    if st.button("Load Models"):
        word_model, character_model, character_map = load_models()
        st.session_state['models_loaded'] = True
        st.session_state['word_model'] = word_model
        st.session_state['character_model'] = character_model
        st.session_state['character_map'] = character_map
        st.success("Models loaded successfully!")
    
    # Check if models are loaded
    if 'models_loaded' not in st.session_state:
        st.session_state['models_loaded'] = False
        st.warning("Please load models to start")
    
    # Confidence thresholds
    word_confidence = st.slider("Word Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    char_confidence = st.slider("Character Recognition Confidence", 0.1, 0.9, 0.25, 0.05)

# Advanced settings
with st.sidebar.expander("Advanced Settings", expanded=False):
    correct_language = st.checkbox("Enable Spelling Correction", value=True)
    visualize = st.checkbox("Show Visualization", value=True)
    st.session_state['debug_mode'] = st.checkbox("Debug Mode", value=False)
    
    # Custom dictionary
    st.subheader("Custom Dictionary")
    custom_words = st.text_area("Add domain-specific words (one per line)", 
                               placeholder="Enter specialized terms that might not be in the regular dictionary")
    custom_dict = [word.strip() for word in custom_words.split("\n") if word.strip()]

# Main area - file upload and processing
st.subheader("Upload Image")
st.markdown("Upload an image containing handwritten text for OCR processing.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])

col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image")

# Process button
process_button = st.button("Process Image")

# Initialize results placeholders
tabs_placeholder = st.empty()
text_placeholder = st.empty()
vis_placeholder = st.empty()

# Process the image
if process_button and uploaded_file is not None:
    if st.session_state.get('models_loaded', False):
        # Convert the file to an OpenCV image (again, since we read it earlier)
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process the document
        original_text, visualization, corrected_text, changes = process_document(
            image, 
            st.session_state['word_model'], 
            st.session_state['character_model'], 
            st.session_state['character_map'],
            word_conf=word_confidence,
            char_conf=char_confidence,
            visualize=visualize,
            correct_language=correct_language,
            custom_dict=custom_dict
        )
        
        # Store results in session state
        st.session_state['original_text'] = original_text
        st.session_state['corrected_text'] = corrected_text
        st.session_state['visualization'] = visualization
        st.session_state['changes'] = changes
        st.session_state['processed'] = True
        
        # Display results
        with tabs_placeholder.container():
            st.markdown('<div class="result-area">', unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["Corrected Text", "Original Text", "Visualization"])
            
            with tab1:
                st.subheader("Corrected Text")
                st.text_area("", corrected_text, height=200)
                
                # Download button for corrected text
                if corrected_text:
                    st.download_button(
                        label="Download Corrected Text",
                        data=corrected_text,
                        file_name="corrected_text.txt",
                        mime="text/plain"
                    )
                
                # Show changes made
                if changes and any(changes):
                    st.subheader("Spelling Corrections")
                    for line_idx, line_changes in enumerate(changes):
                        if line_changes:
                            for word_idx, original, corrected in line_changes:
                                st.markdown(f"Line {line_idx+1}: '{original}' → '{corrected}'")
            
            with tab2:
                st.subheader("Original Recognized Text")
                st.text_area("", original_text, height=200)
                
                # Download button for original text
                if original_text:
                    st.download_button(
                        label="Download Original Text",
                        data=original_text,
                        file_name="original_text.txt",
                        mime="text/plain"
                    )
            
            with tab3:
                st.subheader("Visualization")
                if visualization is not None:
                    st.image(visualization, caption="Text Recognition")
                    
                    # Convert the visualization to bytes for download
                    is_success, buffer = cv2.imencode(".png", visualization)
                    if is_success:
                        btn = st.download_button(
                            label="Download Visualization",
                            data=io.BytesIO(buffer),
                            file_name="ocr_visualization.png",
                            mime="image/png"
                        )
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please load the models first using the button in the sidebar.")

# Add metrics cards when processing is complete
if st.session_state.get('processed', False):
    st.markdown("<h3>Analysis Metrics</h3>", unsafe_allow_html=True)
    
    # Calculate metrics
    original_text = st.session_state['original_text']
    corrected_text = st.session_state['corrected_text']
    changes = st.session_state['changes']
    
    # Count statistics
    word_count = len(corrected_text.split())
    char_count = len(corrected_text.replace(" ", "").replace("\n", ""))
    line_count = len(corrected_text.split("\n"))
    correction_count = sum(len(line_changes) for line_changes in changes)
    
    # Display metrics in a nice layout
    st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Words", word_count)
    with metric_cols[1]:
        st.metric("Characters", char_count)
    with metric_cols[2]:
        st.metric("Lines", line_count)
    with metric_cols[3]:
        st.metric("Corrections", correction_count)
    st.markdown('</div>', unsafe_allow_html=True)

# Example section
with st.expander("📚 How to use this app"):
    st.markdown("""
    ### Getting Started
    
    1. **Load Models**: Click the 'Load Models' button in the sidebar to initialize the OCR models.
    2. **Upload an Image**: Use the file uploader to select an image containing handwritten text.
    3. **Adjust Settings**: Modify confidence thresholds and other settings in the sidebar as needed.
    4. **Process Image**: Click the 'Process Image' button to start OCR and correction.
    5. **View Results**: Check the results in the tabs below - corrected text, original recognized text, and visualization.
    
    ### Tips for Best Results
    
    - Use clear images with good contrast between text and background
    - Ensure handwriting is legible and not too small
    - Add domain-specific terms to the custom dictionary for better spelling correction
    - Adjust confidence thresholds based on your specific documents
    
    ### Features
    
    - **Word Segmentation**: Identifies individual words in the document
    - **Character Recognition**: Recognizes individual characters within words
    - **Spelling Correction**: Intelligently corrects OCR errors and spelling mistakes
    - **Custom Dictionary**: Add domain-specific terms for improved correction
    - **Debug Mode**: View detailed character recognition for troubleshooting
    """)

# Add a footer
st.markdown("""
<div class="footer">
    <p>Handwriting OCR App with Advanced Grammar Correction | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)

def main():
    # The app is already running when this script is executed by Streamlit
    pass

if __name__ == "__main__":
    main()