# app.py
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import parselmouth
from parselmouth.praat import call
import numpy as np
import soundfile as sf
import logging
import time
from functools import wraps
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# Configure upload folder
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Basic rate limiting
REQUEST_HISTORY = {}
MAX_REQUESTS_PER_MINUTE = 10

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr
        current_time = time.time()
        
        # Clean up old entries
        for key in list(REQUEST_HISTORY.keys()):
            if current_time - REQUEST_HISTORY[key][-1] > 60:
                del REQUEST_HISTORY[key]
        
        # Check rate limit
        if ip in REQUEST_HISTORY:
            requests = REQUEST_HISTORY[ip]
            requests = [t for t in requests if current_time - t < 60]
            
            if len(requests) >= MAX_REQUESTS_PER_MINUTE:
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
            
            REQUEST_HISTORY[ip] = requests + [current_time]
        else:
            REQUEST_HISTORY[ip] = [current_time]
        
        return func(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_remove(file_path):
    """Safely remove a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")
    except Exception as e:
        logger.error(f"Error removing file {file_path}: {str(e)}")

def convert_to_wav_if_needed(input_file, output_file=None):
    """Convert audio to WAV format if it's not already WAV."""
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.wav'
    
    # Check if already WAV format
    if input_file.lower().endswith('.wav'):
        # Just copy the file
        try:
            shutil.copy(input_file, output_file)
            logger.info(f"Copied WAV file from {input_file} to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error copying WAV file: {str(e)}")
            raise
    
    try:
        # Load audio file
        logger.info(f"Converting {input_file} to WAV format at {output_file}")
        data, samplerate = sf.read(input_file)
        
        # Save as WAV with explicit format
        sf.write(output_file, data, samplerate, subtype='PCM_16', format='WAV')
        
        # Verify the file was created successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Successfully converted to WAV: {output_file} ({os.path.getsize(output_file)} bytes)")
            return output_file
        else:
            logger.error(f"Failed to create WAV file at {output_file}")
            raise Exception(f"Failed to create WAV file at {output_file}")
    except Exception as e:
        logger.error(f"Error converting file {input_file}: {str(e)}")
        raise

def transfer_pitch(source_file, target_file, output_file):
    """Extract pitch from source_file and apply it to target_file."""
    source_wav = None
    target_wav = None
    
    try:
        # Ensure source and target are different files
        if os.path.samefile(source_file, target_file):
            logger.error("Source and target files are the same")
            return False, "Source and target files are the same"
            
        # Convert to WAV if needed
        source_wav = convert_to_wav_if_needed(source_file, os.path.join(os.path.dirname(source_file), f"source_{uuid.uuid4()}.wav"))
        target_wav = convert_to_wav_if_needed(target_file, os.path.join(os.path.dirname(target_file), f"target_{uuid.uuid4()}.wav"))
        
        logger.info(f"Processing files: source={source_wav}, target={target_wav}")
        
        # Load sound files
        logger.info(f"Loading source sound from {source_wav}")
        source_sound = parselmouth.Sound(source_wav)
        logger.info(f"Loading target sound from {target_wav}")
        target_sound = parselmouth.Sound(target_wav)
        
        # Extract pitch from source
        logger.info("Extracting pitch from source")
        source_pitch = source_sound.to_pitch()
        
        # Manipulate the target sound with the source pitch
        logger.info("Creating manipulation object")
        manipulation = call(target_sound, "To Manipulation", 0.01, 75, 600)
        
        # Extract pitch tier from source pitch
        logger.info("Extracting pitch tier")
        pitch_tier = call(source_pitch, "Down to PitchTier")
        
        # Replace pitch in manipulation object
        logger.info("Replacing pitch tier")
        call([pitch_tier, manipulation], "Replace pitch tier")
        
        # Generate new sound
        logger.info("Generating new sound")
        new_sound = call(manipulation, "Get resynthesis (overlap-add)")
        
        # Convert to numpy array for saving with soundfile
        logger.info("Converting to numpy array")
        y = np.array(new_sound.values)
        sample_rate = new_sound.sampling_frequency
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the output - specify PCM_16 format explicitly
        logger.info(f"Saving output to {output_file} with sample rate {int(sample_rate)}")
        sf.write(output_file, y, int(sample_rate), subtype='PCM_16', format='WAV')
        
        # Verify the file was created successfully
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.error(f"Failed to create output file at {output_file}")
            return False, f"Failed to create output file at {output_file}"
            
        logger.info(f"Successfully created output file: {output_file} ({os.path.getsize(output_file)} bytes)")
        
        return True, "Processing successful"
    except Exception as e:
        logger.error(f"Error in transfer_pitch: {str(e)}")
        return False, str(e)
    finally:
        # Clean up temporary WAV files
        if source_wav and source_wav != source_file:
            safe_remove(source_wav)
        if target_wav and target_wav != target_file:
            safe_remove(target_wav)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/process', methods=['POST'])
@rate_limit
def process_audio():
    source_path = None
    target_path = None
    output_path = None
    
    try:
        logger.info("Received pitch transfer request")
        
        # Check if files exist in request
        if 'source_audio' not in request.files or 'target_audio' not in request.files:
            logger.error("Missing source_audio or target_audio in request")
            return jsonify({"error": "Missing source_audio or target_audio file"}), 400
        
        source_file = request.files['source_audio']
        target_file = request.files['target_audio']
        
        logger.info(f"Received files: source={source_file.filename} ({source_file.content_type}), target={target_file.filename} ({target_file.content_type})")
        
        # Check if filenames are valid
        if source_file.filename == '' or target_file.filename == '':
            logger.error("Empty filename provided")
            return jsonify({"error": "No selected file"}), 400
        
        # Check file extensions
        if not (allowed_file(source_file.filename) and allowed_file(target_file.filename)):
            logger.error(f"Invalid file types: source={source_file.filename}, target={target_file.filename}")
            return jsonify({"error": "File type not allowed"}), 400
        
        # Create unique filenames with different UUIDs
        source_filename = f"{uuid.uuid4()}_source_{secure_filename(source_file.filename)}"
        target_filename = f"{uuid.uuid4()}_target_{secure_filename(target_file.filename)}"
        output_filename = f"{uuid.uuid4()}_output.wav"
        
        # Save uploaded files
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        source_file.save(source_path)
        target_file.save(target_path)
        
        logger.info(f"Saved files to: source={source_path}, target={target_path}")
        
        # Check if files were saved correctly
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            logger.error("Failed to save uploaded files")
            return jsonify({"error": "Failed to save uploaded files"}), 500
            
        if os.path.getsize(source_path) == 0 or os.path.getsize(target_path) == 0:
            logger.error("Uploaded files are empty")
            return jsonify({"error": "Uploaded files are empty"}), 400
        
        # Process audio files
        logger.info(f"Starting pitch transfer process: source={source_path}, target={target_path}, output={output_path}")
        success, message = transfer_pitch(source_path, target_path, output_path)
        
        if not success:
            logger.error(f"Processing failed: {message}")
            return jsonify({"error": f"Processing failed: {message}"}), 500
        
        # Check if output file exists and is not empty
        if not os.path.exists(output_path):
            logger.error(f"Output file does not exist: {output_path}")
            return jsonify({"error": "Output file was not created"}), 500
            
        if os.path.getsize(output_path) == 0:
            logger.error(f"Output file is empty: {output_path}")
            return jsonify({"error": "Output file is empty"}), 500
        
        logger.info(f"Processing successful, returning file: {output_path} ({os.path.getsize(output_path)} bytes)")
        
        # Return the processed file
        response = send_file(output_path, 
                            as_attachment=True, 
                            download_name="processed_audio.wav",
                            mimetype="audio/wav")
        
        # Clean up files after sending response
        @response.call_on_close
        def cleanup():
            logger.info("Cleaning up temporary files")
            safe_remove(source_path)
            safe_remove(target_path)
            safe_remove(output_path)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        # Clean up files in case of error
        if source_path:
            safe_remove(source_path)
        if target_path:
            safe_remove(target_path)
        if output_path:
            safe_remove(output_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
