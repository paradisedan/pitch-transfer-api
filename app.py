# app.py
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS  # Import Flask-CORS
from werkzeug.utils import secure_filename
import os
import uuid
import parselmouth
from parselmouth.praat import call
import numpy as np
import soundfile as sf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# Configure upload folder
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav_if_needed(input_file, output_file=None):
    """Convert audio to WAV format if it's not already WAV."""
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.wav'
    
    # Check if already WAV format
    if input_file.lower().endswith('.wav'):
        # Just copy the file
        import shutil
        shutil.copy(input_file, output_file)
        return output_file
    
    try:
        # Load audio file
        data, samplerate = sf.read(input_file)
        # Save as WAV
        sf.write(output_file, data, samplerate, subtype='PCM_16')
        return output_file
    except Exception as e:
        logger.error(f"Error converting file {input_file}: {str(e)}")
        raise

def transfer_pitch(source_file, target_file, output_file):
    """Extract pitch from source_file and apply it to target_file."""
    try:
        # Convert to WAV if needed
        source_wav = convert_to_wav_if_needed(source_file)
        target_wav = convert_to_wav_if_needed(target_file)
        
        logger.info(f"Processing files: source={source_wav}, target={target_wav}")
        
        # Load sound files
        source_sound = parselmouth.Sound(source_wav)
        target_sound = parselmouth.Sound(target_wav)
        
        # Extract pitch from source
        source_pitch = source_sound.to_pitch()
        
        # Manipulate the target sound with the source pitch
        manipulation = call(target_sound, "To Manipulation", 0.01, 75, 600)
        
        # Extract pitch tier from source pitch
        pitch_tier = call(source_pitch, "Down to PitchTier")
        
        # Replace pitch in manipulation object
        call([pitch_tier, manipulation], "Replace pitch tier")
        
        # Generate new sound
        new_sound = call(manipulation, "Get resynthesis (overlap-add)")
        
        # Convert to numpy array for saving with soundfile
        y = np.array(new_sound.values)
        sample_rate = new_sound.sampling_frequency
        
        # Save the output
        sf.write(output_file, y, int(sample_rate))
        
        # Clean up temporary WAV files if they were created
        if source_wav != source_file and os.path.exists(source_wav):
            os.remove(source_wav)
        if target_wav != target_file and os.path.exists(target_wav):
            os.remove(target_wav)
            
        return True, "Processing successful"
    except Exception as e:
        logger.error(f"Error in transfer_pitch: {str(e)}")
        return False, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/process', methods=['POST'])
def process_audio():
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
        return jsonify({"error": "No selected file"}), 400
    
    # Check file extensions
    if not (allowed_file(source_file.filename) and allowed_file(target_file.filename)):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        # Create unique filenames
        source_filename = str(uuid.uuid4()) + '_' + secure_filename(source_file.filename)
        target_filename = str(uuid.uuid4()) + '_' + secure_filename(target_file.filename)
        output_filename = str(uuid.uuid4()) + '.wav'
        
        # Save uploaded files
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        source_file.save(source_path)
        target_file.save(target_path)
        
        logger.info(f"Saved files to: source={source_path}, target={target_path}")
        
        # Process audio files
        success, message = transfer_pitch(source_path, target_path, output_path)
        
        if not success:
            # Clean up files
            for path in [source_path, target_path]:
                if os.path.exists(path):
                    os.remove(path)
            logger.error(f"Processing failed: {message}")
            return jsonify({"error": f"Processing failed: {message}"}), 500
        
        logger.info(f"Processing successful, returning file: {output_path}")
        
        # Return the processed file
        response = send_file(output_path, as_attachment=True, 
                            download_name="processed_audio.wav",
                            mimetype="audio/wav")
        
        # Clean up files after sending response
        @response.call_on_close
        def cleanup():
            for path in [source_path, target_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
