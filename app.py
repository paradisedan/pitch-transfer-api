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

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins, or specify allowed domains like ["https://yourdomain.com"]
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

def transfer_pitch(source_file, target_file, output_file):
    """Extract pitch from source_file and apply it to target_file."""
    try:
        # Load sound files
        source_sound = parselmouth.Sound(source_file)
        target_sound = parselmouth.Sound(target_file)
        
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
        
        return True, "Processing successful"
    except Exception as e:
        return False, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/process', methods=['POST'])
def process_audio():
    # Check if files exist in request
    if 'source_audio' not in request.files or 'target_audio' not in request.files:
        return jsonify({"error": "Missing source or target file"}), 400
    
    source_file = request.files['source_audio']
    target_file = request.files['target_audio']
    
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
        
        # Process audio files
        success, message = transfer_pitch(source_path, target_path, output_path)
        
        if not success:
            # Clean up files
            for path in [source_path, target_path]:
                if os.path.exists(path):
                    os.remove(path)
            return jsonify({"error": f"Processing failed: {message}"}), 500
        
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
