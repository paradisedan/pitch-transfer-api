# app.py
from flask import Flask, request, send_file, jsonify, after_this_request
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
import wave
import io
import tempfile
import subprocess

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

def is_valid_wav(file_path):
    """Check if a file is a valid WAV file."""
    try:
        with wave.open(file_path, 'rb') as wf:
            if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 44100:
                return True
    except Exception as e:
        logger.error(f"Error checking WAV file {file_path}: {str(e)}")
    return False

def convert_to_wav(input_file, output_file=None):
    """Convert audio file to WAV format using ffmpeg or soundfile as fallback"""
    if output_file is None:
        output_file = f"/tmp/{os.path.splitext(os.path.basename(input_file))[0]}_{uuid.uuid4()}.wav"
    
    # Check if the file is already a WAV file with the right format
    if is_valid_wav(input_file):
        logger.info(f"File {input_file} is already a valid WAV file")
        return input_file, False
    
    # First try with ffmpeg if available
    try:
        # Check if ffmpeg is available without running the full conversion
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=1)
        
        logger.info(f"Converting {input_file} to WAV using ffmpeg")
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file, 
            '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
            output_file
        ], capture_output=True, check=True)
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Successfully converted to WAV: {output_file} ({os.path.getsize(output_file)} bytes)")
            return output_file, True
        else:
            logger.error(f"ffmpeg conversion failed to produce output file")
            raise Exception("ffmpeg conversion failed to produce output file")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting file {input_file}: {e}")
        logger.warning(f"ffmpeg conversion failed, trying soundfile: {e}")
    except FileNotFoundError as e:
        logger.warning(f"ffmpeg not found, using soundfile instead: {e}")
    except Exception as e:
        logger.error(f"Error converting file {input_file}: {e}")
        logger.warning(f"ffmpeg conversion failed, trying soundfile: {e}")
    
    # Fallback to soundfile
    try:
        logger.info(f"Converting {input_file} to WAV using soundfile")
        audio_data, sample_rate = sf.read(input_file)
        sf.write(output_file, audio_data, sample_rate, subtype='PCM_16', format='WAV')
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Successfully converted to WAV: {output_file} ({os.path.getsize(output_file)} bytes)")
            return output_file, True
        else:
            logger.error(f"soundfile conversion failed to produce output file")
            raise Exception("soundfile conversion failed to produce output file")
    except Exception as e:
        logger.error(f"Error converting file with soundfile {input_file}: {e}")
        raise Exception(f"Failed to convert audio file: {str(e)}")

def safe_remove(file_path):
    """Safely remove a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")
    except Exception as e:
        logger.error(f"Error removing file {file_path}: {str(e)}")

def transfer_pitch(source_file, target_file, output_file, 
                  time_step=0.01, min_pitch=75, max_pitch=600,
                  resynthesis_method="psola", voicing_threshold=0.3,
                  octave_cost=0.01, octave_jump_cost=0.5, voiced_unvoiced_cost=0.14,
                  preserve_formants=True):
    """
    Transfer pitch from source audio to target audio.
    
    Args:
        source_file: Path to source audio file
        target_file: Path to target audio file
        output_file: Path to output audio file
        time_step: Time step for pitch analysis in seconds
        min_pitch: Minimum pitch in Hz
        max_pitch: Maximum pitch in Hz (default: 300)
        resynthesis_method: Method for resynthesis (overlap-add, psola, or LPC)
        voicing_threshold: Threshold for voiced/unvoiced decision
        octave_cost: Cost for octave jumps
        octave_jump_cost: Cost for octave jumps
        voiced_unvoiced_cost: Cost for voiced/unvoiced transitions
        preserve_formants: Whether to preserve formants
        
    Returns:
        Tuple of (success, message)
    """
    source_wav = None
    target_wav = None
    
    try:
        # Check if source and target are the same file
        if os.path.abspath(source_file) == os.path.abspath(target_file):
            logger.error("Source and target files are the same")
            return False, "Source and target files are the same"
            
        # Convert to WAV if needed
        source_wav, _ = convert_to_wav(source_file, os.path.join(os.path.dirname(source_file), f"source_{uuid.uuid4()}.wav"))
        target_wav, _ = convert_to_wav(target_file, os.path.join(os.path.dirname(target_file), f"target_{uuid.uuid4()}.wav"))
        
        logger.info(f"Processing files: source={source_wav}, target={target_wav}")
        
        # Load source and target sounds
        logger.info(f"Loading source sound from {source_wav}")
        source_sound = parselmouth.Sound(source_wav)
        logger.info(f"Source sound loaded: duration={source_sound.duration} seconds, sampling frequency={source_sound.sampling_frequency} Hz")
        
        logger.info(f"Loading target sound from {target_wav}")
        target_sound = parselmouth.Sound(target_wav)
        logger.info(f"Target sound loaded: duration={target_sound.duration} seconds, sampling frequency={target_sound.sampling_frequency} Hz")
        
        # Extract pitch from source with specified parameters
        logger.info(f"Extracting pitch from source with time_step={time_step}, min_pitch={min_pitch}, max_pitch={max_pitch}")
        source_pitch = source_sound.to_pitch(
            time_step=time_step, 
            pitch_floor=min_pitch, 
            pitch_ceiling=max_pitch
        )
        logger.info(f"Source pitch extracted: {source_pitch}")
        
        # Smooth the pitch contour for better quality
        logger.info("Smoothing pitch contour (bandwidth=1.5)")
        try:
            smoothed_pitch = source_pitch.smooth(bandwidth=1.5)  # Bandwidth of 1.5 semitones
            logger.info("Pitch contour smoothed successfully")
            source_pitch = smoothed_pitch
        except Exception as e:
            logger.warning(f"Failed to smooth pitch contour: {str(e)}")
        
        # Create manipulation object with specified parameters
        logger.info(f"Creating manipulation object with time_step={time_step}, min_pitch={min_pitch}, max_pitch={max_pitch}")
        manipulation = call(target_sound, "To Manipulation", time_step, min_pitch, max_pitch)
        
        # Extract pitch tier from manipulation
        logger.info("Extracting pitch tier from manipulation")
        pitch_tier = call([manipulation], "Extract pitch tier")
        
        # Replace pitch tier with source pitch
        logger.info("Creating pitch tier from source pitch")
        source_pitch_tier = call([source_pitch], "Down to PitchTier")
        
        logger.info("Replacing pitch tier in manipulation")
        call([manipulation, source_pitch_tier], "Replace pitch tier")
        
        # Determine resynthesis command based on method
        if resynthesis_method.lower() == "psola":
            logger.info("Using PSOLA (pitch-synchronous overlap-add) resynthesis method")
            
            # Simplified PSOLA validation - check essential requirements
            psola_compatible = True
            psola_issues = []
            
            # Check 1: Mono sound
            if target_sound.n_channels != 1:
                psola_compatible = False
                psola_issues.append("Sound must be mono")
            
            # Check 2: Time step is short enough
            if time_step > 0.01:
                psola_compatible = False
                psola_issues.append(f"Time step too large: {time_step}")
            
            # Check 3: Quick pitch tier validation
            try:
                pitch_tier_check = call([manipulation], "Extract pitch tier") # Re-extract for check
                points = call(pitch_tier_check, "Get number of points")
                if points == 0:
                    psola_compatible = False
                    psola_issues.append("No pitch points detected")
            except Exception as e:
                psola_compatible = False
                psola_issues.append(f"Could not analyze pitch tier: {str(e)}")
            
            # Log PSOLA compatibility status
            if psola_compatible:
                logger.info("Sound meets requirements for PSOLA")
            else:
                logger.warning(f"Sound does not meet PSOLA requirements: {', '.join(psola_issues)}")
            
            # Simplified resynthesis selection
            if psola_compatible:
                try:
                    new_sound = call(manipulation, "Get resynthesis (PSOLA)")
                    logger.info(f"PSOLA successful: duration={new_sound.duration}s")
                except Exception as e:
                    logger.warning(f"PSOLA failed despite compatibility checks: {str(e)}")
                    new_sound = call(manipulation, "Get resynthesis (overlap-add)")
                    logger.info(f"Fallback to overlap-add: duration={new_sound.duration}s")
            else:
                logger.warning("Using overlap-add due to PSOLA compatibility issues")
                new_sound = call(manipulation, "Get resynthesis (overlap-add)")
                logger.info(f"Using overlap-add: duration={new_sound.duration}s")
        elif resynthesis_method.lower() == "lpc":
            resynthesis_command = "Get resynthesis (LPC)"
            new_sound = call(manipulation, resynthesis_command)
            logger.info(f"New sound generated with {resynthesis_method}: duration={new_sound.duration} seconds")
        else:  # Default to overlap-add (though should typically hit psola logic first)
            resynthesis_command = "Get resynthesis (overlap-add)"
            new_sound = call(manipulation, resynthesis_command)
            logger.info(f"New sound generated with {resynthesis_method}: duration={new_sound.duration} seconds")

        # --- Apply Formant Preservation (if enabled) ---
        if preserve_formants:
            logger.info("Attempting formant preservation...")
            try:
                # Extract formants from the original target sound
                # Parameters: time_step, max_num_formants, max_formant_freq, window_length, pre_emphasis_from
                formants = call(target_sound, "To Formant (burg)...", 0.01, 5, 5500, 0.025, 50)
                logger.info("Formants extracted from target sound.")
                
                # Filter the newly generated sound with the extracted formants
                filtered_sound = call([new_sound, formants], "Filter (formants)...")
                logger.info("New sound filtered with target formants.")
                new_sound = filtered_sound # Replace the new_sound with the filtered version
            except Exception as e:
                logger.warning(f"Formant preservation failed: {str(e)}. Proceeding without formant filtering.")
        else:
            logger.info("Formant preservation disabled.")
        # --- End Formant Preservation ---
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the output
        logger.info(f"Saving output to {output_file}")
        try:
            # Save the sound directly with Parselmouth
            new_sound.save(output_file, "WAV")
            
            # Verify the file was created successfully
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Successfully saved WAV file: {output_file} ({os.path.getsize(output_file)} bytes)")
            else:
                logger.error(f"Failed to create WAV file at {output_file}")
                return False, "Failed to save output file"
        except Exception as e:
            logger.error(f"Error saving sound: {str(e)}")
            return False, f"Error saving sound: {str(e)}"
        
        # Verify the file was created successfully
        if not os.path.exists(output_file):
            logger.error(f"Output file does not exist: {output_file}")
            return False, f"Output file was not created"
            
        logger.info(f"Successfully created output file: {output_file} ({os.path.getsize(output_file)} bytes)")
        
        # Try to validate the output file by reading it back
        try:
            logger.info("Validating output file by reading it back")
            with wave.open(output_file, 'rb') as wf:
                logger.info(f"Output file validation: channels={wf.getnchannels()}, sample width={wf.getsampwidth()}, framerate={wf.getframerate()}, frames={wf.getnframes()}")
        except Exception as e:
            logger.warning(f"Output file validation failed: {str(e)}")
            # Try to fix the file with ffmpeg
            try:
                logger.info("Attempting to fix output file with ffmpeg")
                fixed_output = os.path.join(os.path.dirname(output_file), f"fixed_{uuid.uuid4()}.wav")
                convert_to_wav_with_ffmpeg(output_file, fixed_output)
                os.replace(fixed_output, output_file)
                logger.info(f"Fixed output file: {output_file}")
            except Exception as fix_error:
                logger.error(f"Failed to fix output file: {str(fix_error)}")
                return False, f"Output file validation failed: {str(e)}"
        
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
def process_audio():
    source_path = None
    target_path = None
    output_path = None
    temp_files = []
    
    try:
        logger.info("Received pitch transfer request")
        
        # Check if files exist in request
        if 'source_audio' not in request.files or 'target_audio' not in request.files:
            logger.error("Missing source_audio or target_audio in request")
            return jsonify({"error": "Missing source_audio or target_audio file"}), 400
        
        source_file = request.files['source_audio']
        target_file = request.files['target_audio']
        
        # Get parameters from request with defaults
        time_step = float(request.form.get('time_step', 0.01))  # Default to 0.01
        min_pitch = float(request.form.get('min_pitch', 75))
        max_pitch = float(request.form.get('max_pitch', 600))  # Default to 600
        resynthesis_method = request.form.get('resynthesis_method', 'psola')  # Default to psola
        voicing_threshold = float(request.form.get('voicing_threshold', 0.3))
        octave_cost = float(request.form.get('octave_cost', 0.01))
        octave_jump_cost = float(request.form.get('octave_jump_cost', 0.5))
        voiced_unvoiced_cost = float(request.form.get('voiced_unvoiced_cost', 0.14))
        preserve_formants = request.form.get('preserve_formants', 'True').lower() == 'true'
        
        logger.info(f"Received files: source={source_file.filename} ({source_file.content_type}), target={target_file.filename} ({target_file.content_type})")
        logger.info(f"Processing parameters: time_step={time_step}, min_pitch={min_pitch}, max_pitch={max_pitch}, resynthesis_method={resynthesis_method}, voicing_threshold={voicing_threshold}, octave_cost={octave_cost}, octave_jump_cost={octave_jump_cost}, voiced_unvoiced_cost={voiced_unvoiced_cost}, preserve_formants={preserve_formants}")
        
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
        
        temp_files.extend([source_path, target_path, output_path])
        
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
        
        # Process audio files with new parameters
        logger.info(f"Starting pitch transfer process: source={source_path}, target={target_path}, output={output_path}")
        success, message = transfer_pitch(
            source_path, 
            target_path, 
            output_path,
            time_step=time_step,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            resynthesis_method=resynthesis_method,
            voicing_threshold=voicing_threshold,
            octave_cost=octave_cost,
            octave_jump_cost=octave_jump_cost,
            voiced_unvoiced_cost=voiced_unvoiced_cost,
            preserve_formants=preserve_formants
        )
        
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
        
        # Set up cleanup function
        @after_this_request
        def cleanup(response):
            def do_cleanup():
                logger.info("Cleaning up temporary files")
                for file_path in temp_files:
                    safe_remove(file_path)
            
            # Schedule cleanup for after response is sent
            from threading import Timer
            Timer(1.0, do_cleanup).start()
            return response
        
        # Return the processed file
        return send_file(output_path, 
                        as_attachment=True, 
                        download_name="processed_audio.wav",
                        mimetype="audio/wav")
        
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        # Clean up files in case of error
        for file_path in temp_files:
            safe_remove(file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=2)
        logger.info("ffmpeg is available for audio conversion")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("ffmpeg is not available, will use soundfile for audio conversion")
    
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
