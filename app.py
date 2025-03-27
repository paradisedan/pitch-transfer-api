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

def convert_to_wav_with_ffmpeg(input_file, output_file):
    """Use ffmpeg to convert audio to WAV format."""
    try:
        logger.info(f"Converting {input_file} to WAV using ffmpeg")
        # Use ffmpeg to convert to standard WAV format
        cmd = [
            'ffmpeg', '-y', 
            '-i', input_file, 
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '44100',          # 44.1kHz sample rate
            '-ac', '1',              # Mono
            output_file
        ]
        
        # Run the ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            raise Exception(f"ffmpeg conversion failed: {result.stderr}")
        
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

def convert_to_wav_with_soundfile(input_file, output_file):
    """Convert audio to WAV format using soundfile."""
    try:
        logger.info(f"Converting {input_file} to WAV using soundfile")
        # Load audio file
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
        logger.error(f"Error converting file with soundfile {input_file}: {str(e)}")
        raise

def convert_to_wav(input_file, output_file=None):
    """Convert audio to WAV format using multiple methods."""
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.wav'
    
    # Try multiple methods to convert to WAV
    try:
        # First try with ffmpeg if available
        try:
            return convert_to_wav_with_ffmpeg(input_file, output_file)
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed, trying soundfile: {str(e)}")
        
        # If ffmpeg fails, try with soundfile
        return convert_to_wav_with_soundfile(input_file, output_file)
    except Exception as e:
        logger.error(f"All conversion methods failed for {input_file}: {str(e)}")
        raise

def save_sound_to_wav_direct(sound, output_file):
    """Save a Parselmouth Sound object to a WAV file using direct Parselmouth save."""
    try:
        # Save directly with Parselmouth
        logger.info(f"Saving sound directly with Parselmouth to {output_file}")
        sound.save(output_file, "WAV")
        
        # Verify the file was created successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Successfully saved WAV file: {output_file} ({os.path.getsize(output_file)} bytes)")
            return True
        else:
            logger.error(f"Failed to create WAV file at {output_file}")
            return False
    except Exception as e:
        logger.error(f"Error saving sound with Parselmouth: {str(e)}")
        return False

def save_sound_to_wav_wave(sound, output_file):
    """Save a Parselmouth Sound object to a WAV file using the wave module."""
    try:
        # Get values and parameters from the Sound object
        y = np.array(sound.values)
        sample_rate = int(sound.sampling_frequency)
        
        # Normalize the audio to 16-bit range
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val * 32767
        
        # Convert to 16-bit integers
        y = y.astype(np.int16)
        
        # Create a wave file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes for 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(y.tobytes())
        
        # Verify the file was created successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Successfully saved WAV file with wave module: {output_file} ({os.path.getsize(output_file)} bytes)")
            return True
        else:
            logger.error(f"Failed to create WAV file at {output_file}")
            return False
    except Exception as e:
        logger.error(f"Error saving sound with wave module: {str(e)}")
        return False

def save_sound_to_wav_soundfile(sound, output_file):
    """Save a Parselmouth Sound object to a WAV file using soundfile."""
    try:
        # Get values and parameters from the Sound object
        y = np.array(sound.values)
        sample_rate = int(sound.sampling_frequency)
        
        # Save using soundfile
        sf.write(output_file, y, sample_rate, subtype='PCM_16', format='WAV')
        
        # Verify the file was created successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Successfully saved WAV file with soundfile: {output_file} ({os.path.getsize(output_file)} bytes)")
            return True
        else:
            logger.error(f"Failed to create WAV file at {output_file}")
            return False
    except Exception as e:
        logger.error(f"Error saving sound with soundfile: {str(e)}")
        return False

def save_sound_to_wav(sound, output_file):
    """Save a Parselmouth Sound object to a WAV file using multiple methods."""
    # Try multiple methods to save the sound
    
    # First try with direct Parselmouth save
    if save_sound_to_wav_direct(sound, output_file):
        return True
    
    logger.warning("Direct Parselmouth save failed, trying wave module")
    
    # If direct save fails, try with wave module
    if save_sound_to_wav_wave(sound, output_file):
        return True
    
    logger.warning("Wave module save failed, trying soundfile")
    
    # If wave module fails, try with soundfile
    if save_sound_to_wav_soundfile(sound, output_file):
        return True
    
    # If all methods fail, try saving to a temporary file and using ffmpeg
    try:
        logger.warning("All direct save methods failed, trying with temporary file and ffmpeg")
        temp_file = os.path.join(os.path.dirname(output_file), f"temp_{uuid.uuid4()}.wav")
        
        # Try to save with any method to a temporary file
        if (save_sound_to_wav_direct(sound, temp_file) or 
            save_sound_to_wav_wave(sound, temp_file) or 
            save_sound_to_wav_soundfile(sound, temp_file)):
            
            # If successful, use ffmpeg to convert to a standard format
            convert_to_wav_with_ffmpeg(temp_file, output_file)
            safe_remove(temp_file)
            return True
    except Exception as e:
        logger.error(f"Temporary file and ffmpeg approach failed: {str(e)}")
    
    logger.error("All save methods failed")
    return False

def transfer_pitch(source_file, target_file, output_file, 
                  time_step=0.005, min_pitch=75, max_pitch=600,
                  resynthesis_method="overlap-add", voicing_threshold=0.4,
                  octave_cost=0.01, octave_jump_cost=0.5, voiced_unvoiced_cost=0.14,
                  preserve_formants=True):
    """Extract pitch from source_file and apply it to target_file."""
    source_wav = None
    target_wav = None
    
    try:
        # Validate parameters
        time_step = max(0.001, min(0.05, float(time_step)))  # Ensure time_step is within reasonable bounds
        min_pitch = max(50, min(300, float(min_pitch)))      # Ensure min_pitch is within reasonable bounds
        max_pitch = max(300, min(1000, float(max_pitch)))    # Ensure max_pitch is within reasonable bounds
        voicing_threshold = max(0.1, min(0.9, float(voicing_threshold)))
        octave_cost = max(0, min(0.1, float(octave_cost)))
        octave_jump_cost = max(0.1, min(1.0, float(octave_jump_cost)))
        voiced_unvoiced_cost = max(0.1, min(1.0, float(voiced_unvoiced_cost)))
        
        # Map resynthesis method to Praat command
        resynthesis_methods = {
            "overlap-add": "Get resynthesis (overlap-add)",
            "straight": "Get resynthesis (STRAIGHT)",
            "lpc": "Get resynthesis (LPC)"
        }
        resynthesis_command = resynthesis_methods.get(resynthesis_method.lower(), "Get resynthesis (overlap-add)")
        
        # Ensure source and target are different files
        if os.path.samefile(source_file, target_file):
            logger.error("Source and target files are the same")
            return False, "Source and target files are the same"
            
        # Convert to WAV if needed
        source_wav = convert_to_wav(source_file, os.path.join(os.path.dirname(source_file), f"source_{uuid.uuid4()}.wav"))
        target_wav = convert_to_wav(target_file, os.path.join(os.path.dirname(target_file), f"target_{uuid.uuid4()}.wav"))
        
        logger.info(f"Processing files: source={source_wav}, target={target_wav}")
        
        # Load sound files
        logger.info(f"Loading source sound from {source_wav}")
        source_sound = parselmouth.Sound(source_wav)
        logger.info(f"Source sound loaded: duration={source_sound.duration} seconds, sampling frequency={source_sound.sampling_frequency} Hz")
        
        logger.info(f"Loading target sound from {target_wav}")
        target_sound = parselmouth.Sound(target_wav)
        logger.info(f"Target sound loaded: duration={target_sound.duration} seconds, sampling frequency={target_sound.sampling_frequency} Hz")
        
        # Extract pitch from source with specified parameters
        logger.info(f"Extracting pitch from source with time_step={time_step}, min_pitch={min_pitch}, max_pitch={max_pitch}, voicing_threshold={voicing_threshold}")
        source_pitch = source_sound.to_pitch_ac(
            time_step=time_step, 
            pitch_floor=min_pitch, 
            pitch_ceiling=max_pitch,
            voicing_threshold=voicing_threshold,
            octave_cost=octave_cost,
            octave_jump_cost=octave_jump_cost,
            voiced_unvoiced_cost=voiced_unvoiced_cost
        )
        logger.info(f"Source pitch extracted: {source_pitch}")
        
        # Smooth the pitch contour for better quality
        logger.info("Smoothing pitch contour")
        try:
            smoothed_pitch = call(source_pitch, "Smooth...", 2)  # Bandwidth of 2 semitones
            logger.info("Pitch contour smoothed successfully")
            source_pitch = smoothed_pitch
        except Exception as e:
            logger.warning(f"Failed to smooth pitch contour: {str(e)}")
        
        # Create manipulation object with specified parameters
        logger.info(f"Creating manipulation object with time_step={time_step}, min_pitch={min_pitch}, max_pitch={max_pitch}")
        manipulation = call(target_sound, "To Manipulation", time_step, min_pitch, max_pitch)
        
        # Extract pitch tier from source pitch
        logger.info("Extracting pitch tier")
        pitch_tier = call(source_pitch, "Down to PitchTier")
        
        # Replace pitch in manipulation object
        logger.info("Replacing pitch tier")
        call([pitch_tier, manipulation], "Replace pitch tier")
        
        # If preserve_formants is True, try to maintain the original formants
        if preserve_formants:
            logger.info("Preserving formants using LPC analysis")
            try:
                if resynthesis_method.lower() == "lpc":
                    # LPC already preserves formants by design
                    logger.info("Using LPC resynthesis which inherently preserves formants")
                else:
                    # Extract LPC coefficients from target for later use
                    logger.info("Extracting LPC coefficients from target audio")
                    target_lpc = call(target_sound, "To LPC (autocorrelation)", 16, 0.025, 0.005, 50.0)
                    logger.info("Successfully extracted LPC coefficients")
            except Exception as e:
                logger.warning(f"Failed to prepare LPC analysis: {str(e)}")
        
        # Generate new sound with selected resynthesis method
        logger.info(f"Generating new sound using resynthesis method: {resynthesis_method}")
        try:
            new_sound = call(manipulation, resynthesis_command)
            logger.info(f"New sound generated with {resynthesis_method}: duration={new_sound.duration} seconds")
            
            # Apply LPC-based formant preservation if needed
            if preserve_formants and resynthesis_method.lower() != "lpc" and 'target_lpc' in locals():
                try:
                    logger.info("Applying LPC-based formant preservation")
                    # Apply the target formants using LPC filtering
                    new_sound = call([new_sound, target_lpc], "Filter")
                    logger.info("Successfully preserved formants using LPC filtering")
                except Exception as e:
                    logger.warning(f"Failed to apply LPC-based formant preservation: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to use {resynthesis_method} method: {str(e)}. Falling back to overlap-add.")
            new_sound = call(manipulation, "Get resynthesis (overlap-add)")
            logger.info(f"New sound generated with fallback method: duration={new_sound.duration} seconds")
            
            # Try to apply LPC-based formant preservation to the fallback sound
            if preserve_formants and 'target_lpc' in locals():
                try:
                    logger.info("Applying LPC-based formant preservation to fallback sound")
                    new_sound = call([new_sound, target_lpc], "Filter")
                    logger.info("Successfully preserved formants on fallback sound")
                except Exception as e:
                    logger.warning(f"Failed to apply LPC-based formant preservation to fallback sound: {str(e)}")
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the output using our multi-method approach
        logger.info(f"Saving output to {output_file}")
        if not save_sound_to_wav(new_sound, output_file):
            logger.error("All save methods failed")
            return False, "Failed to save output file"
        
        # Verify the file was created successfully
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.error(f"Failed to create output file at {output_file}")
            return False, f"Failed to create output file at {output_file}"
            
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
        time_step = float(request.form.get('time_step', 0.005))  # Default to 0.005
        min_pitch = float(request.form.get('min_pitch', 75))
        max_pitch = float(request.form.get('max_pitch', 600))
        resynthesis_method = request.form.get('resynthesis_method', 'overlap-add')  # Default to overlap-add
        voicing_threshold = float(request.form.get('voicing_threshold', 0.4))
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
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        logger.info("ffmpeg is available: " + result.stdout.split('\n')[0])
    except Exception as e:
        logger.warning(f"ffmpeg is not available: {str(e)}")
    
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
