# Pitch Transfer API

A Flask API that uses Parselmouth to transfer pitch from one audio file to another.

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /process` - Process audio files
  - Accepts two files: `source_audio` and `target_audio`
  - Returns the processed audio file
  - Accepts optional parameters:
    - `time_step`: Time step for pitch analysis in seconds (default: 0.005)
    - `min_pitch`: Minimum pitch in Hz (default: 75)
    - `max_pitch`: Maximum pitch in Hz (default: 600)
    - `resynthesis_method`: Method for resynthesis (default: "straight")
      - Options: "straight" (high quality), "overlap-add" (faster), "lpc" (preserves formants)

## Recommended Settings

- For speech-to-speech transfers with high quality: 
  - `time_step`: 0.005
  - `resynthesis_method`: "straight"

- For faster processing: 
  - `time_step`: 0.01
  - `resynthesis_method`: "overlap-add"

- For male voices:
  - `min_pitch`: 75
  - `max_pitch`: 300

- For female voices:
  - `min_pitch`: 100
  - `max_pitch`: 500

## Local Development

1. Install dependencies: