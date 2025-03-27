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
    - `resynthesis_method`: Method for resynthesis (default: "overlap-add")
      - Options: "overlap-add" (standard method), "lpc" (preserves formants), "straight" (if available)
    - `voicing_threshold`: Threshold for voiced/unvoiced decision (default: 0.4)
    - `octave_cost`: Cost for octave jumps in pitch analysis (default: 0.01)
    - `octave_jump_cost`: Cost for octave jumps between adjacent frames (default: 0.5)
    - `voiced_unvoiced_cost`: Cost for voiced/unvoiced transitions (default: 0.14)
    - `preserve_formants`: Whether to preserve the target's formant structure (default: true)

## Recommended Settings

- For speech-to-speech transfers with high quality (default settings): 
  - `time_step`: 0.005
  - `resynthesis_method`: "overlap-add"
  - `voicing_threshold`: 0.4
  - `octave_jump_cost`: 0.5
  - `preserve_formants`: true

- For faster processing: 
  - `time_step`: 0.01
  - `resynthesis_method`: "overlap-add"
  - `preserve_formants`: false

- For male voices:
  - `min_pitch`: 75
  - `max_pitch`: 300

- For female voices:
  - `min_pitch`: 100
  - `max_pitch`: 500

## Notes on Resynthesis Methods

- **overlap-add**: Standard PSOLA method available in all Praat installations
- **lpc**: Linear Predictive Coding, good for preserving formants
- **straight**: Advanced method that may not be available in standard Praat installations. If specified but not available, the API will automatically fall back to overlap-add.

## Quality Improvements

The current implementation includes several features to improve the naturalness of speech-to-speech transfers:

1. **Formant Preservation**: 
   - Preserves the vocal characteristics of the target speaker
   - Reduces "robotic" qualities in the output
   - Maintains more natural timbre while changing pitch

2. **Advanced Pitch Analysis**:
   - `voicing_threshold`: Controls which parts of the signal are considered voiced (lower values = more voiced segments)
   - `octave_cost`: Penalizes pitch candidates that are far from the median pitch (higher values = more stable pitch)
   - `octave_jump_cost`: Penalizes large jumps in pitch between adjacent frames (higher values = smoother pitch contours)
   - `voiced_unvoiced_cost`: Penalizes transitions between voiced and unvoiced segments (higher values = fewer transitions)

## Troubleshooting

- For fixing "robotic" sounding results:
  - `voicing_threshold`: 0.4 (lower for more voiced segments)
  - `octave_cost`: 0.05 (higher to prevent octave jumps)
  - `octave_jump_cost`: 0.5 (higher for smoother pitch contours)
  - `preserve_formants`: true

- For fixing "mumbled" sounding results:
  - `voicing_threshold`: 0.5 (higher for clearer voiced/unvoiced distinction)
  - `voiced_unvoiced_cost`: 0.2 (higher to prevent rapid voiced/unvoiced transitions)

## Local Development

1. Install dependencies: