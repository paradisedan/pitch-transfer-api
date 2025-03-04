
# Pitch Transfer API

A Flask API that uses Parselmouth to transfer pitch from one audio file to another.

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /process` - Process audio files
  - Accepts two files: `source` and `target`
  - Returns the processed audio file

## Local Development

1. Install dependencies: