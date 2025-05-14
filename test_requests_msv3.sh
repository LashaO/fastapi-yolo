#!/bin/bash

URL="http://localhost:8000/predict/?model_id=msv3&image_url=https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Green_sea_turtle_%28Chelonia_mydas%29_Moorea.jpg/500px-Green_sea_turtle_%28Chelonia_mydas%29_Moorea.jpg"
N=${1:-5}  # Default to 5 if no argument provided

time (
  for ((i = 1; i <= N; i++)); do
    curl -s -X POST "$URL" &
  done
  wait
)
