#!/bin/bash
# Start the F1 Prediction API

cd "/Users/mehulchandna/CS Coursework/CS4342/pitwall-prophet"

echo "Starting Pitwall Prophet API..."
echo "This will take ~60 seconds to train the model..."
echo ""

python3 api.py

echo "API stopped."
