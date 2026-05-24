#!/bin/bash

echo "Starting MLflow server..."
echo "UI will be available at: http://127.0.0.1:8080"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""

mlflow server --host 127.0.0.1 --port 8080
