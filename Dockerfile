# Use a base image with Python and GPU support for CUDA 12.8 containers
FROM nvidia/cuda:12.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., Python3, pip)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && pip3 install --upgrade pip \
    && apt-get clean

# Copy the local files to the container
COPY . /app

# Install the Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose the port for FastAPI to listen on
EXPOSE 8000

# Run the FastAPI app using Uvicorn with multiple workers for better performance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
