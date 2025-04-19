# Use a base image with Python and GPU support (for CUDA-enabled containers)
FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && pip3 install --upgrade pip

# Copy the local files to the container
COPY . /app

# Install the Python dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
