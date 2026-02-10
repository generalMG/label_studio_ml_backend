# Use PyTorch with CUDA support as base
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
# We install instructions separately to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV MODEL_DIR=/data/models
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose the port the app runs on
EXPOSE 9090

# Command to run the application
# We use the _wsgi.py script which initializes the app
CMD ["python", "label_studio_backend/_wsgi.py", "--port", "9090", "--host", "0.0.0.0"]
