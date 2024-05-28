# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container at /app
COPY . /app/

# Set environment variables for wandb
ENV WANDB_API_KEY=833a9c73668aca67fd18dd8a303891e920c83afe

# Define the command to run your application
CMD ["python", "src/models/ABBV_StockPrediction1.py"]
