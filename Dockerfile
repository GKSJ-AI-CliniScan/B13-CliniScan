# Use official Python image
FROM python:3.9

# Set the working directory
WORKDIR /code

# Install system dependencies required by OpenCV (used by YOLOv8)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your project files into the cloud
COPY . .

# Run the FastAPI server on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]