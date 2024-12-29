FROM python:3.10.16

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && apt-get clean

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
COPY classifier_model.h5 /app/
COPY yolov5s.pt /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]