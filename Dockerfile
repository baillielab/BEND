FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /

COPY  . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install BEND
RUN pip install -e .
