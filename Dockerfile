# Step 3: Containerization.
#
# This packages the API + its exact dependency versions + a Python runtime
# into one portable image. "Works on my machine" stops being a problem
# because the container IS the machine, wherever it runs.

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (before the rest of the code) so Docker can cache
# this layer -- as long as requirements.txt doesn't change, rebuilds after
# code edits skip reinstalling every package, which is a big speedup.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest: serving code + the trained model artifacts.
# (model.pkl and feature_names.json must exist -- run `python train.py`
# locally before building the image.)
COPY app.py .
COPY model.pkl .
COPY feature_names.json .

EXPOSE 8000

# --host 0.0.0.0 is required inside a container -- 127.0.0.1 would only
# accept connections from inside the container itself, unreachable from outside.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
