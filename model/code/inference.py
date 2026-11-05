"""Placeholder handler for the GPU capacity-pool endpoint.

The endpoint exists to hold one GPU instance. This handler loads nothing and
echoes the request, so the container passes the /ping health check and the
endpoint reaches InService without a real model artifact.
"""
import json


def model_fn(model_dir):
    return {"placeholder": True, "model_dir": model_dir}


def input_fn(request_body, content_type="application/json"):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    try:
        return json.loads(request_body)
    except (TypeError, ValueError):
        return {"raw": request_body}


def predict_fn(data, model):
    return {"echo": data, "placeholder": True}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
