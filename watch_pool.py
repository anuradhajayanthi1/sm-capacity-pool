#!/usr/bin/env python
"""Outer retry loop for the gpu-capacity-pool endpoint.

SageMaker walks the instance pools for at most 3600 s per CreateEndpoint. If
every pool is dry for that long the endpoint lands in Failed, so this script
deletes it and issues a fresh CreateEndpoint, until a GPU is actually held.

One cycle:
  Creating/Updating -> print status, exit 0 (keep waiting)
  Failed            -> print reason; if capacity-related, delete the endpoint,
                       wait for deletion, recreate from the same config; exit 0
  InService         -> register Application Auto Scaling (min 1, max 1) if not
                       already registered; print which pool won; exit 10 (DONE)
  other failure     -> print and exit 1 (needs a human)

Usage:
    python watch_pool.py                      # one cycle (for cron / EventBridge)
    python watch_pool.py --loop               # repeat every 10 min until DONE or a
                                              # non-capacity failure
    python watch_pool.py --loop --interval 300
"""
import argparse
import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-west-2")
NAME = "gpu-capacity-pool"
RESOURCE_ID = f"endpoint/{NAME}/variant/AllTraffic"
DIMENSION = "sagemaker:variant:DesiredInstanceCount"
CAPACITY_MARKERS = ("insufficient capacity", "capacity", "InsufficientInstanceCapacity")

sm = boto3.client("sagemaker", region_name=REGION)
aas = boto3.client("application-autoscaling", region_name=REGION)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def describe():
    return sm.describe_endpoint(EndpointName=NAME)


def pools(d):
    for v in d.get("ProductionVariants", []):
        return {"current": v.get("CurrentInstanceCount"), "desired": v.get("DesiredInstanceCount"),
                "pools": v.get("InstancePools")}
    return {}


def recreate(reason):
    log(f"deleting failed endpoint (reason: {reason})")
    sm.delete_endpoint(EndpointName=NAME)
    for _ in range(60):
        time.sleep(10)
        try:
            describe()
        except ClientError as e:
            if "Could not find" in str(e):
                break
            raise
    else:
        log("endpoint still present after 10 min; will retry next cycle")
        return
    sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME,
                       Tags=[{"Key": "purpose", "Value": "gpu-capacity-pool"}])
    log("RECREATED: new CreateEndpoint issued, pools will be retried for up to 3600s")


def ensure_autoscaling():
    existing = aas.describe_scalable_targets(ServiceNamespace="sagemaker", ResourceIds=[RESOURCE_ID],
                                             ScalableDimension=DIMENSION)["ScalableTargets"]
    if existing:
        t = existing[0]
        log(f"autoscaling already registered: min={t['MinCapacity']} max={t['MaxCapacity']}")
        return
    aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=RESOURCE_ID,
                                 ScalableDimension=DIMENSION, MinCapacity=1, MaxCapacity=1)
    log("autoscaling registered: min=1 max=1 (scale-out walks pools g6->g6e->g7->g7e)")


def main():
    try:
        d = describe()
    except ClientError as e:
        if "Could not find" in str(e):
            log("endpoint does not exist; recreating from config")
            sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
            return 0
        raise
    status = d["EndpointStatus"]
    log(f"status={status} {json.dumps(pools(d), default=str)}")
    if status == "InService":
        ensure_autoscaling()
        won = [p for p in (pools(d).get("pools") or []) if p.get("CurrentInstanceCount", 0) > 0]
        log(f"DONE: InService, procured {json.dumps(won, default=str)}")
        return 10
    if status == "Failed":
        reason = d.get("FailureReason", "")
        if any(m.lower() in reason.lower() for m in CAPACITY_MARKERS):
            recreate(reason)
            return 0
        log(f"NON-CAPACITY FAILURE, needs a human: {reason}")
        return 1
    return 0


def loop(interval):
    cycle = 0
    while True:
        cycle += 1
        log(f"--- cycle {cycle} ---")
        try:
            rc = main()
        except ClientError as e:
            # Throttling or a transient API error: keep going. Expired
            # credentials are not transient; stop so a human can refresh them.
            if e.response.get("Error", {}).get("Code") in ("ExpiredToken", "ExpiredTokenException"):
                log(f"credentials expired, stopping: {e}")
                return 1
            log(f"transient error, retrying next cycle: {e}")
            rc = 0
        if rc != 0:
            return rc
        log(f"sleeping {interval}s")
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", action="store_true", help="repeat until InService or a non-capacity failure")
    ap.add_argument("--interval", type=int, default=600, help="seconds between cycles with --loop (default 600)")
    args = ap.parse_args()
    sys.exit(loop(args.interval) if args.loop else main())
