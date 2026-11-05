#!/usr/bin/env python
"""Create a SageMaker real-time endpoint whose single variant uses InstancePools
(g6 -> g6e -> g7 -> g7e, all 2xlarge) so SageMaker itself falls back on
insufficient capacity.

Usage:
    python setup_pool.py            # dry run: print plan, create nothing
    python setup_pool.py --go       # create everything that is missing
    python setup_pool.py --status   # describe endpoint + per-pool counts

Idempotent: every step checks for the resource first.
"""
import io
import json
import os
import sys
import tarfile
import time

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-west-2")
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
NAME = "gpu-capacity-pool"
ROLE_NAME = "SageMakerGpuCapacityPoolRole"
BUCKET = f"sagemaker-{NAME}-{ACCOUNT}-{REGION}"
ARTIFACT_KEY = f"{NAME}/model.tar.gz"
# PyTorch 2.6 GPU inference DLC (763104351884 is the public DLC account).
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker"
POOLS = ["ml.g6.2xlarge", "ml.g6e.2xlarge", "ml.g7.2xlarge", "ml.g7e.2xlarge"]
PROVISION_TIMEOUT = 3600  # max allowed; SageMaker cycles the pools for this long
TAGS = [{"Key": "purpose", "Value": "gpu-capacity-pool"}, {"Key": "owner", "Value": os.environ.get("USER", "unknown")}]

sm = boto3.client("sagemaker", region_name=REGION)
iam = boto3.client("iam", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_role(go):
    try:
        arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        log(f"role exists: {arn}")
        return arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
    arn = f"arn:aws:iam::{ACCOUNT}:role/{ROLE_NAME}"
    log(f"role missing, {'creating' if go else 'would create'}: {arn}")
    if not go:
        return arn
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}},
        }],
    }
    # Least privilege: read the one artifact, pull the DLC image, write endpoint logs/metrics.
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject"],
             "Resource": f"arn:aws:s3:::{BUCKET}/{ARTIFACT_KEY}"},
            {"Effect": "Allow", "Action": ["s3:ListBucket"],
             "Resource": f"arn:aws:s3:::{BUCKET}"},
            {"Effect": "Allow",
             "Action": ["ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
             "Resource": "*"},
            {"Effect": "Allow",
             "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                        "logs:DescribeLogStreams"],
             "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/sagemaker/Endpoints/{NAME}*"},
            {"Effect": "Allow", "Action": ["cloudwatch:PutMetricData"], "Resource": "*",
             "Condition": {"StringEquals": {"cloudwatch:namespace": ["AWS/SageMaker", "/aws/sagemaker/Endpoints"]}}},
        ],
    }
    iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                    Description="Execution role for the GPU capacity-pool placeholder endpoint",
                    Tags=TAGS)
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="placeholder-endpoint", PolicyDocument=json.dumps(policy))
    log("role created; waiting 15s for IAM propagation")
    time.sleep(15)
    return arn


def ensure_artifact(go):
    uri = f"s3://{BUCKET}/{ARTIFACT_KEY}"
    try:
        s3.head_bucket(Bucket=BUCKET)
        log(f"bucket exists: {BUCKET}")
    except ClientError:
        log(f"bucket missing, {'creating' if go else 'would create'}: {BUCKET}")
        if go:
            s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
            s3.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True,
                "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    try:
        s3.head_object(Bucket=BUCKET, Key=ARTIFACT_KEY)
        log(f"artifact exists: {uri}")
        return uri
    except ClientError:
        pass
    log(f"artifact missing, {'uploading' if go else 'would upload'}: {uri}")
    if go:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add("model/code/inference.py", arcname="code/inference.py")
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=ARTIFACT_KEY, Body=buf.getvalue())
    return uri


def ensure_model(go, role_arn, artifact):
    try:
        sm.describe_model(ModelName=NAME)
        log(f"model exists: {NAME}")
        return
    except ClientError as e:
        if "Could not find" not in str(e):
            raise
    log(f"model missing, {'creating' if go else 'would create'}: {NAME} image={IMAGE}")
    if go:
        sm.create_model(
            ModelName=NAME, ExecutionRoleArn=role_arn, Tags=TAGS,
            PrimaryContainer={
                "Image": IMAGE, "ModelDataUrl": artifact,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": REGION,
                },
            },
        )


def ensure_endpoint_config(go):
    try:
        sm.describe_endpoint_config(EndpointConfigName=NAME)
        log(f"endpoint config exists: {NAME}")
        return
    except ClientError as e:
        if "Could not find" not in str(e):
            raise
    variant = {
        "VariantName": "AllTraffic",
        "ModelName": NAME,
        "InitialInstanceCount": 1,
        "InstancePools": [{"InstanceType": t, "Priority": i + 1} for i, t in enumerate(POOLS)],
        "VariantInstanceProvisionTimeoutInSeconds": PROVISION_TIMEOUT,
    }
    log(f"endpoint config missing, {'creating' if go else 'would create'}:\n{json.dumps(variant, indent=2)}")
    if go:
        sm.create_endpoint_config(EndpointConfigName=NAME, ProductionVariants=[variant], Tags=TAGS)


def ensure_endpoint(go):
    try:
        d = sm.describe_endpoint(EndpointName=NAME)
        log(f"endpoint exists: {NAME} status={d['EndpointStatus']}")
        return d
    except ClientError as e:
        if "Could not find" not in str(e):
            raise
    log(f"endpoint missing, {'creating' if go else 'would create'}: {NAME}")
    if go:
        sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME, Tags=TAGS)
        return sm.describe_endpoint(EndpointName=NAME)


def status():
    try:
        d = sm.describe_endpoint(EndpointName=NAME)
    except ClientError as e:
        if "Could not find" in str(e):
            print(json.dumps({"EndpointName": NAME, "EndpointStatus": "NotFound"}, indent=2))
            return None
        raise
    out = {"EndpointName": d["EndpointName"], "EndpointStatus": d["EndpointStatus"],
           "FailureReason": d.get("FailureReason"), "CreationTime": str(d["CreationTime"]),
           "LastModifiedTime": str(d["LastModifiedTime"])}
    for v in d.get("ProductionVariants", []):
        out["Variant"] = {
            "CurrentInstanceCount": v.get("CurrentInstanceCount"),
            "DesiredInstanceCount": v.get("DesiredInstanceCount"),
            "InstancePools": v.get("InstancePools"),
        }
    print(json.dumps(out, indent=2, default=str))
    return d


def main():
    go = "--go" in sys.argv
    if "--status" in sys.argv:
        status()
        return
    log(f"account={boto3.client('sts').get_caller_identity()['Account']} region={REGION} mode={'CREATE' if go else 'DRY RUN'}")
    role_arn = ensure_role(go)
    artifact = ensure_artifact(go)
    ensure_model(go, role_arn, artifact)
    ensure_endpoint_config(go)
    ensure_endpoint(go)
    if go:
        log("creation requested; use --status to watch")


if __name__ == "__main__":
    main()
