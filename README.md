# sm-capacity-pool

Hold one GPU instance in SageMaker using a real-time endpoint whose single variant
uses **instance pools** (`InstancePools` on the endpoint config). SageMaker itself
walks the pools in priority order on insufficient capacity, for up to
`VariantInstanceProvisionTimeoutInSeconds` (3600 s, the maximum):

```
ml.g6.2xlarge -> ml.g6e.2xlarge -> ml.g7.2xlarge -> ml.g7e.2xlarge
```

No HyperPod, no EC2 Auto Scaling group. Once the endpoint is `InService`,
Application Auto Scaling is registered on the variant (min 1 / max 1); any later
scale-out walks the same pool order.

## What gets created

| Resource | Name |
|---|---|
| IAM execution role (least-privilege inline policy) | `SageMakerGpuCapacityPoolRole` |
| S3 bucket + placeholder `model.tar.gz` | `sagemaker-gpu-capacity-pool-<account>-<region>` |
| SageMaker Model (PyTorch inference DLC, echo handler) | `gpu-capacity-pool` |
| Endpoint config with four `InstancePools` | `gpu-capacity-pool` |
| Endpoint | `gpu-capacity-pool` |

The placeholder model (`model/code/inference.py`) loads no weights; it only
answers `/ping` so the endpoint can reach `InService` and hold the instance.

## Prerequisites

- AWS credentials for the target account in your shell (Admin or equivalent).
- Service quota `ml.<type>.2xlarge for endpoint usage` >= 1 for each pool type.
- boto3 >= 1.43 (`InstancePools` is not in older SDKs / AWS CLI v1).

```
python -m venv venv && venv/bin/pip install boto3
```

Region defaults to `us-west-2`; override with `AWS_REGION`. The account ID is
read from STS.

## Usage

```
venv/bin/python setup_pool.py            # dry run, creates nothing
venv/bin/python setup_pool.py --go       # create whatever is missing
venv/bin/python setup_pool.py --status   # endpoint status + per-pool counts
```

`watch_pool.py` is the outer loop. SageMaker only walks the pools for one hour
per `CreateEndpoint`; if every pool stays dry that long the endpoint lands in
`Failed`. The watcher deletes it and issues a fresh `CreateEndpoint`, repeating
until a GPU is held. On `InService` it registers autoscaling and exits.

```
venv/bin/python watch_pool.py --loop                 # poll every 10 min until a GPU is held
venv/bin/python watch_pool.py --loop --interval 300  # poll every 5 min
venv/bin/python watch_pool.py                        # one cycle, for cron / EventBridge
```

Exit codes: `10` = done (InService), `1` = needs a human (non-capacity failure
or expired credentials), `0` = still waiting (single-cycle mode only).

Typical sequence:

```
venv/bin/python setup_pool.py --go && venv/bin/python watch_pool.py --loop
```

`--loop` runs for as long as it takes, so start it under `nohup`, `tmux`, or a
scheduler if the shell may close. Credentials must outlive the wait; a short-lived
session token will stop the loop with exit 1 when it expires.

## Notes

- The SageMaker console does not yet render `InstancePools`; the endpoint-config
  page shows `-` for instance type. Verify with
  `aws sagemaker describe-endpoint-config --endpoint-config-name gpu-capacity-pool`
  or the `CreateEndpointConfig` event in CloudTrail.
- The instance bills from the moment it is procured. Delete the endpoint to stop;
  the config and model stay reusable.
