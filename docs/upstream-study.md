# Model Deck: upstream project study

Containerized serverless ML inference templates.

**Deepak Gangwar’s documentation for a local study copy of [SeamlessML](https://github.com/newtonjain/Docker_Hackathon).** The portfolio display name has changed; the upstream implementation and authorship have not. Original authors: newtonjain/Docker_Hackathon contributors.

## At a glance

| Item | Status |
|---|---|
| Purpose | Architecture study and preparation for an attributed extension |
| Current work in this copy | Portfolio naming and recruiter-oriented documentation, prepared with AI assistance; no upstream application-code changes |
| Application code changes | None |
| Runtime verification | Not performed for this clone |
| Hosted demo | No deployment of this clone; previous similarly named demos are separate implementations |
| Upstream snapshot | `d6db83f2a0307dfe82da7dfa46891d79f15fb1ac` |

## Problem and approach

Moving a trained model into a callable endpoint requires packaging its runtime, deploying infrastructure and validating request handling.

```text
Training notebook → packaged model + inference handler → Docker image → AWS SAM infrastructure → Lambda API endpoint
```

**Technology:** Python, Docker, AWS Lambda, AWS SAM, scikit-learn, XGBoost, TensorFlow and PyTorch.

## Explore the repository

| Path | What to inspect |
|---|---|
| [Scikit-learn Machine Learning Inference API](https://github.com/newtonjain/Docker_Hackathon/tree/d6db83f2a0307dfe82da7dfa46891d79f15fb1ac/Scikit-learn%20Machine%20Learning%20Inference%20API) | Scikit-learn inference example |
| [XGBoost Machine Learning Inference API](https://github.com/newtonjain/Docker_Hackathon/tree/d6db83f2a0307dfe82da7dfa46891d79f15fb1ac/XGBoost%20Machine%20Learning%20Inference%20API) | XGBoost inference example |
| [Tensorflow Machine Learning Inference API](https://github.com/newtonjain/Docker_Hackathon/tree/d6db83f2a0307dfe82da7dfa46891d79f15fb1ac/Tensorflow%20Machine%20Learning%20Inference%20API) | TensorFlow inference example |
| [PyTorch Machine Learning Inference API](https://github.com/newtonjain/Docker_Hackathon/tree/d6db83f2a0307dfe82da7dfa46891d79f15fb1ac/PyTorch%20Machine%20Learning%20Inference%20API) | PyTorch inference example |

## Setup and demonstration status

Choose one framework directory and read its README.md, app/Dockerfile, app/requirements.txt and template.yaml before building. AWS SAM CLI, Docker and an appropriately configured AWS account are required for cloud deployment. Inspect existing samconfig.toml values before using them in your environment. No AWS resources were created for this clone; cloud deployment can incur charges.

## What I can discuss with a recruiter

**Relevant areas:** Python inference services, model packaging, Docker, infrastructure as code, cloud deployment and MLOps concepts.

Explain the handler contract, artifact loading, local versus cloud testing, Lambda cold starts and how you would monitor latency and failed predictions.

An accurate introduction is: “I am studying SeamlessML under the working portfolio name Model Deck. I have organized its documentation; the implementation is upstream. I will identify and demonstrate my own engineering changes separately as they are completed.”

## Scope and evidence

These are upstream framework examples, not the separately deployed Model Deck registry/dashboard. Existing model artifacts are upstream files and have not been retrained or independently evaluated here. Historical upstream deployment-speed claims are not measurements from this copy.

Renaming and README editing do not establish implementation experience, benchmark results or a working deployment. Upstream awards belong to the original team. This upstream copy should be described as a study copy until substantive changes are implemented and verified. The separate application in this repository has its own implementation and verification record in the root README.

## Attribution and reuse

Original source: [SeamlessML](https://github.com/newtonjain/Docker_Hackathon). Original authors: newtonjain/Docker_Hackathon contributors. Upstream authorship and applicable notices are recorded here; the upstream source is not included in this repository.

No clear reuse licence was found in the inspected files. This is a local reference copy; public redistribution needs appropriate permission.

## Relationship to this repository

This document describes the original upstream reference, not the application deployed from this repository. Only study documentation is published here; the upstream source is not included in this repository. Use the [repository README](../README.md) for the current application, live demo, tests and contribution details.
