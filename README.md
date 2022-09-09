# Wind Farm Optimization Workshop

Real-time equipment optimization is a challenge for most field environments. Traditional industrial control systems rely on human intervention, and modern systems use the cloud for automation. However, these may still be prone to connectivity and latency limitations. In this workshop, learn how to use AWS IoT and Amazon Machine Learning services that can be deployed at the edge to adapt to changing environmental conditions in real time, without human intervention or the need for continuous connectivity. This automated feedback loop allows for faster decision making and responses to changing environment conditions that optimize performance and limit equipment downtime. You must bring your laptop to participate.

## Getting started

This repository contains the files and data required for the machine learning part of the AWS Wind Farm Optimization Workshop. If you launched the workshop from the AWS CloudFormation template, then you don't need to do anything else.

If you want to use this repository independently from the workshop, make sure to use a `ptorch_p38` kernel on Amazon SageMaker and to install the `dlr` library with the following script

```
pip install dlr
```

## Problem description

This workshop will simulate the use of intentional yaw angle misalignment in wind farms for cooperative control. We will use machine learning to create models of wake dynamics and power generation in a wind farm, and will use that model to find the configuration that will optimize total power generation.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.