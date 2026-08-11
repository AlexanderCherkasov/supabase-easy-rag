# Enterprise Cloud Infrastructure Architecture

## Overview
Modern web applications rely on multi-region, resilient cloud deployments spanning Amazon Web Services (AWS), Google Cloud Platform (GCP), and Azure.

## Networking & Load Balancing
Application Load Balancers (ALB) distribute incoming HTTP/HTTPS traffic across auto-scaling groups of EC2 instances or ECS tasks. TLS termination occurs at the edge, utilizing ACM certificates with automatic 90-day rotations.

## Microservices Design Patterns
Services communicate via asynchronous event buses (RabbitMQ, Apache Kafka, or AWS EventBridge) for decoupled event-driven workflows. Synchronous communication is handled via gRPC over HTTP/2 for low latency internal service calls.

## Observability & Monitoring
Prometheus scrapes metrics endpoints every 15 seconds, emitting alerts to Alertmanager and Grafana dashboards for CPU, RAM, and Disk IOPS tracking.
