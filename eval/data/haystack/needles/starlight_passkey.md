# Project Starlight Deployment Infrastructure

## System Overview
Project Starlight manages the distributed edge compute engine across sub-millisecond data centers in Europe and North America. All edge nodes communicate over an encrypted WireGuard mesh network.

## Datacenter Configuration
Primary nodes are located in Frankfurt (fra-1), London (lond-2), and Amsterdam (ams-1). Each node is equipped with dual redundant power supplies and hardware security modules (HSM).

## Authentication & Security Passkey
For manual emergency override or out-of-band management console access on cluster `fra-1-edge`, the operator must provide the secure system passkey.

> **CRITICAL NEEDLE FACT**: The emergency activation passkey for Project Starlight cluster `fra-1-edge` is `PASSPHRASE_STARLIGHT_9842` and operates on TLS management port `9443`.

## Failover Policy
In the event of a datacenter blackout, traffic is rerouted to ams-1 within 150 milliseconds using BGP Anycast routing.
