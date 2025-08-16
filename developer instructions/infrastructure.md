# GitHub Action 1: Static export
We have a GHA that runs for every commit to `main`. It runs all the notebooks using Pluto, generates a static html file for every notebook and publishes the output folder as our GitHub Pages website.

The action uses caching: if a notebook file is exactly the same (same file hash), then the notebook does not run, and the previous result is reused.

The result is https://bmlip.github.io/course/

This uses [**PlutoSliderServer.jl**](https://github.com/JuliaPluto/PlutoSliderServer.jl).

# GitHub Action 2: PR test
We have a GitHub action that runs on every PR commit. It runs the modified notebooks, and checks for unexpected changes compared to the `main` branch.

You can see the result as a "Check" on every PR.

This uses [**PlutoNotebookComparison.jl**](https://github.com/JuliaPluto/PlutoNotebookComparison.jl).


# Slider server (for live `@bind` interaction)
This whole part is optional, the website will work without it. 

We are running a PlutoSliderServer (PSS) to support live instant interactions with `@bind` on the website. This is also used by computationalthinking.mit.edu and featured.plutojl.org. Read the PSS README.md to learn more!

## SURF server
The server is hosted on **SURF Research Cloud**, they provided Fons with credits an account. Our contact is Han Verbiesen (ask Fons for his email, or contact hpcsupport@tue.nl). On the SURF dashboard: you need these things (create in this order):
- Account access (need to request this), Collaborative Organisation (we have one called "BIASLab BMLIP course development (Pluto.jl, Julia)") and a Wallet with credits (we started with 5000, valid until 31-03-2025).
- A reserved IP address. Currently 145.38.187.167
- A Workspace: I used "Ubuntu 22 SUDO", 2 core 16GB RAM, linked to the reserved IP.

## Domain
To get https we need a web domain. Not sure yet!

I set up an A record on the DNS of plutojl.org (cloudflare), so now we have:

https://bmlip-surfer.plutojl.org/ -> http://145.38.187.167/

## Setup
**The current setup is exactly the "Sample setup" from the PSS readme.**

The sample PSS setup works perfectly. The port 8080 is not available, so we have an nginx proxy from port 8080 -> 80 (also from the PSS readme sample instructions).



## Logging

We use BetterStack for system monitoring







I modified the `/usr/local/bin/pluto-slider-server.sh` script to use JSON logging:

```
#!/bin/bash

# this env var allows us to side step various issues with the Julia-bundled git
export JULIA_PKG_USE_CLI_GIT=true

cd /home/fplasvande/BMLIP-course
julia --project="pluto-slider-server-environment" -e "import Pkg; Pkg.instantiate(); using LoggingFormats, LoggingExtras; global_logger(FormatLogger(LoggingFormats.JSON(), stderr)); import PlutoSliderServer; PlutoSliderServer.run_git_directory(\".\")"
```







Vector.yaml:

```

# Forwarding logs to S1215127.eu-nbg-2.betterstackdata.com
# --------------------------------------------------------
# Generated on 2025-02-24: https://telemetry.betterstack.com/vector-yaml/ubuntu/TOKENHIDDEN
# Learn more about Vector configuration: https://vector.dev/docs/reference/configuration/

# - Nginx: v8

# ADDED
# journald service pluto-server

sources:
  
  better_stack_nginx_logs_TOKENHIDDEN:
    type: "file"
    read_from: "beginning"
    ignore_older_secs: 600
    include: ["/var/log/nginx/*.log"]
    exclude: []
  
  better_stack_other_TOKENHIDDEN:
    type: "file"
    read_from: "beginning"
    ignore_older_secs: 600
    include: ["/var/log/*.log","/var/log/**/*.log"]
    exclude: ["/var/log/apache/*.log","/var/log/nginx/*.log","/var/log/postgresql/*.log","/var/log/mysql/*.log","/var/log/mongodb/*.log","/var/log/redis/*.log","/var/log/auth.log","/var/log/ufw.log","/var/log/docker/*.log"]

  better_stack_host_metrics_TOKENHIDDEN:
    type: "host_metrics"
    scrape_interval_secs: 30
    collectors: [cpu, disk, filesystem, load, host, memory, network]
  
  better_stack_pluto_logs:
    type: "journald"
    journal_directory: "/var/log/journal"
    include_units: ["pluto-server.service"]


transforms:
  
  better_stack_nginx_parser_TOKENHIDDEN:
    type: "remap"
    inputs:
      - "better_stack_nginx_logs_TOKENHIDDEN"
    source: |
      del(.source_type)
      .dt = del(.timestamp)
      .nginx = parse_regex(.message, r'^\s*(-|(?P<client>\S+))\s+(-|(?P<remote_user>\S+))\s+(-|(?P<user>\S+))\s+\[(?P<timestamp>.+)\]\s+"(?P<request>-\s+-|(?P<method>\w+)\s+(?P<path>\S+)\s+(?P<protocol>\S+))"\s+(?P<status>\d+)\s+(?P<size>\d+)\s+"(-?|(?P<referrer>.+))"\s+"(-?|(?P<agent>.+))"\s*') ??
          parse_regex(.message, r'^\s*(?P<timestamp>.+)\s+\[(?P<severity>\w+)\]\s+(?P<pid>\d+)\#(?P<tid>\d+):\s+\*(?P<cid>\d+)\s+(?P<message>.*)(?:,\s+client:\s+(?P<client>[^,z]+))(?:,\s+server:\s+(?P<server>[^,z]+))(?:,\s+request:\s+"(?P<request>[^"]+)")(?:,\s+subrequest:\s+"(?P<subrequest>[^"]+)")?(?:,\s+upstream:\s+"(?P<upstream>[^"]+)")?(?:,\s+host:\s+"(?P<host>[^"]+)")(?:,\s+referrer:\s+"(?P<referrer>[^"]+)")?\s*') ??
          parse_nginx_log(.message, format: "combined") ??
          parse_nginx_log(.message, format: "error") ??
          {}
      
      if .nginx != {} {
        .platform = "Nginx"
        .level = del(.nginx.severity)
        .message = del(.nginx.message)
      
        if is_null(.message) { del(.message) }
        if exists(.nginx.timestamp) {
          .dt = format_timestamp!(
            parse_timestamp(.nginx.timestamp, "%d/%b/%Y:%T %z") ??
              parse_timestamp(.nginx.timestamp, "%Y/%m/%d %T") ??
              .dt,
            "%+"
          )
      
          del(.nginx.timestamp)
        }
      
        if is_string(.nginx.status) { .nginx.status = to_int(.nginx.status) ?? .nginx.status }
        if is_string(.nginx.size) { .nginx.size = to_int(.nginx.size) ?? .nginx.size }
        if is_string(.nginx.cid) { .nginx.cid = to_int(.nginx.cid) ?? .nginx.cid }
        if is_string(.nginx.pid) { .nginx.pid = to_int(.nginx.pid) ?? .nginx.pid }
        if is_string(.nginx.tid) { .nginx.tid = to_int(.nginx.tid) ?? .nginx.tid }
      
        if is_null(.nginx.remote_user) { del(.nginx.remote_user) }
        if is_null(.nginx.user) { del(.nginx.user) }
        if is_null(.nginx.subrequest) { del(.nginx.subrequest) }
        if is_null(.nginx.upstream) { del(.nginx.upstream) }
        if is_null(.nginx.referrer) { del(.nginx.referrer) }
        if is_null(.nginx.agent) { del(.nginx.agent) }
      } else {
        del(.nginx)
      }
  
  better_stack_other_parser_TOKENHIDDEN:
    type: "remap"
    inputs:
      - "better_stack_other_TOKENHIDDEN"
    source: |
      del(.source_type)
      .dt = del(.timestamp)


sinks:
  
  better_stack_http_sink_TOKENHIDDEN:
    type: "http"
    method: "post"
    uri: "https://s1215127.eu-nbg-2.betterstackdata.com/"
    encoding:
      codec: "json"
    compression: "gzip"
    auth:
      strategy: "bearer"
      token: "TOKENHIDDEN"
    inputs: [
      "better_stack_nginx_parser_TOKENHIDDEN",
      "better_stack_other_parser_TOKENHIDDEN",
      "better_stack_pluto_logs"]

  better_stack_http_metrics_sink_TOKENHIDDEN:
    type: "http"
    method: "post"
    uri: "https://s1215127.eu-nbg-2.betterstackdata.com/metrics"
    encoding:
      codec: "json"
    compression: "gzip"
    auth:
      strategy: "bearer"
      token: "TOKENHIDDEN"
    inputs: [
      "better_stack_host_metrics_TOKENHIDDEN"
      ]


# --- end of 2025-02-24: https://telemetry.betterstack.com/vector-yaml/ubuntu/TOKENHIDDEN

```






