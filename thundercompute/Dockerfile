FROM ubuntu:24.04

ARG TNR_VERSION=2.0.71

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/thunder \
    PATH=/home/thunder/.tnr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TNR_VERSION=${TNR_VERSION}

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        bash \
        ca-certificates \
        coreutils \
        curl \
        gzip \
        openssh-client \
        python3 \
        tar \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash thunder \
    && mkdir -p /home/thunder/.tnr/bin \
    && chown -R thunder:thunder /home/thunder

USER thunder

# The official installer verifies the downloaded release checksum. Override
# TNR_VERSION at build time only when the laptop's CLI is upgraded and the
# command contract has been re-verified.
RUN curl -fsSL https://raw.githubusercontent.com/Thunder-Compute/thunder-cli/main/scripts/install.sh | bash

WORKDIR /opt/hcmai

COPY --chown=thunder:thunder thundercompute/launch.sh /usr/local/bin/hcmai-thundercompute
COPY --chown=thunder:thunder thundercompute/delete.sh /usr/local/bin/hcmai-thunder-delete

RUN chmod 0755 /usr/local/bin/hcmai-thundercompute /usr/local/bin/hcmai-thunder-delete

ENTRYPOINT ["/usr/local/bin/hcmai-thundercompute"]
