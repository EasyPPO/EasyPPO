FROM ubuntu:24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/*; \
        if apt-get update -o Acquire::Retries=5 && \
            apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
                g++ python3 python3-pip curl ca-certificates ripgrep; then \
            break; \
        fi; \
        if [ "$attempt" = "5" ]; then \
            exit 1; \
        fi; \
        sleep $((attempt * 20)); \
    done; \
    pip3 install --break-system-packages requests && \
    rm -rf /var/lib/apt/lists/*

# Pre-install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash -s -- && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Pre-install Codex CLI (needs Node.js via nvm)
ENV NVM_DIR="/root/.nvm"
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install 22 && nvm alias default 22 && \
    npm install -g @openai/codex@latest && \
    ln -sf "$(which node)" /usr/local/bin/node && \
    ln -sf "$(which codex)" /usr/local/bin/codex

ENV CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000

WORKDIR /app
