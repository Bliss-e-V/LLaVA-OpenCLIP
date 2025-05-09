#syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked set -eux; \
	apt-get update -qq && \
	apt-get install -qqy --no-install-recommends curl; \
	rm -rf /var/lib/apt/lists/*; \
	TINI_VERSION=v0.19.0; \
	TINI_ARCH="$(dpkg --print-architecture)"; \
	curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
	chmod +x /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update -qq && apt-get install -qqy --no-install-recommends \
	make \
	build-essential \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	wget \
	curl \
	llvm \
	libncurses5-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libffi-dev \
	liblzma-dev \
	git \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
	git clone https://github.com/momo-lab/pyenv-install-latest.git "$(pyenv root)"/plugins/pyenv-install-latest && \
	export PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' && \
	export PYTHON_CFLAGS='-O3' && \
	pyenv install-latest "3.11" && \
	pyenv global $(pyenv install-latest --print "3.11") && \
	pip install "wheel<1"
RUN rm -rf /usr/bin/python3 && ln -s `realpath \`pyenv which python\`` /usr/bin/python3 && chmod +x /usr/bin/python3
COPY .cog/tmp/build20250414131725.701717/requirements.txt /tmp/requirements.txt
ENV CFLAGS="-O3 -funroll-loops -fno-strict-aliasing -flto -S"
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt
ENV CFLAGS=
COPY .cog/tmp/build20250414131725.701717/cog-0.14.4-py3-none-any.whl /tmp/cog-0.14.4-py3-none-any.whl
ENV CFLAGS="-O3 -funroll-loops -fno-strict-aliasing -flto -S"
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir /tmp/cog-0.14.4-py3-none-any.whl 'pydantic<2'
ENV CFLAGS=
RUN find / -type f -name "*python*.so" -printf "%h\n" | sort -u > /etc/ld.so.conf.d/cog.conf && ldconfig
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.0.3/pget" && chmod +x /usr/local/bin/pget
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY . /src
