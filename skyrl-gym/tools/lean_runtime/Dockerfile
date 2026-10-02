FROM jasonchou97/sandbox@sha256:1d7d31be0c23030fbd0c868354eb28ca81266e6db83cfd9046f53d05da631135
USER root
RUN curl -fL https://github.com/leanprover/elan/releases/download/v4.1.2/elan-x86_64-unknown-linux-gnu.tar.gz -o /tmp/elan.tar.gz && tar -xzf /tmp/elan.tar.gz -C /tmp && /tmp/elan-init -y --default-toolchain leanprover/lean4:v4.12.0
ENV PATH="/root/.elan/bin:$PATH"
ENV ELAN_TOOLCHAIN="leanprover/lean4:v4.12.0"
RUN mkdir -p /lean4 && cd /lean4 && lake new my_project && cd my_project && echo 'leanprover/lean4:v4.12.0' > lean-toolchain && echo 'require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "809c3fb3b5c8f5d7dace56e200b426187516535a"' >> lakefile.lean && lake exe cache get && lake build
RUN chmod 755 /root
COPY server.py AuditKernel.lean /audit/
WORKDIR /lean4/my_project
ENTRYPOINT ["python3", "/audit/server.py", "--inspector", "/audit/AuditKernel.lean"]
