FROM causal-tracing-exp

RUN apt-get update && apt-get install -y graphviz libgraphviz-dev pkg-config \
    && pip install pygraphviz openai