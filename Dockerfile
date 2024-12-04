# Use the official PyTorch base image with CUDA 11.7
FROM pytorch/manylinux-cuda117

# Use the official Miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory and home directory to /app/GEMS
WORKDIR /app/GEMS
ENV HOME=/app/GEMS

# Copy all contents of the current directory into the container
COPY . /app/GEMS/

# Ensure the NVIDIA runtime is configured for GPU support
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Install Conda dependencies and create the GEMS environment
RUN conda update -n base -c defaults conda && \
    conda create -n GEMS python=3.10 && \
    echo "source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS" >> ~/.bashrc && \
    conda clean -afy

# Install dependencies one at a time in the GEMS environment
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && conda install -c conda-forge "numpy<2.0"
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && conda install -c conda-forge rdkit
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && conda install -c huggingface transformers
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && pip install ankh
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && conda install biopython
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && conda install pytorch=2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && conda install pyg=*=*cu117 -c pyg
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate GEMS && conda install wandb --channel conda-forge

# Set the default command to start a bash shell with the GEMS environment activated
CMD ["/bin/bash", "-c", "source ~/.bashrc && bash"]
