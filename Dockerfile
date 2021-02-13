FROM tensorflow/tensorflow

# Japanese localization
RUN apt -y update && \
    apt -y upgrade && \
    apt -y install language-pack-ja-base language-pack-ja ibus-mozc && \
    locale-gen ja_JP.UTF-8 && \
    echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Install utils without interactive dialogue
RUN DEBIAN_FRONTEND=noninteractive apt -y install vim

# Install Python Library
RUN python3 -m pip install --upgrade \
    pip \
    matplotlib

# Add User
RUN useradd -m test
USER test

# Placing the program
WORKDIR /home/test
COPY nn-hidden_sigmoid.py .