FROM ubuntu

WORKDIR /opt/frontend-apps

RUN apt update

RUN apt install -y python3

CMD ["/bin/bash"]
