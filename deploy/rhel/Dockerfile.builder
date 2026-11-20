FROM rockylinux:9
RUN dnf install -y python3.12 python3.12-pip python3.12-devel gcc gcc-c++ make zlib libstdc++ && dnf clean all
WORKDIR /bundle
ENTRYPOINT ["bash", "deploy/rhel/build.sh"]
