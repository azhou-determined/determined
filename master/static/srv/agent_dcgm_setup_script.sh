docker run -d \
  --gpus=all \
  --publish=9400:9400 \
  --name=dgcm-exporter \
  nvidia/dcgm-exporter:2.0.13-2.1.2-ubuntu18.04