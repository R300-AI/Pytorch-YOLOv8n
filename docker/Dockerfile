FROM ultralytics/ultralytics:latest
ENV DATASET_FORMAT=YOLOv8

#To pass the intranet firewall when install something wih python.
COPY ../third-party /usr/src/ultralytics/third-party
RUN mv ./third-party/ITRIRoot256/ITRIroot256.pem /opt/conda/lib/python3.10/site-packages/certifi/cacert.pem

#Using edgevision framework to deal with workflow
COPY ../edgevision /usr/src/ultralytics/edgevision
COPY ../main.py /usr/src/ultralytics/main.py

#========[Runtime mode]========

#ENTRYPOINT ["python", "main.py"]

#========[Dev mode]========

CMD ["/bin/bash"]