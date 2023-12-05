FROM ultralytics/ultralytics:latest

#若需要在在院區內透過pip安裝套件則要執行以下兩個步驟
#COPY ../third-party /usr/src/ultralytics/third-party
#RUN mv ./third-party/third-party/ITRIRoot256/ITRIroot256.pem /opt/conda/lib/python3.10/site-packages/certifi/cacert.pem

CMD ["sh", "-c", "python ./engine/main.py"] 
#========[Dev mode]========
#CMD ["/bin/bash"]