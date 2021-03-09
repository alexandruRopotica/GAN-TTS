FROM tensorflow/tensorflow:latest-gpu

RUN mkdir /GANTTS

RUN pip install tensorflow_addons>=0.12.0
RUN pip install transformers
RUN pip install pandas
RUN apt-get install -y libsndfile1
RUN pip install librosa==0.8.0

WORKDIR /GANTTS

COPY . /GANTTS/E2E-GANTTS

ENV PYTHONPATH /GANTTS/E2E-GANTTS


CMD ["python", "./E2E-GANTTS/Tests/testNet.py"]
#CMD ["python", "./E2E-GANTTS/Training/train.py"]
