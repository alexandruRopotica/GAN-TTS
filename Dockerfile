FROM tensorflow/tensorflow:latest-gpu

RUN mkdir /GANTTS
RUN pip install tensorflow_addons==0.12.0
RUN pip install tensorflow_text

WORKDIR /GANTTS

COPY ./Models .
COPY ./Tests .
COPY ./Utils .
COPY ./Training .

CMD [ "python", "./testNet.py"] 