FROM tensorflow/serving:2.0.0

COPY ./serving_model/ /models/serving_model/
COPY ./model_config.txt /models/
COPY ./start.sh /root/

ENTRYPOINT ["/bin/bash","/root/start.sh"]

