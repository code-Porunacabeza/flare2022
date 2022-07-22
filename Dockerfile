FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	# we have found python3.7 in base docker
	python3-pip \
	python3-setuptools \
	build-essential \
	&& \
	apt-get clean && \
	python -m pip install --upgrade pip

WORKDIR /workspace
COPY ./ /workspace

RUN pip3 install pip -U
RUN pip3 install -U scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torch==1.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torchvision==0.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["bash", "predict.sh"]
