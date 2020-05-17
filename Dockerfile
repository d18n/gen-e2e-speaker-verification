FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get -y install libsndfile1

RUN pip install jupyter multiprocess scipy webrtcvad librosa visdom matplotlib umap-learn

COPY encoder /workspace/encoder

# Probably not the most secure configuration, just to be clear, but it's convenient
ENTRYPOINT ["jupyter", "notebook", \
    "--allow-root", \
    "--ip=0.0.0.0", \
    "--no-browser", \
    "--NotebookApp.allow_remote_access=True", \
    "--NotebookApp.disable_check_xsrf=True", \
    "--NotebookApp.password=\"sha1:410ae5b662e7:b26e1049b8497f1801d93029f8a4ec8d0618b6e6\"" \
]