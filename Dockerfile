FROM continuumio/miniconda3

WORKDIR /app

COPY . .
RUN conda env create -f environment.yaml
ENV PATH /opt/conda/envs/control/bin:$PATH

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run ui.py"]
